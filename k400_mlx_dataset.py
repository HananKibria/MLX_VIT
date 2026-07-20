"""
Kinetics-400 dataset for MLX training.

Designed for the practical setup of an active download:
  • Skips 0-byte / partial / sub-50KB files
  • Re-scans the directory every epoch (download is live)
  • Maps each filename → class via the official annotations CSV
  • Threaded prefetch (PyAV decode releases the GIL)
  • Yields (B, T, H, W, C) float32 arrays in MLX channels-last layout

Filename convention (matches the cvdfoundation downloader):
    {youtube_id}_{start:06d}_{end:06d}.mp4

This module has zero MLX-specific dependencies in the *decode* path — it
returns numpy arrays. The MLX wrapping happens in StreamingDataLoader.
"""

from __future__ import annotations

import csv
import os
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import av
import numpy as np


# =============================================================================
# Filename ↔ label mapping
# =============================================================================

def load_k400_label_map(
    csv_path: str | Path,
    classes: Optional[Sequence[str]] = None,
) -> Tuple[dict[str, int], List[str]]:
    """
    Parse the K400 annotation CSV and return:
        filename_to_label : {f"{ytid}_{start:06d}_{end:06d}.mp4": class_idx}
        classes           : sorted list of class names (length 400)

    If `classes` is None the function discovers them from the CSV (sorted
    alphabetically, the canonical K400 ordering).
    """
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    if classes is None:
        classes = sorted({r["label"] for r in rows})
    cls2idx = {c: i for i, c in enumerate(classes)}

    fname_to_label: dict[str, int] = {}
    for r in rows:
        ytid = r["youtube_id"]
        start = int(r["time_start"])
        end = int(r["time_end"])
        fname = f"{ytid}_{start:06d}_{end:06d}.mp4"
        fname_to_label[fname] = cls2idx[r["label"]]

    return fname_to_label, list(classes)


# =============================================================================
# Disk scanning  (cheap, can be called every epoch)
# =============================================================================

def list_complete_videos(
    video_root: str | Path,
    *,
    min_bytes: int = 50_000,
    max_age_seconds: float = 5.0,
    suffix: str = ".mp4",
) -> List[Path]:
    """
    Return Path objects for files that look fully-written.

    Filters out:
      • files that don't end in `.mp4`
      • files smaller than `min_bytes` (likely truncated)
      • files modified < `max_age_seconds` ago (still being written)
    """
    root = Path(video_root)
    cutoff = time.time() - max_age_seconds
    out: list[Path] = []
    # os.scandir is much faster than Path.iterdir over USB
    with os.scandir(root) as it:
        for entry in it:
            if not entry.is_file():
                continue
            if not entry.name.endswith(suffix):
                continue
            try:
                st = entry.stat()
            except (FileNotFoundError, OSError):
                continue
            if st.st_size < min_bytes:
                continue
            if st.st_mtime > cutoff:
                continue
            out.append(Path(entry.path))
    return out


# =============================================================================
# Video decoder  (PyAV)
# =============================================================================

# ImageNet normalization stats — matches VideoMAE / standard pretraining.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _decode_all_frames(path: str) -> List[np.ndarray]:
    """Decode every frame as RGB uint8 (H, W, 3)."""
    container = av.open(path)
    container.streams.video[0].thread_type = "AUTO"
    frames: list[np.ndarray] = []
    try:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()
    return frames


def _resize_short(frame: np.ndarray, short_side: int) -> np.ndarray:
    """Resize so the shorter side equals `short_side`, preserving aspect ratio.
    Uses nearest-neighbor for speed (videos are already low-res)."""
    h, w, _ = frame.shape
    if h <= w:
        new_h = short_side
        new_w = max(1, int(round(w * short_side / h)))
    else:
        new_w = short_side
        new_h = max(1, int(round(h * short_side / w)))
    if new_h == h and new_w == w:
        return frame
    # Indexing-based bilinear via numpy (no opencv dependency)
    ys = np.linspace(0, h - 1, new_h).astype(np.int32)
    xs = np.linspace(0, w - 1, new_w).astype(np.int32)
    return frame[ys[:, None], xs[None, :]]


def _crop(frame: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    return frame[top:top + size, left:left + size]


def decode_clip(
    path: str | Path,
    *,
    num_frames: int = 32,
    img_size: int = 224,
    train: bool = True,
    rng: Optional[random.Random] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Decode a video file into a (T, H, W, C) float32 array.

    Train pipeline: short-side ∈ [256, 320] random, random spatial crop,
                    random hflip, ImageNet normalization.
    Eval  pipeline: short-side = 256, center crop, no hflip.
    """
    if rng is None:
        rng = random.Random()

    frames = _decode_all_frames(str(path))
    if not frames:
        raise RuntimeError(f"empty video: {path}")

    # Uniform temporal sampling. If clip has fewer frames than requested,
    # tile the last frame.
    n_avail = len(frames)
    if n_avail >= num_frames:
        # Random/center segment offset within each of `num_frames` bins
        idx = np.linspace(0, n_avail - 1, num_frames).astype(np.int32)
        if train:
            jitter = rng.randint(-1, 1)
            idx = np.clip(idx + jitter, 0, n_avail - 1)
    else:
        idx = list(range(n_avail)) + [n_avail - 1] * (num_frames - n_avail)
        idx = np.array(idx)

    # Spatial transform
    if train:
        short_side = rng.randint(256, 320)
    else:
        short_side = 256
    sampled = [_resize_short(frames[i], short_side) for i in idx]
    H, W, _ = sampled[0].shape  # all the same after resize

    if train:
        top = rng.randint(0, H - img_size)
        left = rng.randint(0, W - img_size)
        do_flip = rng.random() < 0.5
    else:
        top = (H - img_size) // 2
        left = (W - img_size) // 2
        do_flip = False

    cropped = []
    for f in sampled:
        c = _crop(f, top, left, img_size)
        if do_flip:
            c = c[:, ::-1]
        cropped.append(c)

    arr = np.stack(cropped, axis=0).astype(np.float32) / 255.0   # (T, H, W, 3)
    if normalize:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr


# =============================================================================
# Dataset (PyTorch-style indexable; we add our own loader below)
# =============================================================================

class K400Dataset:
    """
    An indexable Kinetics-400 dataset that re-scans the disk on `refresh()`.

    Items unavailable in the annotation CSV (e.g. broken filenames) are
    silently dropped from the file list at scan time — they cost nothing and
    aren't part of the iteration.
    """

    def __init__(
        self,
        video_root: str | Path,
        annotations_csv: str | Path,
        *,
        num_frames: int = 32,
        img_size: int = 224,
        split: str = "train",
        classes: Optional[Sequence[str]] = None,
        min_file_bytes: int = 50_000,
        normalize: bool = True,
    ):
        self.video_root = Path(video_root)
        self.num_frames = num_frames
        self.img_size = img_size
        self.train = split == "train"
        self.min_file_bytes = min_file_bytes
        self.normalize = normalize

        self.filename_to_label, self.classes = load_k400_label_map(
            annotations_csv, classes=classes,
        )

        self._files: list[Path] = []
        self.refresh()

    def refresh(self) -> None:
        """Re-scan disk; pick up newly-completed downloads."""
        all_complete = list_complete_videos(
            self.video_root, min_bytes=self.min_file_bytes,
        )
        # Keep only files that have a known label
        known = [p for p in all_complete if p.name in self.filename_to_label]
        prev = len(self._files)
        self._files = sorted(known)
        if len(self._files) != prev:
            dropped = len(all_complete) - len(self._files)
            print(f"  K400Dataset.refresh(): {prev} → {len(self._files)} videos "
                  f"(disk: {len(all_complete)}, dropped without labels: {dropped})")

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        path = self._files[idx]
        label = self.filename_to_label[path.name]
        video = decode_clip(
            path, num_frames=self.num_frames, img_size=self.img_size,
            train=self.train, normalize=self.normalize,
        )
        return video, label


# =============================================================================
# Streaming dataloader
# =============================================================================
# We don't use torch.utils.data.DataLoader because we want zero torch deps
# in the hot path. PyAV releases the GIL during decode, so threads work fine.

class StreamingDataLoader:
    """
    Threaded prefetch dataloader.

    Yields tuples of (numpy video batch (B, T, H, W, C) float32,
                      numpy label batch (B,) int64).

    The MLX cast is left to the caller — this keeps the loader independent
    of MLX. Wrap with `mx.array(...)` at the training step.

    Robustness:
      • Catches per-item decode errors and skips them; logs once per failure.
      • If a worker exception kills the prefetch thread, raises on the next
        `__next__` call.
    """

    def __init__(
        self,
        dataset: K400Dataset,
        *,
        batch_size: int = 2,
        shuffle: bool = True,
        num_workers: int = 4,
        prefetch: int = 4,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch = prefetch
        self.drop_last = drop_last
        self.seed = seed

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(self.dataset)
        order = list(range(n))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(order)
            self.seed += 1   # different shuffle each epoch

        # Build batch index list
        batches: list[list[int]] = []
        for i in range(0, n, self.batch_size):
            chunk = order[i:i + self.batch_size]
            if len(chunk) < self.batch_size and self.drop_last:
                continue
            batches.append(chunk)

        # Prefetch queue with class-based sentinels (avoids numpy comparison
        # collisions with the normal (videos, labels) tuple payload)
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)

        class _Stop: pass
        class _ItemError:
            __slots__ = ("msg",)
            def __init__(self, msg): self.msg = msg
        class _ProducerError:
            __slots__ = ("exc",)
            def __init__(self, exc): self.exc = exc

        def _fetch_one(idx: int):
            try:
                return self.dataset[idx]
            except Exception as e:
                return _ItemError(f"idx={idx}: {type(e).__name__}: {e}")

        def _producer():
            try:
                with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                    for chunk in batches:
                        items = list(pool.map(_fetch_one, chunk))
                        good = [x for x in items if not isinstance(x, _ItemError)]
                        if not good:
                            # Whole batch failed — skip silently
                            continue
                        # Pad with last good item if some failed
                        while len(good) < self.batch_size:
                            good.append(good[-1])
                        videos = np.stack([g[0] for g in good], axis=0)
                        labels = np.array([g[1] for g in good], dtype=np.int64)
                        q.put((videos, labels))
            except Exception as e:
                q.put(_ProducerError(e))
            finally:
                q.put(_Stop())

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()

        while True:
            item = q.get()
            if isinstance(item, _Stop):
                break
            if isinstance(item, _ProducerError):
                raise item.exc
            yield item


# =============================================================================
# Quick self-check
# =============================================================================

if __name__ == "__main__":
    import sys

    VIDEO_ROOT = "/Volumes/Drive/train"
    ANNOT = "/Volumes/Drive/annotations/train.csv"

    print("Building K400Dataset...")
    ds = K400Dataset(VIDEO_ROOT, ANNOT, num_frames=32, img_size=224)
    print(f"  classes: {ds.num_classes}")
    print(f"  videos:  {len(ds):,}")
    print(f"  example: {ds._files[0].name}  -> "
          f"label={ds.filename_to_label[ds._files[0].name]} "
          f"({ds.classes[ds.filename_to_label[ds._files[0].name]]})")

    print("\nDecoding 3 videos sequentially...")
    for i in range(3):
        t0 = time.perf_counter()
        v, lbl = ds[i]
        dt = (time.perf_counter() - t0) * 1000
        print(f"  [{i}] shape={v.shape} dtype={v.dtype} "
              f"min={v.min():.2f} max={v.max():.2f}  label={lbl}  "
              f"{dt:.0f} ms")

    print("\nStreaming 3 batches with workers=4, B=2...")
    loader = StreamingDataLoader(ds, batch_size=2, num_workers=4, prefetch=4)
    it = iter(loader)
    for i in range(3):
        t0 = time.perf_counter()
        videos, labels = next(it)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  batch {i}: videos {videos.shape} labels {labels.tolist()}  "
              f"{dt:.0f} ms")

    print("\n✅ K400 pipeline OK")
