"""
Something-Something V2 (SSv2) dataset for MLX training.

Layout:
    /Volumes/Drive/data/ssv2/
    ├── 20bn-something-something-v2/{id}.webm     (220K videos)
    └── annotations/labels/
        ├── labels.json         "Template Without Brackets" -> "0".."173"  (174 classes)
        ├── train.json          [{id: "78687", template: "Holding [something] ...", ...}, ...]
        └── validation.json     same schema

Key differences from K400:
  • Decode .webm (PyAV handles VP8/VP9 fine)
  • Class lookup: strip brackets from template, then look up in labels.json
  • Direction-sensitive: NO horizontal flip during training
  • Stronger temporal jitter is OK (32 frames over a 2-3s clip)

Yields (B, T, H, W, C) float32 numpy arrays, ImageNet-normalized.
The MLX cast happens at the training step.
"""

from __future__ import annotations

import json
import math
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
# Annotations
# =============================================================================

def _strip_brackets(template: str) -> str:
    """'Holding [something] next to [something]' -> 'Holding something next to something'.

    SSv2's train.json / validation.json templates use [bracket] placeholders;
    labels.json keys are the same string with brackets removed.
    """
    return template.replace("[", "").replace("]", "")


def load_ssv2_label_map(
    annotations_dir: str | Path,
    split: str = "train",
) -> Tuple[dict[str, int], List[str]]:
    """
    Parse SSv2 annotation JSONs and return:
        id_to_label : {video_id_string: int_class}
        classes     : list[str] of length 174 (canonical order from labels.json,
                       indexed by integer class)
    """
    ann_dir = Path(annotations_dir)
    labels = json.load(open(ann_dir / "labels.json"))
    # labels.json: {"Template": "0", ...}  — values are str integers
    cls2idx = {k: int(v) for k, v in labels.items()}
    n = len(cls2idx)
    classes = [""] * n
    for k, i in cls2idx.items():
        classes[i] = k

    if split == "train":
        recs = json.load(open(ann_dir / "train.json"))
    elif split in ("val", "validation"):
        recs = json.load(open(ann_dir / "validation.json"))
    else:
        raise ValueError(f"unknown split: {split}")

    id_to_label: dict[str, int] = {}
    n_unmapped = 0
    for r in recs:
        key = _strip_brackets(r["template"])
        if key not in cls2idx:
            n_unmapped += 1
            continue
        id_to_label[str(r["id"])] = cls2idx[key]
    if n_unmapped:
        print(f"  ⚠ {n_unmapped} {split} entries had templates not in labels.json")
    return id_to_label, classes


# =============================================================================
# Disk scanning
# =============================================================================

def list_complete_videos(
    video_root: str | Path,
    *,
    min_bytes: int = 8_000,
    suffix: str = ".webm",
) -> List[Path]:
    """Return Path objects for files that look fully-written.

    SSv2 webms can be small (some clips are <30KB), so the threshold is much
    lower than K400's. We don't need the recency check — SSv2 is shipped as a
    single archive, not an active download.
    """
    root = Path(video_root)
    out: list[Path] = []
    with os.scandir(root) as it:
        for entry in it:
            if not entry.is_file() or not entry.name.endswith(suffix):
                continue
            try:
                if entry.stat().st_size < min_bytes:
                    continue
            except (FileNotFoundError, OSError):
                continue
            out.append(Path(entry.path))
    return out


# =============================================================================
# Video decoder (PyAV)
# =============================================================================

# ImageNet stats — matches VideoMAE pretraining.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _decode_all_frames(path: str) -> List[np.ndarray]:
    """Decode every frame as RGB uint8 (H, W, 3). PyAV handles webm fine."""
    container = av.open(path)
    container.streams.video[0].thread_type = "AUTO"
    frames: list[np.ndarray] = []
    try:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()
    return frames


def _resize(frame: np.ndarray, new_h: int, new_w: int,
            mode: str = "bilinear") -> np.ndarray:
    """Resize a single frame. Bilinear for train (smoother augmentation),
    nearest for eval (faster, same-quality at center crop)."""
    h, w, _ = frame.shape
    if h == new_h and w == new_w:
        return frame
    if mode == "nearest":
        ys = np.linspace(0, h - 1, new_h).astype(np.int32)
        xs = np.linspace(0, w - 1, new_w).astype(np.int32)
        return frame[ys[:, None], xs[None, :]]
    # Bilinear via meshgrid + bilinear blend
    ys = np.linspace(0, h - 1, new_h).astype(np.float32)
    xs = np.linspace(0, w - 1, new_w).astype(np.float32)
    y0 = np.floor(ys).astype(np.int32);  y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(xs).astype(np.int32);  x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0).astype(np.float32)[:, None, None]
    wx = (xs - x0).astype(np.float32)[None, :, None]
    f = frame.astype(np.float32)
    a = f[y0[:, None], x0[None, :]];  b = f[y0[:, None], x1[None, :]]
    c = f[y1[:, None], x0[None, :]];  d = f[y1[:, None], x1[None, :]]
    out = (1 - wy) * ((1 - wx) * a + wx * b) + wy * ((1 - wx) * c + wx * d)
    return out.astype(frame.dtype)


def _random_resized_crop_params(
    H: int, W: int, *,
    scale: Tuple[float, float] = (0.5, 1.0),
    ratio: Tuple[float, float] = (0.75, 1.333),
    rng: random.Random,
    n_attempts: int = 10,
) -> Tuple[int, int, int, int]:
    """Sample (top, left, crop_h, crop_w) for torchvision-equivalent
    `RandomResizedCrop`. Falls back to a centre square on failure."""
    area = H * W
    log_ratio = (math.log(ratio[0]), math.log(ratio[1])) if False else \
                (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(n_attempts):
        target_area = rng.uniform(*scale) * area
        aspect = math.exp(rng.uniform(*log_ratio))
        crop_w = int(round(math.sqrt(target_area * aspect)))
        crop_h = int(round(math.sqrt(target_area / aspect)))
        if 0 < crop_w <= W and 0 < crop_h <= H:
            top = rng.randint(0, H - crop_h)
            left = rng.randint(0, W - crop_w)
            return top, left, crop_h, crop_w
    # Fallback: centre square
    side = min(H, W)
    return (H - side) // 2, (W - side) // 2, side, side


def _color_jitter_params(brightness: float, contrast: float,
                         saturation: float, hue: float,
                         rng: random.Random) -> dict:
    """Sample multiplicative b/c/s factors and additive hue (per-clip;
    re-used for every frame so frames stay temporally consistent)."""
    return {
        "brightness": rng.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness),
        "contrast":   rng.uniform(max(0.0, 1.0 - contrast),   1.0 + contrast),
        "saturation": rng.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation),
        "hue":        rng.uniform(-hue, hue),
    }


# RGB→YIQ/YIQ→RGB matrices (used for hue rotation; matches torchvision exactly).
_RGB2YIQ = np.array([
    [0.299,  0.587,  0.114],
    [0.596, -0.274, -0.322],
    [0.211, -0.523,  0.312],
], dtype=np.float32).T  # shape (3, 3) — applied as `frame @ M`
_YIQ2RGB = np.array([
    [1.0,  0.956,  0.621],
    [1.0, -0.272, -0.647],
    [1.0, -1.106,  1.703],
], dtype=np.float32).T


def _apply_color_jitter(frame: np.ndarray, p: dict) -> np.ndarray:
    """Apply b/c/s/h (matches torchvision ColorJitter ordering: b → c → s → h).
    Operates on float32 [0, 1]. Frame shape (H, W, 3)."""
    out = frame
    # Brightness
    out = np.clip(out * p["brightness"], 0.0, 1.0)
    # Contrast: blend with per-channel mean (luminance)
    luma = (0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]).mean()
    out = np.clip((out - luma) * p["contrast"] + luma, 0.0, 1.0)
    # Saturation: blend with grayscale
    gray = 0.299 * out[..., 0:1] + 0.587 * out[..., 1:2] + 0.114 * out[..., 2:3]
    out = np.clip(gray + (out - gray) * p["saturation"], 0.0, 1.0)
    # Hue: rotate in YIQ space
    if abs(p["hue"]) > 1e-6:
        yiq = out @ _RGB2YIQ                      # (H, W, 3)
        theta = p["hue"] * 2 * math.pi
        cs, sn = math.cos(theta), math.sin(theta)
        i_new = yiq[..., 1] * cs - yiq[..., 2] * sn
        q_new = yiq[..., 1] * sn + yiq[..., 2] * cs
        yiq[..., 1] = i_new
        yiq[..., 2] = q_new
        out = np.clip(yiq @ _YIQ2RGB, 0.0, 1.0)
    return out


def decode_clip(
    path: str | Path,
    *,
    num_frames: int = 32,
    img_size: int = 224,
    train: bool = True,
    rng: Optional[random.Random] = None,
    normalize: bool = True,
    use_hflip: bool = False,
    # RandomResizedCrop parameters (matches PyTorch notebook)
    rrc_scale: Tuple[float, float] = (0.5, 1.0),
    rrc_ratio: Tuple[float, float] = (0.75, 1.333),
    # ColorJitter parameters (matches PyTorch notebook)
    color_jitter_bcs: float = 0.4,
    color_jitter_hue: float = 0.1,
    # RandomErasing (matches PyTorch notebook's `T.RandomErasing(p=reprob)`).
    # Applied AFTER normalization, so the fill is in normalized-space (0.0 ≈
    # the per-channel ImageNet mean once normalized). The same rectangle is
    # used on every frame in the clip so temporal consistency is preserved.
    random_erase_prob: float = 0.0,
    random_erase_scale: Tuple[float, float] = (0.02, 1 / 3),
    random_erase_ratio: Tuple[float, float] = (0.3, 3.3),
) -> np.ndarray:
    """Decode a video file into a (T, H, W, C) float32 array.

    Train pipeline (matches the notebook's `VideoTransform`):
      • Random sample one frame per uniform segment.
      • RandomResizedCrop with shared (top, left, h, w) across all frames.
      • ColorJitter (brightness, contrast, saturation, hue) with shared params
        across all frames in the clip — temporally consistent.
      • Optional hflip (OFF by default for SSv2).
      • ImageNet normalization.

    Eval pipeline:
      • Uniform segment-centre sampling.
      • Resize short-side to int(img_size * 256/224), centre crop, normalize.
    """
    if rng is None:
        rng = random.Random()

    frames = _decode_all_frames(str(path))
    if not frames:
        raise RuntimeError(f"empty video: {path}")
    n_avail = len(frames)

    # ------- Temporal sampling -------
    # Train: random frame per segment; Eval: segment-centre.
    if n_avail >= num_frames:
        seg = n_avail / num_frames
        if train:
            idx = np.array([rng.randint(int(i * seg),
                                        max(int(i * seg), int((i + 1) * seg) - 1))
                            for i in range(num_frames)], dtype=np.int32)
        else:
            idx = np.array([min(int((i + 0.5) * seg), n_avail - 1)
                            for i in range(num_frames)], dtype=np.int32)
        idx = np.clip(idx, 0, n_avail - 1)
    else:
        idx = np.array(list(range(n_avail)) + [n_avail - 1] * (num_frames - n_avail),
                       dtype=np.int32)
    sampled = [frames[i] for i in idx]
    H0, W0, _ = sampled[0].shape

    # ------- Spatial pipeline -------
    if train:
        # Random resized crop — sample once for the whole clip
        top, left, crop_h, crop_w = _random_resized_crop_params(
            H0, W0, scale=rrc_scale, ratio=rrc_ratio, rng=rng,
        )
        do_flip = use_hflip and (rng.random() < 0.5)
        jitter_p = _color_jitter_params(
            color_jitter_bcs, color_jitter_bcs, color_jitter_bcs,
            color_jitter_hue, rng=rng,
        )

        out_frames = []
        for f in sampled:
            # Crop → resize to img_size
            c = f[top:top + crop_h, left:left + crop_w]
            c = _resize(c, img_size, img_size, mode="bilinear")
            c = c.astype(np.float32) / 255.0
            c = _apply_color_jitter(c, jitter_p)
            if do_flip:
                c = c[:, ::-1]
            out_frames.append(c)
    else:
        # Eval: resize short-side to int(img_size*256/224) then centre crop
        target_short = int(img_size * 256 / 224)
        if H0 <= W0:
            new_h, new_w = target_short, max(1, int(round(W0 * target_short / H0)))
        else:
            new_w, new_h = target_short, max(1, int(round(H0 * target_short / W0)))
        top = (new_h - img_size) // 2
        left = (new_w - img_size) // 2
        out_frames = []
        for f in sampled:
            r = _resize(f, new_h, new_w, mode="bilinear")
            c = r[top:top + img_size, left:left + img_size]
            out_frames.append(c.astype(np.float32) / 255.0)

    arr = np.stack(out_frames, axis=0)
    if normalize:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    # RandomErasing — train-only. Per-clip rectangle, same on all frames.
    if train and random_erase_prob > 0.0 and rng.random() < random_erase_prob:
        T_, H_, W_, _ = arr.shape
        area = H_ * W_
        for _ in range(10):
            target_area = rng.uniform(*random_erase_scale) * area
            log_ratio = (math.log(random_erase_ratio[0]),
                         math.log(random_erase_ratio[1]))
            aspect = math.exp(rng.uniform(*log_ratio))
            h = int(round(math.sqrt(target_area * aspect)))
            w = int(round(math.sqrt(target_area / aspect)))
            if 0 < h < H_ and 0 < w < W_:
                top = rng.randint(0, H_ - h)
                left = rng.randint(0, W_ - w)
                # Fill with zero in normalized space → ImageNet-mean color
                # in pixel space. Matches torchvision's default
                # `value="random"` is more diverse, but 0 is the documented
                # default and the most common in practice.
                arr[:, top:top + h, left:left + w, :] = 0.0
                break

    return arr.astype(np.float32)


# =============================================================================
# Dataset
# =============================================================================

class SSv2Dataset:
    """Indexable SSv2 dataset.

    Uses the annotation JSON as the source of truth for the file list — we
    don't scan the video directory because (a) SSv2 ships as a static archive,
    not an active download, and (b) `os.scandir` over 220K files on USB
    storage takes minutes for no benefit.

    File paths are built as `<video_root>/<video_id>.webm`. If a file is
    missing or unreadable at decode time, the StreamingDataLoader's per-item
    error handler skips it.

    Set `verify_existence=True` to do a cheap stat() check on every file
    during __init__ and prune missing IDs upfront — useful if you suspect
    the archive is incomplete. Default False (fast init).
    """

    def __init__(
        self,
        video_root: str | Path,
        annotations_dir: str | Path,
        *,
        num_frames: int = 32,
        img_size: int = 224,
        split: str = "train",
        use_hflip: bool = False,
        normalize: bool = True,
        verify_existence: bool = False,
        suffix: str = ".webm",
        # RandomErasing: configured probability + the epoch from which it
        # turns on. Before then it stays at 0.0. The trainer flips this on
        # by calling `dataset.set_epoch(e)` at the start of each epoch.
        random_erase_prob: float = 0.0,
        random_erase_start_epoch: int = 0,
    ):
        self.video_root = Path(video_root)
        self.num_frames = num_frames
        self.img_size = img_size
        self.train = split == "train"
        self.use_hflip = use_hflip
        self.normalize = normalize
        self._suffix = suffix
        # RE is gated by epoch — see set_epoch() and __getitem__().
        self._random_erase_prob_cfg = random_erase_prob
        self._random_erase_start_epoch = random_erase_start_epoch
        self._random_erase_prob_active = 0.0  # off until set_epoch unlocks it

        self.id_to_label, self.classes = load_ssv2_label_map(annotations_dir, split=split)

        # Build file list directly from annotation IDs — no disk scan.
        ids_sorted = sorted(self.id_to_label.keys())
        self._files: list[Path] = [self.video_root / f"{vid}{suffix}" for vid in ids_sorted]

        if verify_existence:
            t0 = time.perf_counter()
            present = [p for p in self._files if p.exists()]
            n_missing = len(self._files) - len(present)
            self._files = present
            dt = time.perf_counter() - t0
            print(f"  SSv2Dataset[{split}]: {len(self._files):,} videos "
                  f"({n_missing} missing, verified in {dt:.1f}s)")
        else:
            print(f"  SSv2Dataset[{split}]: {len(self._files):,} videos "
                  f"(from annotations; existence not verified)")

    def refresh(self) -> None:
        """No-op — annotations are static. Kept for API compatibility with
        the K400 dataset (where refresh re-scans for new downloads)."""
        return

    def set_epoch(self, epoch: int) -> None:
        """Activate epoch-gated augmentations. Called by the trainer at the
        start of each epoch; subsequent __getitem__ calls use whatever
        augmentations are now unlocked."""
        if (self.train
                and self._random_erase_prob_cfg > 0.0
                and epoch >= self._random_erase_start_epoch):
            self._random_erase_prob_active = self._random_erase_prob_cfg
        else:
            self._random_erase_prob_active = 0.0

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        path = self._files[idx]
        label = self.id_to_label[path.stem]
        # Try .webm first; if missing fall back to .mp4 (some SSv2 mirrors
        # reuse the original .mp4 archive). Errors propagate to the loader's
        # per-item handler which skips the failed sample.
        if not path.exists() and self._suffix == ".webm":
            mp4 = path.with_suffix(".mp4")
            if mp4.exists():
                path = mp4
        video = decode_clip(
            path, num_frames=self.num_frames, img_size=self.img_size,
            train=self.train, normalize=self.normalize, use_hflip=self.use_hflip,
            random_erase_prob=self._random_erase_prob_active,
        )
        return video, label


# =============================================================================
# Streaming dataloader (threaded prefetch)
# =============================================================================

class StreamingDataLoader:
    """Threaded prefetch dataloader. Yields (videos (B,T,H,W,C) f32, labels (B,) i64)."""

    def __init__(
        self,
        dataset: SSv2Dataset,
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
            self.seed += 1

        batches: list[list[int]] = []
        for i in range(0, n, self.batch_size):
            chunk = order[i:i + self.batch_size]
            if len(chunk) < self.batch_size and self.drop_last:
                continue
            batches.append(chunk)

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
                            continue
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
# Self-check
# =============================================================================

if __name__ == "__main__":
    VIDEO_ROOT = "/Volumes/Drive/data/ssv2/20bn-something-something-v2"
    ANN_DIR    = "/Volumes/Drive/data/ssv2/annotations/labels"

    print("Building SSv2 train dataset...")
    ds = SSv2Dataset(VIDEO_ROOT, ANN_DIR, num_frames=32, img_size=224, split="train")
    print(f"  classes: {ds.num_classes}")
    print(f"  videos:  {len(ds):,}")
    sample_path = ds._files[0]
    print(f"  example: {sample_path.name} -> "
          f"label={ds.id_to_label[sample_path.stem]} "
          f"({ds.classes[ds.id_to_label[sample_path.stem]]})")

    print("\nDecoding 3 videos sequentially...")
    for i in range(3):
        t0 = time.perf_counter()
        v, lbl = ds[i]
        dt = (time.perf_counter() - t0) * 1000
        print(f"  [{i}] shape={v.shape} dtype={v.dtype} "
              f"min={v.min():.2f} max={v.max():.2f}  label={lbl}  {dt:.0f} ms")

    print("\nStreaming 3 batches with workers=4, B=2...")
    loader = StreamingDataLoader(ds, batch_size=2, num_workers=4, prefetch=4)
    it = iter(loader)
    for i in range(3):
        t0 = time.perf_counter()
        videos, labels = next(it)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  batch {i}: videos {videos.shape} labels {labels.tolist()}  {dt:.0f} ms")

    print("\nBuilding val dataset (sanity)...")
    val_ds = SSv2Dataset(VIDEO_ROOT, ANN_DIR, num_frames=32, img_size=224, split="validation")
    print(f"  val videos: {len(val_ds):,}")
    print("\n✅ SSv2 pipeline OK")
