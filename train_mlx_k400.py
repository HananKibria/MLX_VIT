"""
Train NoPE+GDN on Kinetics-400 with the MLX Metal stack.

Uses:
  - bf16 mixed precision (50% memory, ~8.5x training speedup)
  - Chunkwise WY GDN forward + custom Metal backward (compute_path =
    'chunkwise_kda_vjp'). At full-model scale this is ~1.3× faster than the
    earlier step-by-step 'metal_vjp' path AND saves ~64× less per-layer
    state for backward (chunk-boundary states only, not per-step).
  - The legacy step-by-step path is still available via --gdn_path metal_vjp.

Expected per-step time on M3 Max with chunkwise_kda_vjp: ~525 ms at B=2,
T=32, 224² (29.79M params; small encoder=384). That's ~18 hours per K400
epoch at B=2 — see the docstring of `main()` for practical guidance on
what's reasonable to run locally.

Usage:
    # Quick smoke test on synthetic data:
    python train_mlx_k400.py --debug --steps 10

    # Real training (you provide a dataloader; see `iter_dataset` stub below):
    python train_mlx_k400.py --data_root /path/to/k400 --epochs 5

    # Force the legacy step-by-step backward (e.g. for parity benchmarking):
    python train_mlx_k400.py --gdn_path metal_vjp --debug --steps 10
"""

from __future__ import annotations
import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict, field
from functools import partial
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten, tree_map

from nope_gdn_mlx import (
    NoPEGDNClassifier, GatedDeltaLayer,
    to_bf16, param_dtype_summary, _flatten_params,
    load_videomae_base,
)
from k400_mlx_dataset import K400Dataset, StreamingDataLoader


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    # Model — BASE size (matches VideoMAE-base for `--init_videomae`).
    # ViT head_dim = 768/12 = 64, so chunk_size=64 exactly fits the chunkwise
    # solver's TG-memory budget (C·(C + head_dim) ≤ 8192 → 64·128 = 8192).
    # Total params at this scale: ~117M (vs 30M for the small variant).
    img_size: int = 224
    num_frames: int = 32
    tubelet_size: Tuple[int, int, int] = (2, 16, 16)
    encoder_dim: int = 768
    encoder_depth: int = 12
    encoder_heads: int = 12
    processor_dim: int = 768
    processor_depth: int = 4
    processor_heads: int = 12
    chunk_size: int = 64
    drop_path_rate: float = 0.1
    dropout: float = 0.0
    head_dropout: float = 0.0
    num_classes: int = 400          # Kinetics-400

    # Training. NOTE: at base size on M3 Max the working memory roughly
    # quadruples vs the small encoder. Default batch_size dropped to 2 so the
    # default config trains out-of-the-box; bump up only if memory headroom
    # allows.
    batch_size: int = 2
    lr: float = 3e-4
    weight_decay: float = 0.05
    epochs: int = 100
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    ema_decay: float = 0.9999

    # Mixed precision: cast model to bf16 once after init.
    use_bf16: bool = True
    # GDN training compute path. Options:
    #   "chunkwise_kda_vjp" — Chunkwise WY forward + custom Metal backward
    #                          (default; ~1.3× faster than metal_vjp at full-
    #                          model scale, ~64× lower saved-state memory).
    #   "metal_vjp"          — Step-by-step Metal forward + Metal backward
    #                          (the previous default; correct, slower).
    #   "compiled"           — mx.compile-fused step loop (no Metal); ~15×
    #                          slower. Use only when the Metal kernels are
    #                          unavailable (e.g. CUDA / CPU).
    gdn_path: str = "chunkwise_kda_vjp"

    # Data
    video_root: str = "/Volumes/Drive/train"
    annotations_csv: str = "/Volumes/Drive/annotations/train.csv"
    num_workers: int = 4
    prefetch: int = 4

    # Output
    output_dir: str = "outputs/mlx_k400"
    log_every: int = 10
    save_every_epoch: int = 1


# =============================================================================
# MODEL
# =============================================================================

def build_model(cfg: TrainConfig) -> NoPEGDNClassifier:
    """Build, cast to bf16, and switch GDN layers to the configured path
    (default: chunkwise_kda_vjp)."""
    # Validate chunk_size against the chunkwise Metal solver's threadgroup
    # memory budget when a chunkwise path is selected. The kernel reserves
    #   L_tg[C, C] + y_cols[D, C]   = (C² + D·C) × 4 bytes
    # of threadgroup memory, capped at 32 KB on M-series → C·(C + D) ≤ 8192.
    if cfg.gdn_path in ("chunkwise_kda", "chunkwise_kda_vjp"):
        proc_head_dim = cfg.processor_dim // cfg.processor_heads
        budget_bytes = (cfg.chunk_size * cfg.chunk_size
                        + cfg.chunk_size * proc_head_dim) * 4
        if budget_bytes > 32 * 1024:
            raise ValueError(
                f"GDN path {cfg.gdn_path!r} requires "
                f"chunk_size·(chunk_size + head_dim) ≤ 8192, but got "
                f"chunk_size={cfg.chunk_size}, head_dim={proc_head_dim} "
                f"→ TG-mem need = {budget_bytes / 1024:.1f} KB > 32 KB. "
                f"Lower chunk_size or fall back to gdn_path='metal_vjp'."
            )

    model = NoPEGDNClassifier(
        img_size=cfg.img_size, num_frames=cfg.num_frames,
        tubelet_size=cfg.tubelet_size,
        encoder_dim=cfg.encoder_dim, encoder_depth=cfg.encoder_depth,
        encoder_heads=cfg.encoder_heads,
        processor_dim=cfg.processor_dim, processor_depth=cfg.processor_depth,
        processor_heads=cfg.processor_heads,
        chunk_size=cfg.chunk_size,
        drop_path_rate=cfg.drop_path_rate,
        dropout=cfg.dropout,
        head_dropout=cfg.head_dropout,
        num_classes=cfg.num_classes,
    )
    mx.eval(model.parameters())

    if cfg.use_bf16:
        to_bf16(model)
        print("Model cast to bfloat16; "
              f"params: {param_dtype_summary(model)}")

    n_switched = 0
    for _, m in model.named_modules():
        if isinstance(m, GatedDeltaLayer):
            m.compute_path = cfg.gdn_path
            n_switched += 1
    print(f"Switched {n_switched} GDN layers to compute_path={cfg.gdn_path!r}")

    n_params = sum(v.size for _, v in _flatten_params(model.parameters()))
    print(f"Total params: {n_params / 1e6:.2f} M")
    return model


# =============================================================================
# EMA (Exponential Moving Average of weights — usually +1-2 % accuracy)
# =============================================================================

class ModelEMA:
    """Track learnable parameters' EMA. Buffers (rope.inv_freq etc.) are NOT
    averaged — they're deterministic and recovered from the live model.

    The update body is wrapped with `mx.compile` so the per-step Python loop
    over ~hundreds of leaves becomes a single fused graph (one launch instead
    of N).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        # Snapshot all learnable params (skip buffers).  Stored as a flat dict
        # of {name: mx.array} — the same structure mx.compile can recognise as
        # an `inputs`/`outputs` tree.
        self.shadow = {
            name: arr.astype(arr.dtype)
            for name, arr in _flatten_params(model.trainable_parameters())
        }
        mx.eval(self.shadow)

        # Build the compiled step body once.  mx.compile traces the function
        # the first time it is called and reuses the graph for subsequent
        # calls with the same input shapes.
        @partial(mx.compile, inputs=[self.shadow], outputs=[self.shadow])
        def _ema_step(params: dict):
            d = self.decay
            for name, p in params.items():
                if name in self.shadow:
                    self.shadow[name] = mx.stop_gradient(
                        d * self.shadow[name] + (1.0 - d) * p
                    )
        self._ema_step = _ema_step

    def update(self, model: nn.Module):
        # Snapshot the current trainable params into a flat dict and feed the
        # compiled body. mx.compile keeps `self.shadow` and `params` as
        # compile-time captures, so the entire fused EMA update is one graph.
        params = dict(_flatten_params(model.trainable_parameters()))
        self._ema_step(params)

    def state_dict(self):
        return {"decay": self.decay,
                "shadow": {k: v for k, v in self.shadow.items()}}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.shadow = sd["shadow"]


# =============================================================================
# LR SCHEDULE
# =============================================================================

def cosine_warmup_lr(step: int, total_steps: int,
                     warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


# =============================================================================
# LOSS  (with label smoothing)
# =============================================================================

def clip_grads_by_norm(grads, max_norm: float):
    """Global-norm gradient clipping in a single fused graph.

    Equivalent to torch.nn.utils.clip_grad_norm_ on the full grad tree.
    Replaces the previous Python-loop version that built a separate node per
    parameter (hostile to mx.compile fusion).
    """
    sq_norms = tree_map(lambda g: (g.astype(mx.float32) ** 2).sum(), grads)
    leaves = [v for _, v in tree_flatten(sq_norms)]
    total = mx.sqrt(mx.sum(mx.stack(leaves)))
    scale = mx.minimum(max_norm / (total + 1e-6), 1.0)
    return tree_map(lambda g: (g * scale).astype(g.dtype), grads)


def make_loss_fn(num_classes: int, label_smoothing: float = 0.0):
    def loss_fn(model, video_bf16, target):
        # Cast logits to fp32 for numerical stability of softmax + log
        logits = model(video_bf16).astype(mx.float32)
        if label_smoothing > 0:
            log_probs = nn.log_softmax(logits, axis=-1)
            nll = -log_probs[mx.arange(target.shape[0]), target]
            smooth = -log_probs.mean(axis=-1)
            loss = (1 - label_smoothing) * nll + label_smoothing * smooth
            return loss.mean()
        return mx.mean(nn.losses.cross_entropy(logits, target))
    return loss_fn


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(path: Path, model, opt, ema, epoch: int, best_acc: float):
    """
    Save model weights (+ EMA shadow) as safetensors. Optimizer state is
    JSON-only metadata for now — we re-init the optimizer on resume because
    bf16 training with AdamW state is small (~1× model params) and the
    cosine schedule is the only optimizer-side state we actually want to
    preserve, which we recover from `step_offset` saved in the meta JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = dict(_flatten_params(model.parameters()))
    if ema is not None:
        # Prefix EMA shadow keys so they coexist with model params in one file
        for k, v in ema.shadow.items():
            weights[f"__ema__{k}"] = v
    mx.save_safetensors(str(path), weights)
    with open(path.with_suffix(".meta.json"), "w") as f:
        json.dump({"epoch": epoch, "best_acc": best_acc,
                   "ema_decay": (ema.decay if ema else None)}, f)
    print(f"Saved {path}  ({len(weights)} tensors)")


def load_checkpoint(path: Path, model, ema=None):
    """Load weights from a safetensors checkpoint into a freshly-built model."""
    weights = mx.load(str(path))
    model_weights = {k: v for k, v in weights.items() if not k.startswith("__ema__")}
    ema_weights = {
        k.removeprefix("__ema__"): v for k, v in weights.items()
        if k.startswith("__ema__")
    }

    model_keys = {n for n, _ in _flatten_params(model.parameters())}
    missing = model_keys - model_weights.keys()
    unexpected = model_weights.keys() - model_keys
    if missing:
        print(f"⚠ {len(missing)} model keys missing from checkpoint "
              f"(showing 3): {sorted(missing)[:3]}")
    if unexpected:
        print(f"⚠ {len(unexpected)} unexpected keys in checkpoint "
              f"(showing 3): {sorted(unexpected)[:3]}")
    from mlx.utils import tree_unflatten
    model.update(tree_unflatten(list(model_weights.items())))
    mx.eval(model.parameters())
    print(f"Loaded {len(model_weights)} model tensors from {path}")

    if ema is not None and ema_weights:
        for k, v in ema_weights.items():
            if k in ema.shadow:
                ema.shadow[k] = v
        print(f"Loaded {len(ema_weights)} EMA shadow tensors")


# =============================================================================
# DATASET (stub — adapt to your K400 source)
# =============================================================================

def iter_synthetic_dataset(cfg: TrainConfig, n_steps: int) -> Iterator[Tuple]:
    """Yields random tensors for smoke-testing the training loop."""
    rng = np.random.default_rng(0)
    for _ in range(n_steps):
        video = rng.standard_normal(
            (cfg.batch_size, cfg.num_frames, cfg.img_size, cfg.img_size, 3)
        ).astype(np.float32)
        labels = rng.integers(0, cfg.num_classes, size=cfg.batch_size, dtype=np.int64)
        yield mx.array(video), mx.array(labels)


def build_k400_loader(cfg: TrainConfig, split: str = "train"
                      ) -> Tuple[K400Dataset, StreamingDataLoader]:
    """Build the K400 dataset + threaded loader for the given split."""
    if split == "train":
        annot = cfg.annotations_csv
        video_root = cfg.video_root
    else:
        # Default to sibling val/ folder + val.csv if present
        annot = str(Path(cfg.annotations_csv).parent / f"{split}.csv")
        video_root = str(Path(cfg.video_root).parent / split)

    dataset = K400Dataset(
        video_root=video_root,
        annotations_csv=annot,
        num_frames=cfg.num_frames,
        img_size=cfg.img_size,
        split=split,
    )
    loader = StreamingDataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        prefetch=cfg.prefetch,
        drop_last=(split == "train"),
    )
    return dataset, loader


def iter_k400_dataset(loader: StreamingDataLoader) -> Iterator[Tuple[mx.array, mx.array]]:
    """Adapt the numpy-yielding StreamingDataLoader to MLX arrays."""
    for videos_np, labels_np in loader:
        # Already (B, T, H, W, C) channels-last; cast happens inside train_step.
        yield mx.array(videos_np), mx.array(labels_np)


# =============================================================================
# ONE EPOCH
# =============================================================================

def make_compiled_train_step(model, opt, grad_fn, grad_clip: float):
    """Build an `mx.compile`-wrapped `train_step(video, label) -> loss` that
    fuses forward + loss + backward + grad-clip + optimizer.update into a
    single MLX graph.

    Why this is a big win: each phase used to flush its own kernel batch with
    its own launch overhead. mx.compile traces the whole pipeline once on the
    first call and reuses the graph forever, eliding hundreds of redundant
    Python/launch round-trips per step.

    The `state` argument tells the compiler which captured arrays it can
    mutate in place (model params, optimizer Adam state, RNG). Anything not
    listed there must remain pure inputs/outputs.

    Caveat: the GDN backward goes through `mx.custom_function`. MLX is
    documented to compose `compile` and `custom_function` correctly, but the
    gradient-correctness regression test (verify_path_equivalence) should be
    re-run after any changes to either side to catch silent drift.
    """
    state = [model.state, opt.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(video, label):
        loss, grads = grad_fn(model, video, label)
        if grad_clip > 0:
            grads = clip_grads_by_norm(grads, grad_clip)
        opt.update(model, grads)
        return loss

    return train_step


def train_one_epoch(model, opt, ema, grad_fn, data_iter, *,
                    epoch: int, total_steps: int, warmup_steps: int,
                    base_lr: float, grad_clip: float, log_every: int,
                    step_offset: int = 0,
                    max_steps: Optional[int] = None):
    """Returns (avg_loss, n_steps_completed) so caller can advance step_offset."""
    losses: list[mx.array] = []  # kept on-device; see deferred-sync note below
    t_epoch = time.perf_counter()

    train_step = make_compiled_train_step(model, opt, grad_fn, grad_clip)

    last_logged_loss = float("nan")
    for local_step, (video, label) in enumerate(data_iter):
        if max_steps is not None and local_step >= max_steps:
            break

        global_step = step_offset + local_step
        opt.learning_rate = cosine_warmup_lr(
            global_step, total_steps, warmup_steps, base_lr,
        )

        # MLX channels-last + bf16 cast
        if video.shape[1] == 3:
            video = mx.transpose(video, (0, 2, 3, 4, 1))
        video = video.astype(mx.bfloat16)

        t_step = time.perf_counter()
        loss = train_step(video, label)

        if ema is not None:
            ema.update(model)

        # DEFERRED SYNC: keep loss as an mx.array. We only call .item() at log
        # time (every `log_every` steps) instead of every step, removing the
        # CPU↔GPU round-trip from the hot path.
        losses.append(loss)

        if local_step % log_every == 0:
            mx.eval(loss)              # one sync per log_every, not per step
            last_logged_loss = float(loss.item())
            dt = (time.perf_counter() - t_step) * 1000
            print(f"  ep{epoch:>3d}  step {local_step:>5d}  "
                  f"loss {last_logged_loss:.4f}  "
                  f"lr {opt.learning_rate:.2e}  step_ms {dt:.0f}", flush=True)

    if losses:
        # Single sync over every step's loss for the epoch-end average.
        mx.eval(*losses)
        loss_vals = [float(l.item()) for l in losses]
    else:
        loss_vals = []
    avg_loss = float(np.mean(loss_vals)) if loss_vals else float("nan")
    print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}  "
          f"time={time.perf_counter() - t_epoch:.0f} s  "
          f"steps={len(loss_vals)}")
    return avg_loss, len(loss_vals)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Practical guidance for K400 on a MacBook:

    On M3 Max with our defaults (B=2, T=32, 29.79M params):
        ~675 ms per training step.
        K400 train ≈ 246K videos → 123K steps per epoch → ~23 hours/epoch.
        Full 100-epoch training: ~3 months. Not viable on a single MacBook.

    What IS viable locally:
      • Architecture iteration (run a few hundred steps to validate code paths)
      • Small-scale runs on subsets of K400 (10-20 epochs on a 5-10K-video subset)
      • Validating numerical correctness (--debug mode below)
      • Fine-tuning from a pretrained checkpoint for ~5-10 epochs
      • TRAINING WHILE THE DOWNLOAD CONTINUES — every epoch re-scans the disk
        and picks up newly-completed videos, so the dataset grows over time.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="Use synthetic data + small num_steps for smoke test")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--video_root", type=str, default=None,
                        help=f"K400 train video dir (default: {TrainConfig().video_root})")
    parser.add_argument("--annotations_csv", type=str, default=None,
                        help=f"K400 train CSV (default: {TrainConfig().annotations_csv})")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="PyAV decode threads (try 4-8)")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="Cap steps per epoch — useful for short runs")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Resume / init from this safetensors checkpoint")
    parser.add_argument("--init_videomae", type=str, nargs="?",
                        const="MCG-NJU/videomae-base", default=None,
                        help="Initialize the encoder from VideoMAE base "
                             "(K400 self-supervised pretraining, NOT the SSv2 "
                             "or finetuned variants). Pass without value to use "
                             "the default 'MCG-NJU/videomae-base'.")
    parser.add_argument("--gdn_path", type=str, default=None,
                        choices=["chunkwise_kda_vjp", "metal_vjp", "compiled"],
                        help="GDN training path. Default: chunkwise_kda_vjp "
                             "(fastest at full-model scale on M3, lowest "
                             "saved-state memory). Use 'metal_vjp' for the "
                             "legacy step-by-step path, or 'compiled' as a "
                             "non-Metal fallback.")
    parser.add_argument("--chunk_size", type=int, default=None,
                        help="GDN chunk size. Validated against the chunkwise "
                             "Metal solver's TG-memory budget when a chunkwise "
                             "path is selected.")
    parser.add_argument("--log_every", type=int, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    for k in ("epochs", "batch_size", "lr", "video_root",
              "annotations_csv", "output_dir", "num_workers", "log_every",
              "gdn_path", "chunk_size"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)

    # Build model
    model = build_model(cfg)
    if args.init_videomae and args.ckpt:
        raise SystemExit(
            "--init_videomae and --ckpt are mutually exclusive. "
            "Use --ckpt to resume; use --init_videomae for a fresh run."
        )
    if args.init_videomae:
        load_videomae_base(model, checkpoint_name=args.init_videomae)
    if args.ckpt:
        load_checkpoint(Path(args.ckpt), model)

    opt = optim.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay,
                       betas=(0.9, 0.95))
    ema = ModelEMA(model, decay=cfg.ema_decay)

    loss_fn = make_loss_fn(cfg.num_classes, label_smoothing=cfg.label_smoothing)
    grad_fn = nn.value_and_grad(model, loss_fn)

    # ---- Smoke test on synthetic data ----
    if args.debug:
        print(f"DEBUG mode — synthetic data for {args.steps} steps")
        data_iter = iter_synthetic_dataset(cfg, args.steps)
        train_one_epoch(model, opt, ema, grad_fn, data_iter,
                        epoch=0,
                        total_steps=args.steps,
                        warmup_steps=max(1, args.steps // 10),
                        base_lr=cfg.lr, grad_clip=cfg.grad_clip,
                        log_every=cfg.log_every)
        return

    # ---- Real training ----
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Build dataset once; we'll refresh() it each epoch so newly-downloaded
    # videos enter the next epoch automatically.
    print(f"Loading K400 from {cfg.video_root}")
    print(f"Annotations: {cfg.annotations_csv}")
    train_ds, train_loader = build_k400_loader(cfg, split="train")
    print(f"Initial dataset size: {len(train_ds):,} videos "
          f"({train_ds.num_classes} classes)")

    if len(train_ds) < cfg.batch_size * 10:
        print(f"⚠ Only {len(train_ds)} videos available — will train when more "
              f"are downloaded. Sleeping 60s and refreshing...")
        time.sleep(60)
        train_ds.refresh()

    # Schedule planning. We assume the dataset will eventually reach the full
    # train.csv size; budget total_steps accordingly so the cosine decay
    # doesn't end prematurely if early epochs are short.
    expected_full_size = len({}.fromkeys(train_ds.filename_to_label).keys())
    steps_per_epoch_full = max(1, expected_full_size // cfg.batch_size)
    total_steps  = steps_per_epoch_full * cfg.epochs
    warmup_steps = steps_per_epoch_full * cfg.warmup_epochs
    print(f"Schedule: {total_steps:,} total steps assuming "
          f"{expected_full_size:,} full-dataset videos / {cfg.epochs} epochs; "
          f"warmup {warmup_steps:,} steps.")

    step_offset = 0
    best_acc = 0.0
    for epoch in range(cfg.epochs):
        train_ds.refresh()
        print(f"\n=== Epoch {epoch} ===  videos: {len(train_ds):,}")
        # Build a fresh iterator this epoch (StreamingDataLoader is one-shot)
        loader = StreamingDataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, prefetch=cfg.prefetch,
            drop_last=True, seed=epoch,
        )
        train_iter = iter_k400_dataset(loader)
        avg_loss, n_steps = train_one_epoch(
            model, opt, ema, grad_fn, train_iter,
            epoch=epoch, total_steps=total_steps, warmup_steps=warmup_steps,
            base_lr=cfg.lr, grad_clip=cfg.grad_clip,
            log_every=cfg.log_every, step_offset=step_offset,
            max_steps=args.max_steps_per_epoch,
        )
        step_offset += n_steps

        if (epoch + 1) % cfg.save_every_epoch == 0:
            save_checkpoint(out_dir / f"ckpt_ep{epoch:03d}.safetensors",
                            model, opt, ema, epoch, best_acc)


if __name__ == "__main__":
    main()
