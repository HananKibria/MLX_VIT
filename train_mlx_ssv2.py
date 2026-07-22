"""
Train NoPE+GDN (BASE, 117M) on SSv2 with VideoMAE-base-ssv2 initialization.

Defaults match the TPAMI paper's setup for the supervised SSv2 ablation:
  • Base architecture: 12-block NoPE encoder + 4-block 3:1 hybrid processor
  • Tubelet (2, 16, 16), embed dim 768, 12 heads
  • Encoder initialized from MCG-NJU/videomae-base-ssv2 (the SSv2 self-supervised
    pretraining checkpoint), processor + head random
  • Chunkwise WY GDN forward + custom Metal backward (chunkwise_kda_vjp)
  • bf16 mixed precision, mx.compile-fused train_step
  • Cosine LR schedule with linear warmup
  • Optional layer-wise LR decay (default 0.7) — standard for VideoMAE finetune
  • Optional encoder freeze for the first N epochs (default 5)
  • EMA weights (decay 0.9999), gradient clipping, label smoothing
  • Per-epoch validation with top-1 / top-5

Usage:

    # Smoke test (synthetic data, 10 steps)
    python train_mlx_ssv2.py --debug --steps 10

    # Real training on /Users/hanan/Downloads/ssv2 (defaults are sensible)
    python train_mlx_ssv2.py

    # Skip VideoMAE init (random encoder, e.g. for an ablation)
    python train_mlx_ssv2.py --no_videomae_init

    # Validation-only on an existing checkpoint
    python train_mlx_ssv2.py --eval_only --ckpt outputs/mlx_ssv2/ckpt_ep020.safetensors
"""

from __future__ import annotations
import argparse
import json
import math
import time
from dataclasses import dataclass, asdict, field
from functools import partial
from pathlib import Path
from typing import Iterator, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from nope_gdn_mlx import (
    NoPEGDNClassifier, GatedDeltaLayer,
    to_bf16, param_dtype_summary, _flatten_params,
    load_videomae_base,
)
from ssv2_mlx_dataset import SSv2Dataset, StreamingDataLoader


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class TrainConfig:
    # ----- Model (BASE, paper config — matches nope_gdn_video_backbone32.ipynb) -----
    img_size: int = 224
    num_frames: int = 32
    tubelet_size: Tuple[int, int, int] = (2, 16, 16)
    encoder_dim: int = 768
    encoder_depth: int = 12
    encoder_heads: int = 12
    processor_dim: int = 768
    processor_depth: int = 4
    # Paper §III-B-3: 8 heads × 96-dim, giving a 96×96 state matrix per head.
    # Per §IV-F: ~74K parameters of temporal working memory per layer
    # (8 × 96² = 73,728). Matches the notebook's get_config(size="base").
    processor_heads: int = 8
    # M-series threadgroup-memory cap: C·(C + head_dim) ≤ 8192. With head_dim=96,
    # max chunk_size = 32 (32·128 = 4096 ≤ 8192). On CUDA + FLA's chunk_kda the
    # notebook uses chunk_size=64 — but CUDA shared mem isn't capped at 32 KB,
    # so the notebook's choice doesn't fit on Metal. C=32 → ~1.3-1.6× slower
    # GDN backward vs C=64, but state size matches the paper exactly.
    chunk_size: int = 32
    drop_path_rate: float = 0.2      # match notebook
    dropout: float = 0.2             # match notebook
    head_dropout: float = 0.3        # match notebook
    num_classes: int = 174

    # ----- Training (matches nope_gdn_video_backbone32.ipynb) -----
    batch_size: int = 2              # base is memory-heavy; bump if you have headroom
    grad_accum_steps: int = 2        # effective batch = batch_size × grad_accum_steps
    lr: float = 5e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.2        # notebook value
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999        # notebook uses (0.9, 0.999), not (0.9, 0.95)
    epochs: int = 50                 # notebook value
    warmup_epochs: int = 5           # notebook value
    label_smoothing: float = 0.2     # notebook value
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    layer_decay: float = 0.70        # layer-wise LR decay
    freeze_encoder_epochs: int = 5   # freeze VideoMAE encoder for first N epochs

    # ----- Mixup / Cutmix (matches notebook DataConfig) -----
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    mixup_prob: float = 0.5          # P(any mixing applied)
    cutmix_share: float = 0.5        # P(cutmix | mixing applied), else mixup

    # Mixed precision: cast model to bf16 once after init.
    use_bf16: bool = True
    gdn_path: str = "chunkwise_kda_vjp"

    # ----- Data -----
    video_root: str = "/Users/hanan/Downloads/ssv2/20bn-something-something-v2"
    annotations_dir: str = "/Users/hanan/Downloads/ssv2/annotations/labels"
    num_workers: int = 4
    prefetch: int = 4
    use_hflip: bool = False
    # ----- RandomErasing (matches the notebook's `T.RandomErasing(p=reprob)`) -----
    # Per-clip rectangle, applied AFTER ImageNet normalization. Same rectangle
    # on every frame so temporal coherence is preserved. Default 0.0 = off.
    # Set to 0.4 for paper-faithful regularization. The `_start_epoch` field
    # lets you defer turning it on (e.g. finish a no-RE epoch first, then
    # resume from a checkpoint with RE enabled).
    random_erase_prob: float = 0.0
    random_erase_start_epoch: int = 0

    # ----- Pretrained init -----
    init_videomae: bool = True
    videomae_checkpoint: str = "MCG-NJU/videomae-base-ssv2"

    # ----- Output -----
    output_dir: str = "outputs/mlx_ssv2_base_videomae"
    log_every: int = 20
    save_every_epoch: int = 1
    eval_every_epoch: int = 1


# =============================================================================
# MODEL
# =============================================================================

def build_model(cfg: TrainConfig) -> NoPEGDNClassifier:
    """Build, cast to bf16, set GDN compute path."""
    if cfg.gdn_path in ("chunkwise_kda", "chunkwise_kda_vjp"):
        head_dim = cfg.processor_dim // cfg.processor_heads
        budget = (cfg.chunk_size ** 2 + cfg.chunk_size * head_dim) * 4
        if budget > 32 * 1024:
            raise ValueError(
                f"chunk_size {cfg.chunk_size} × head_dim {head_dim} exceeds "
                f"32 KB threadgroup memory ({budget/1024:.1f} KB needed). "
                f"Lower chunk_size or use --gdn_path metal_vjp."
            )
    if cfg.gdn_path == "metal_vjp":
        # The metal_vjp backward kernel keeps the per-step state matrix S
        # (D·D fp32) in threadgroup memory plus 8 D-vectors and ~140 bytes
        # of reduction scratch. Total budget = D·(D + 8)·4 + 140 bytes.
        # M-series TG limit is 32 KB; D=80 fits, D ≥ 88 doesn't. Catch this
        # at config time so the user sees a clear path-suggestion error
        # instead of a cryptic kernel-load failure mid-training.
        head_dim = cfg.processor_dim // cfg.processor_heads
        budget = (head_dim * head_dim + 8 * head_dim) * 4 + 140
        if budget > 32 * 1024:
            raise ValueError(
                f"--gdn_path metal_vjp at processor head_dim={head_dim} "
                f"would need ~{budget / 1024:.1f} KB of threadgroup memory "
                f"(limit 32 KB). The S[D·D] matrix alone is "
                f"{head_dim * head_dim * 4 / 1024:.1f} KB. "
                f"At this head_dim use --gdn_path chunkwise_kda_vjp "
                f"(no per-step S in TG memory)."
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
        print(f"Model cast to bfloat16; params: {param_dtype_summary(model)}")

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
# LAYER-WISE LR DECAY  (standard for VideoMAE fine-tuning)
# =============================================================================

def build_lr_scales(model: NoPEGDNClassifier, cfg: TrainConfig) -> dict[str, float]:
    """Per-parameter LR multiplier via layer-wise decay.

    Convention (matches the user's PyTorch trainer + VideoMAE finetune practice):
        head        → 1.0
        norm/embed  → cfg.layer_decay ** total_layers
        encoder.i   → cfg.layer_decay ** (total_layers - i)
        processor.i → cfg.layer_decay ** (total_layers - encoder_depth - i)

    The deepest layer right before the head gets scale 1.0; the embedding
    learns slowest. Returns a flat dict {param_name: scale}.
    """
    if cfg.layer_decay >= 1.0:
        # No decay — all params get LR scale 1.0
        return {n: 1.0 for n, _ in _flatten_params(model.parameters())}

    total_layers = cfg.encoder_depth + cfg.processor_depth
    decay = cfg.layer_decay
    scales: dict[str, float] = {}
    for name, _ in _flatten_params(model.parameters()):
        # 1) Head + final norms = full LR
        if name.startswith("head") or name.startswith("backbone.processor.norm"):
            layer_id = total_layers   # → 0.7^0 = 1.0
        # 2) Tubelet embed = slowest
        elif name.startswith("backbone.encoder.tubelet_embed"):
            layer_id = 0              # → slowest
        # 3) Encoder block i
        elif name.startswith("backbone.encoder.blocks."):
            try:
                i = int(name.split("backbone.encoder.blocks.")[1].split(".")[0])
                layer_id = i + 1      # block 0 → id=1, block 11 → id=12
            except (IndexError, ValueError):
                layer_id = total_layers
        # 4) Processor block i
        elif name.startswith("backbone.processor.blocks."):
            try:
                i = int(name.split("backbone.processor.blocks.")[1].split(".")[0])
                layer_id = cfg.encoder_depth + 1 + i
            except (IndexError, ValueError):
                layer_id = total_layers
        # 5) Anything else (encoder.norm, processor.norm intermediate) → full LR
        else:
            layer_id = total_layers

        scales[name] = decay ** (total_layers - layer_id)
    return scales


# =============================================================================
# EMA
# =============================================================================

class ModelEMA:
    """EMA over learnable params, fused with mx.compile."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {
            name: arr.astype(arr.dtype)
            for name, arr in _flatten_params(model.trainable_parameters())
        }
        mx.eval(self.shadow)

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
        params = dict(_flatten_params(model.trainable_parameters()))
        self._ema_step(params)

    def state_dict(self):
        return {"decay": self.decay, "shadow": dict(self.shadow)}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.shadow = sd["shadow"]


# =============================================================================
# LR / LOSS UTILS
# =============================================================================

def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int,
                     base_lr: float, min_lr: float = 0.0) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def clip_grads_by_norm(grads, max_norm: float):
    """Global-norm gradient clipping in a single fused graph."""
    sq_norms = tree_map(lambda g: (g.astype(mx.float32) ** 2).sum(), grads)
    leaves = [v for _, v in tree_flatten(sq_norms)]
    total = mx.sqrt(mx.sum(mx.stack(leaves)))
    scale = mx.minimum(max_norm / (total + 1e-6), 1.0)
    return tree_map(lambda g: (g * scale).astype(g.dtype), grads)


def scale_grads_per_layer(grads, scales: dict[str, float]):
    """Multiply each leaf gradient by its per-layer LR scale."""
    flat_grads = tree_flatten(grads)
    scaled = []
    for name, g in flat_grads:
        s = scales.get(name, 1.0)
        if s != 1.0:
            scaled.append((name, (g * s).astype(g.dtype)))
        else:
            scaled.append((name, g))
    return tree_unflatten(scaled)


def freeze_encoder_grads(grads, encoder_prefix: str = "backbone.encoder."):
    """Zero out gradients for all encoder params (first N epochs)."""
    flat = tree_flatten(grads)
    out = []
    for name, g in flat:
        if name.startswith(encoder_prefix):
            out.append((name, mx.zeros_like(g)))
        else:
            out.append((name, g))
    return tree_unflatten(out)


def make_loss_fn(num_classes: int, label_smoothing: float = 0.0):
    """Hard-target cross entropy with optional label smoothing.
    Returns (loss, n_correct) so the train loop can track top-1 accuracy
    against the same logits used for the gradient (no extra forward pass).
    `n_correct` is treated as an aux value by mx.value_and_grad — only the
    first return (loss) is differentiated.
    Use this when no mixup/cutmix is applied (target is int [B])."""
    def loss_fn(model, video_bf16, target):
        logits = model(video_bf16).astype(mx.float32)
        if label_smoothing > 0:
            log_probs = nn.log_softmax(logits, axis=-1)
            nll = -log_probs[mx.arange(target.shape[0]), target]
            smooth = -log_probs.mean(axis=-1)
            loss = ((1 - label_smoothing) * nll + label_smoothing * smooth).mean()
        else:
            loss = mx.mean(nn.losses.cross_entropy(logits, target))
        pred = mx.argmax(logits, axis=-1)
        n_correct = mx.sum(pred == target).astype(mx.int32)
        return loss, n_correct
    return loss_fn


def make_soft_loss_fn():
    """Soft-target cross entropy. Use this when mixup/cutmix is applied
    (target is one-hot or mixed-one-hot float [B, num_classes]).

    Accuracy is reported against the DOMINANT label (argmax of soft target),
    matching the convention used in the original PyTorch trainer."""
    def loss_fn(model, video_bf16, soft_target):
        logits = model(video_bf16).astype(mx.float32)
        log_probs = nn.log_softmax(logits, axis=-1)
        loss = (-mx.sum(soft_target * log_probs, axis=-1)).mean()
        pred = mx.argmax(logits, axis=-1)
        true_label = mx.argmax(soft_target, axis=-1)
        n_correct = mx.sum(pred == true_label).astype(mx.int32)
        return loss, n_correct
    return loss_fn


# =============================================================================
# MIXUP / CUTMIX  (matches the notebook's DataConfig)
# =============================================================================

def _onehot(labels_np: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((labels_np.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels_np.shape[0]), labels_np] = 1.0
    return out


def mixup_cutmix_batch(video_mx: mx.array, labels_mx: mx.array,
                       num_classes: int, *,
                       mixup_alpha: float, cutmix_alpha: float,
                       mixup_prob: float, cutmix_share: float,
                       label_smoothing: float):
    """Apply Mixup or Cutmix (mutually exclusive per batch).

    Returns:
        mixed_video    [B, T, H, W, C]  (same dtype as input)
        soft_target    [B, num_classes] float32 — pass to soft loss
        applied        bool — whether mixing was actually applied this step
    """
    B = video_mx.shape[0]
    # Build soft labels (with smoothing). When mixing isn't applied we still
    # return soft targets so the call site can use a single soft-loss path.
    labels_np = np.array(labels_mx).astype(np.int64)
    one_hot = _onehot(labels_np, num_classes)
    if label_smoothing > 0:
        eps = label_smoothing
        one_hot = one_hot * (1 - eps) + eps / num_classes

    # Roll mixing dice
    if np.random.rand() >= mixup_prob:
        return video_mx, mx.array(one_hot), False

    perm = np.random.permutation(B)
    perm_mx = mx.array(perm.astype(np.int64))
    use_cutmix = np.random.rand() < cutmix_share

    if use_cutmix and cutmix_alpha > 0:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        # Sample bounding box
        H, W = int(video_mx.shape[2]), int(video_mx.shape[3])
        cut_h = max(1, int(H * math.sqrt(1 - lam)))
        cut_w = max(1, int(W * math.sqrt(1 - lam)))
        cy = np.random.randint(H);  cx = np.random.randint(W)
        bby1 = max(0, cy - cut_h // 2);  bby2 = min(H, cy + cut_h // 2)
        bbx1 = max(0, cx - cut_w // 2);  bbx2 = min(W, cx + cut_w // 2)

        # Build mask in numpy then upload to MLX (cheaper than indexed assign)
        mask_np = np.ones((1, 1, H, W, 1), dtype=np.float32)
        mask_np[:, :, bby1:bby2, bbx1:bbx2, :] = 0.0
        mask = mx.array(mask_np).astype(video_mx.dtype)
        mixed_video = mask * video_mx + (1.0 - mask) * video_mx[perm_mx]

        # Adjust lambda by actual cut area
        lam_adj = 1.0 - (bbx2 - bbx1) * (bby2 - bby1) / float(H * W)
    else:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha)) if mixup_alpha > 0 else 1.0
        mixed_video = lam * video_mx + (1.0 - lam) * video_mx[perm_mx]
        lam_adj = lam

    soft_target = lam_adj * one_hot + (1.0 - lam_adj) * one_hot[perm]
    return mixed_video, mx.array(soft_target), True


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(path: Path, model, ema, epoch: int, best_acc: float, top1: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = dict(_flatten_params(model.parameters()))
    if ema is not None:
        for k, v in ema.shadow.items():
            weights[f"__ema__{k}"] = v
    mx.save_safetensors(str(path), weights)
    with open(path.with_suffix(".meta.json"), "w") as f:
        json.dump({
            "epoch": epoch, "best_acc": best_acc, "current_acc": top1,
            "ema_decay": (ema.decay if ema else None),
        }, f)
    print(f"Saved {path}  ({len(weights)} tensors)")


def load_checkpoint(path: Path, model, ema=None):
    weights = mx.load(str(path))
    model_w = {k: v for k, v in weights.items() if not k.startswith("__ema__")}
    ema_w = {k.removeprefix("__ema__"): v for k, v in weights.items()
             if k.startswith("__ema__")}

    model_keys = {n for n, _ in _flatten_params(model.parameters())}
    missing = model_keys - model_w.keys()
    unexpected = model_w.keys() - model_keys
    if missing:
        print(f"⚠ {len(missing)} model keys missing from checkpoint")
    if unexpected:
        print(f"⚠ {len(unexpected)} unexpected keys in checkpoint")
    model.update(tree_unflatten(list(model_w.items())))
    mx.eval(model.parameters())
    print(f"Loaded {len(model_w)} model tensors from {path}")

    if ema is not None and ema_w:
        for k, v in ema_w.items():
            if k in ema.shadow:
                ema.shadow[k] = v
        print(f"Loaded {len(ema_w)} EMA shadow tensors")


# =============================================================================
# DATA
# =============================================================================

def iter_synthetic_dataset(cfg: TrainConfig, n_steps: int) -> Iterator[Tuple[mx.array, mx.array]]:
    rng = np.random.default_rng(0)
    for _ in range(n_steps):
        v = rng.standard_normal(
            (cfg.batch_size, cfg.num_frames, cfg.img_size, cfg.img_size, 3)
        ).astype(np.float32)
        l = rng.integers(0, cfg.num_classes, size=cfg.batch_size, dtype=np.int64)
        yield mx.array(v), mx.array(l)


def build_ssv2_loader(cfg: TrainConfig, split: str) -> Tuple[SSv2Dataset, StreamingDataLoader]:
    ds = SSv2Dataset(
        video_root=cfg.video_root,
        annotations_dir=cfg.annotations_dir,
        num_frames=cfg.num_frames,
        img_size=cfg.img_size,
        split=split,
        use_hflip=cfg.use_hflip,
        # RE only takes effect on train, only for epochs ≥ start; the
        # trainer flips it on via dataset.set_epoch(epoch) per epoch.
        random_erase_prob=cfg.random_erase_prob if split == "train" else 0.0,
        random_erase_start_epoch=cfg.random_erase_start_epoch,
    )
    loader = StreamingDataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        prefetch=cfg.prefetch,
        drop_last=(split == "train"),
    )
    return ds, loader


def iter_ssv2_dataset(loader: StreamingDataLoader) -> Iterator[Tuple[mx.array, mx.array]]:
    for videos_np, labels_np in loader:
        yield mx.array(videos_np), mx.array(labels_np)


# =============================================================================
# TRAIN STEP (compiled)
# =============================================================================

def make_compiled_train_step(model, opt, grad_fn, grad_clip: float,
                             lr_scales: Optional[dict[str, float]] = None,
                             freeze_encoder: bool = False):
    """Build mx.compile-wrapped train_step that fuses everything except dataloading.

    The grad_fn is treated as a black box — it dispatches to soft-loss when
    mixup/cutmix was applied to this batch, hard-loss otherwise.

    Note on grad_accum_steps: the notebook uses 2 (effective batch 32 with
    batch_size 16). On MLX/M3 batch_size is typically 2 due to memory; cross-
    batch accumulation in a compiled MLX graph requires extra state plumbing
    that complicates this simple pipeline. If you need a larger effective
    batch, the cleanest option is to lower the LR proportionally (linear
    rule) — which is what most papers do anyway.
    """
    state = [model.state, opt.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(video, label):
        # loss_fn returns (loss, n_correct); mx.value_and_grad differentiates
        # only the first element and passes the rest through as aux.
        (loss, n_correct), grads = grad_fn(model, video, label)
        if grad_clip > 0:
            grads = clip_grads_by_norm(grads, grad_clip)
        if lr_scales is not None:
            grads = scale_grads_per_layer(grads, lr_scales)
        if freeze_encoder:
            grads = freeze_encoder_grads(grads)
        opt.update(model, grads)
        return loss, n_correct

    return train_step


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, val_iter) -> Tuple[float, float, int]:
    """Single-view top-1/top-5 on the validation set. Returns (top1, top5, n)."""
    correct1 = 0
    correct5 = 0
    n_total = 0

    for video, label in val_iter:
        if video.shape[1] == 3:
            video = mx.transpose(video, (0, 2, 3, 4, 1))
        video = video.astype(mx.bfloat16)
        logits = model(video).astype(mx.float32)
        # Top-5: argmax over -logits and check membership
        topk = mx.argpartition(-logits, kth=5, axis=-1)[:, :5]
        pred1 = mx.argmax(logits, axis=-1)
        mx.eval(pred1, topk, label)
        pred1_np = np.array(pred1)
        topk_np = np.array(topk)
        label_np = np.array(label)
        correct1 += int((pred1_np == label_np).sum())
        correct5 += int(np.any(topk_np == label_np[:, None], axis=1).sum())
        n_total += int(label_np.shape[0])

    if n_total == 0:
        return 0.0, 0.0, 0
    return 100.0 * correct1 / n_total, 100.0 * correct5 / n_total, n_total


# =============================================================================
# ONE EPOCH
# =============================================================================

def train_one_epoch(model, opt, ema, grad_fn_hard, grad_fn_soft, data_iter, *,
                    cfg: TrainConfig,
                    epoch: int, total_steps: int, warmup_steps: int,
                    base_lr: float, min_lr: float, grad_clip: float,
                    log_every: int, step_offset: int = 0,
                    max_steps: Optional[int] = None,
                    lr_scales: Optional[dict[str, float]] = None,
                    freeze_encoder: bool = False):
    """Train one epoch with mixup/cutmix applied per batch.

    Two compiled train_steps are kept resident — one for hard targets (no
    mixing this batch) and one for soft targets (mixup or cutmix applied).
    Dispatch happens at the Python level so the compiled graph for each
    branch stays static.
    """
    # CRITICAL: store Python floats, NOT lazy mx.arrays. Holding a list of
    # un-eval'd mx.arrays keeps every step's compute graph alive (graph chains
    # via model.state read by the next step), which causes step time to grow
    # monotonically and ultimately OOMs. We eval every step instead — see the
    # `mx.eval(...)` below.
    loss_vals: list[float] = []
    # Running top-1: cumulative since the start of this epoch.
    correct_total = 0
    seen_total = 0
    t_epoch = time.perf_counter()

    train_step_hard = make_compiled_train_step(
        model, opt, grad_fn_hard, grad_clip,
        lr_scales=lr_scales, freeze_encoder=freeze_encoder,
    )
    train_step_soft = make_compiled_train_step(
        model, opt, grad_fn_soft, grad_clip,
        lr_scales=lr_scales, freeze_encoder=freeze_encoder,
    )
    if freeze_encoder:
        print(f"  ❄️  Encoder frozen for epoch {epoch}")

    n_mixed = 0
    for local_step, (video, label) in enumerate(data_iter):
        if max_steps is not None and local_step >= max_steps:
            break
        global_step = step_offset + local_step
        opt.learning_rate = cosine_warmup_lr(
            global_step, total_steps, warmup_steps, base_lr, min_lr,
        )

        # Channels-last + bf16
        if video.shape[1] == 3:
            video = mx.transpose(video, (0, 2, 3, 4, 1))
        video = video.astype(mx.bfloat16)

        # Mixup / cutmix dispatch
        mixed_video, soft_target, was_mixed = mixup_cutmix_batch(
            video, label, cfg.num_classes,
            mixup_alpha=cfg.mixup_alpha, cutmix_alpha=cfg.cutmix_alpha,
            mixup_prob=cfg.mixup_prob, cutmix_share=cfg.cutmix_share,
            label_smoothing=cfg.label_smoothing,
        )

        t_step = time.perf_counter()
        if was_mixed:
            n_mixed += 1
            loss, n_correct = train_step_soft(
                mixed_video, soft_target.astype(mx.float32),
            )
        else:
            # No mixup this batch — use hard targets (label smoothing applied
            # inside the hard-loss fn). Pass the original (un-mixed) video.
            loss, n_correct = train_step_hard(video, label)

        if ema is not None:
            ema.update(model)

        # ---- Force-evaluate per step to release the step's compute graph ----
        # The compiled train_step updates model.state / opt.state via the
        # inputs=state, outputs=state contract, but those updates remain LAZY.
        # If we don't eval here, the next step reads still-lazy state arrays
        # whose compute graphs reach back to this step's activations and
        # backward kernels — that's the chain that grows unboundedly.
        # `loss.item()` implicitly evals `loss` plus its dependency closure
        # (which spans model.state and opt.state through the compile output
        # contract); we add `ema.shadow` explicitly because the EMA update
        # writes through a separate compiled graph not connected to `loss`.
        if ema is not None:
            mx.eval(loss, n_correct, ema.shadow)
        else:
            mx.eval(loss, n_correct)
        loss_val = float(loss.item())
        n_correct_val = int(n_correct.item())
        loss_vals.append(loss_val)
        correct_total += n_correct_val
        seen_total += int(label.shape[0])
        # Drop the only remaining Python ref to the step's lazy mx.array so
        # GC can release any temporary allocations now that we have the float.
        del loss, n_correct

        if local_step % log_every == 0:
            dt = (time.perf_counter() - t_step) * 1000
            tag = "mix" if was_mixed else "hard"
            running_acc = 100.0 * correct_total / max(seen_total, 1)
            print(f"  ep{epoch:>3d}  step {local_step:>5d}  [{tag}]  "
                  f"loss {loss_val:.4f}  acc {running_acc:5.2f}%  "
                  f"lr {opt.learning_rate:.2e}  "
                  f"step_ms {dt:.0f}", flush=True)

    avg_loss = float(np.mean(loss_vals)) if loss_vals else float("nan")
    train_acc = 100.0 * correct_total / max(seen_total, 1)
    print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}  "
          f"train_top1={train_acc:.2f}%  "
          f"mixed_steps={n_mixed}/{len(loss_vals)}  "
          f"time={time.perf_counter() - t_epoch:.0f} s")
    return avg_loss, len(loss_vals)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="Synthetic data + small num_steps for smoke test")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None,
                        help=f"Per-step batch size (default: {TrainConfig().batch_size}). "
                             f"Base 117M model on M3 Max bf16: try 2 first, bump to "
                             f"4 if memory allows. Overrides TrainConfig.batch_size.")
    parser.add_argument("--lr", type=float, default=None,
                        help=f"Peak learning rate (default: {TrainConfig().lr}). "
                             f"If you scale batch_size up by k, scale lr by k too "
                             f"(linear rule).")
    parser.add_argument("--video_root", type=str, default=None)
    parser.add_argument("--annotations_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Resume / init from this safetensors checkpoint "
                             "(takes precedence over VideoMAE init)")
    parser.add_argument("--no_videomae_init", action="store_true",
                        help="Skip VideoMAE encoder initialization (random init)")
    parser.add_argument("--videomae_checkpoint", type=str, default=None,
                        help=f"VideoMAE HF id (default: {TrainConfig().videomae_checkpoint})")
    parser.add_argument("--gdn_path", type=str, default=None,
                        choices=["chunkwise_kda_vjp", "metal_vjp", "compiled"])
    parser.add_argument("--chunk_size", type=int, default=None)
    # ----- Size preset (sets encoder_dim/depth/heads + processor_*  +
    #       videomae_checkpoint) — matches `nope_gdn_video_backbone32.ipynb` -----
    parser.add_argument("--size", type=str, default=None,
                        choices=("tiny", "small", "base"),
                        help="Architecture preset. tiny=192d/6L/3H (~3M, no "
                             "VideoMAE init), small=384d/12L/6H (~25M, inits "
                             "from videomae-small-finetuned-ssv2), "
                             "base=768d/12L/12H/8GDN-heads (~117M, inits "
                             "from videomae-base-ssv2). Individual --encoder_dim "
                             "etc. flags below still override this preset.")
    # ----- Per-knob overrides (set automatically by --size; expose for tweaks) -----
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Frames per clip (default 32). 16 ≈ 2× faster.")
    parser.add_argument("--img_size", type=int, default=None,
                        help="Spatial resolution (default 224).")
    parser.add_argument("--encoder_dim", type=int, default=None)
    parser.add_argument("--encoder_heads", type=int, default=None)
    parser.add_argument("--processor_dim", type=int, default=None)
    parser.add_argument("--processor_heads", type=int, default=None)
    parser.add_argument("--encoder_depth", type=int, default=None)
    parser.add_argument("--processor_depth", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    # ----- RandomErasing (paper-faithful regularization) -----
    parser.add_argument("--random_erase_prob", type=float, default=None,
                        help="Per-clip RandomErasing probability "
                             "(default 0.0 = off; paper uses 0.4).")
    parser.add_argument("--random_erase_start_epoch", type=int, default=None,
                        help="Epoch from which RandomErasing turns on. Use "
                             "with --ckpt to resume mid-training and only "
                             "start erasing after the saved epoch.")
    # ----- Resume from a specific epoch (overrides ckpt's saved epoch+1) -----
    parser.add_argument("--start_epoch", type=int, default=None,
                        help="Epoch index to resume at. Default: when --ckpt "
                             "is given, reads the checkpoint's `meta.json` "
                             "and resumes at saved_epoch+1; otherwise 0.")
    parser.add_argument("--freeze_encoder_epochs", type=int, default=None)
    parser.add_argument("--layer_decay", type=float, default=None)
    parser.add_argument("--eval_only", action="store_true",
                        help="Run validation only and exit (use with --ckpt)")
    args = parser.parse_args()

    cfg = TrainConfig()

    # ---- Size preset (applied BEFORE per-knob overrides) ----
    # Matches the canonical `nope_gdn_video_backbone32.ipynb` `get_config()`
    # presets so that `--size {tiny,small,base}` reproduces the notebook's
    # tiny/small/base architectures.
    if args.size is not None:
        SIZE_PRESETS = {
            "tiny":  dict(encoder_dim=192, encoder_depth=6,  encoder_heads=3,
                          processor_dim=192, processor_depth=4, processor_heads=3,
                          # No public VideoMAE-tiny → random init.
                          init_videomae=False, videomae_checkpoint=None),
            "small": dict(encoder_dim=384, encoder_depth=12, encoder_heads=6,
                          processor_dim=384, processor_depth=4, processor_heads=6,
                          # The only public VideoMAE-small is the SSv2-finetuned
                          # variant (the masked-pretraining-only weights are gated).
                          videomae_checkpoint="MCG-NJU/videomae-small-finetuned-ssv2"),
            "base":  dict(encoder_dim=768, encoder_depth=12, encoder_heads=12,
                          processor_dim=768, processor_depth=4, processor_heads=8,
                          videomae_checkpoint="MCG-NJU/videomae-base-ssv2"),
        }
        for k, v in SIZE_PRESETS[args.size].items():
            setattr(cfg, k, v)
        print(f"--size {args.size}: encoder={cfg.encoder_dim}d/{cfg.encoder_depth}L/"
              f"{cfg.encoder_heads}H, processor={cfg.processor_dim}d/"
              f"{cfg.processor_depth}L/{cfg.processor_heads}H "
              f"(videomae={cfg.videomae_checkpoint or 'OFF'})")

    for k in ("epochs", "batch_size", "lr", "video_root", "annotations_dir",
              "output_dir", "num_workers", "log_every", "gdn_path", "chunk_size",
              "freeze_encoder_epochs", "layer_decay", "videomae_checkpoint",
              "num_frames", "img_size",
              "encoder_dim", "encoder_heads", "encoder_depth",
              "processor_dim", "processor_heads", "processor_depth",
              "random_erase_prob", "random_erase_start_epoch"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.no_videomae_init:
        cfg.init_videomae = False

    # ---- Build model ----
    model = build_model(cfg)
    if args.ckpt and cfg.init_videomae and not args.no_videomae_init:
        print("--ckpt given; skipping VideoMAE init (checkpoint takes precedence)")
        cfg.init_videomae = False
    if cfg.init_videomae:
        print(f"Initializing encoder from {cfg.videomae_checkpoint!r} ...")
        load_videomae_base(model, checkpoint_name=cfg.videomae_checkpoint)

    # ---- Optimizer + EMA + grad fns (one for hard targets, one for soft) ----
    # EMA must be constructed BEFORE load_checkpoint so we can pass it in and
    # restore the saved `__ema__*` shadow tensors. Otherwise the saved EMA
    # state is silently discarded on resume.
    opt = optim.AdamW(
        learning_rate=cfg.lr, weight_decay=cfg.weight_decay,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
    )
    ema = ModelEMA(model, decay=cfg.ema_decay)
    if args.ckpt:
        load_checkpoint(Path(args.ckpt), model, ema=ema)
    hard_loss_fn = make_loss_fn(cfg.num_classes, label_smoothing=cfg.label_smoothing)
    soft_loss_fn = make_soft_loss_fn()
    grad_fn_hard = nn.value_and_grad(model, hard_loss_fn)
    grad_fn_soft = nn.value_and_grad(model, soft_loss_fn)

    # ---- Layer-wise LR scales ----
    lr_scales = build_lr_scales(model, cfg) if cfg.layer_decay < 1.0 else None
    if lr_scales is not None:
        unique_scales = sorted(set(lr_scales.values()))
        print(f"Layer-wise LR decay {cfg.layer_decay}: "
              f"{len(unique_scales)} unique scales, "
              f"min={min(unique_scales):.4f}, max={max(unique_scales):.4f}")

    # ---- Smoke test ----
    if args.debug:
        print(f"DEBUG: synthetic data, {args.steps} steps")
        data_iter = iter_synthetic_dataset(cfg, args.steps)
        train_one_epoch(
            model, opt, ema, grad_fn_hard, grad_fn_soft, data_iter,
            cfg=cfg,
            epoch=0, total_steps=args.steps,
            warmup_steps=max(1, args.steps // 10),
            base_lr=cfg.lr, min_lr=cfg.min_lr, grad_clip=cfg.grad_clip,
            log_every=cfg.log_every, lr_scales=lr_scales,
            freeze_encoder=False,
        )
        return

    # ---- Real training ----
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"\n=== Loading SSv2 ===")
    print(f"  videos:      {cfg.video_root}")
    print(f"  annotations: {cfg.annotations_dir}")
    train_ds, _ = build_ssv2_loader(cfg, split="train")
    val_ds, _ = build_ssv2_loader(cfg, split="validation")
    print(f"  train: {len(train_ds):,} videos")
    print(f"  val:   {len(val_ds):,} videos")
    print(f"  classes: {train_ds.num_classes}")

    # ---- Eval-only mode ----
    if args.eval_only:
        if args.ckpt is None:
            print("⚠ --eval_only without --ckpt: evaluating freshly built model")
        val_loader = StreamingDataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, prefetch=cfg.prefetch, drop_last=False,
        )
        # Use EMA weights if present
        if any(v.size for v in ema.shadow.values()):
            print("Using EMA weights for evaluation")
            saved = {n: a for n, a in _flatten_params(model.parameters())}
            model.update(tree_unflatten(list(ema.shadow.items())))
            mx.eval(model.parameters())
        top1, top5, n = evaluate(model, iter_ssv2_dataset(val_loader))
        print(f"VAL  top1={top1:.2f}%  top5={top5:.2f}%  n={n}")
        return

    # Schedule planning (assumes the dataset doesn't grow during training)
    steps_per_epoch = max(1, len(train_ds) // cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs
    print(f"Schedule: {total_steps:,} total steps, "
          f"{steps_per_epoch:,} per epoch, warmup {warmup_steps:,}")

    # ---- Resume planning ----
    # Default: when --ckpt is given, read its meta.json and resume from
    # saved_epoch + 1. Without --ckpt, start at 0. --start_epoch overrides.
    start_epoch = 0
    best_top1 = 0.0
    if args.ckpt:
        meta_path = Path(args.ckpt).with_suffix(".meta.json")
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                start_epoch = int(meta.get("epoch", -1)) + 1
                best_top1 = float(meta.get("best_acc", 0.0))
                print(f"Resuming from {args.ckpt!s} "
                      f"(saved epoch {meta.get('epoch')}, "
                      f"best_acc {best_top1:.2f}%) → starting at epoch "
                      f"{start_epoch}")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠ Couldn't read {meta_path}: {e}; starting at 0")
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
        print(f"--start_epoch override: epoch {start_epoch}")
    if start_epoch >= cfg.epochs:
        print(f"start_epoch {start_epoch} ≥ cfg.epochs {cfg.epochs}; "
              f"nothing to train.")
        return

    step_offset = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, cfg.epochs):
        print(f"\n=== Epoch {epoch} ===")
        # Activate epoch-gated augmentations (RandomErasing).
        train_ds.set_epoch(epoch)
        if (cfg.random_erase_prob > 0.0
                and epoch == cfg.random_erase_start_epoch):
            print(f"  RandomErasing turned ON at epoch {epoch} "
                  f"(p={cfg.random_erase_prob})")
        train_loader = StreamingDataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, prefetch=cfg.prefetch,
            drop_last=True, seed=epoch,
        )
        train_iter = iter_ssv2_dataset(train_loader)
        is_frozen = epoch < cfg.freeze_encoder_epochs

        avg_loss, n_steps = train_one_epoch(
            model, opt, ema, grad_fn_hard, grad_fn_soft, train_iter,
            cfg=cfg,
            epoch=epoch, total_steps=total_steps, warmup_steps=warmup_steps,
            base_lr=cfg.lr, min_lr=cfg.min_lr, grad_clip=cfg.grad_clip,
            log_every=cfg.log_every, step_offset=step_offset,
            max_steps=args.max_steps_per_epoch,
            lr_scales=lr_scales, freeze_encoder=is_frozen,
        )
        step_offset += n_steps

        # Skip validation + save when no real training happened this epoch
        # (e.g. `--max_steps_per_epoch 0` is a "probe the resume/RE plumbing
        # without paying for 30 min of val" mode).
        if args.max_steps_per_epoch == 0:
            print("  (max_steps_per_epoch=0 → skipping validation + save)")
            continue

        # Validation
        top1 = top5 = 0.0
        if (epoch + 1) % cfg.eval_every_epoch == 0:
            print("Running validation...")
            val_loader = StreamingDataLoader(
                val_ds, batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers, prefetch=cfg.prefetch, drop_last=False,
            )
            top1, top5, n = evaluate(model, iter_ssv2_dataset(val_loader))
            print(f"VAL  ep{epoch}  top1={top1:.2f}%  top5={top5:.2f}%  n={n}")

        # Save best + periodic
        is_best = top1 > best_top1
        if is_best:
            best_top1 = top1
            save_checkpoint(out_dir / "best_model.safetensors",
                            model, ema, epoch, best_top1, top1)
            print(f"  ★ New best: {best_top1:.2f}%")
        if (epoch + 1) % cfg.save_every_epoch == 0:
            save_checkpoint(out_dir / f"ckpt_ep{epoch:03d}.safetensors",
                            model, ema, epoch, best_top1, top1)


if __name__ == "__main__":
    main()
