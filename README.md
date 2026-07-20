# NoPE-GDN (MLX)

A causal, streamable video transformer that drops positional embeddings and replaces softmax
attention with **Gated DeltaNet / Kimi Delta Attention (KDA)** linear attention. Written in
[MLX](https://github.com/ml-explore/mlx) with custom Metal kernels, it trains on Kinetics-400 on
Apple Silicon — and runs on NVIDIA GPUs through MLX's CUDA backend.

## Repository

| File | Description |
| --- | --- |
| `nope_gdn_mlx.py` | Model definition and the custom Metal (MSL) KDA kernels — forward, backward (VJP), simdgroup-matrix matmul, and chunkwise WY triangular solves. Exposes several compute paths and falls back automatically on non-Metal backends. |
| `train_mlx_k400.py` | Kinetics-400 training loop: bf16 mixed precision, weight EMA, cosine warmup + decay, gradient clipping, safetensors checkpointing, and optional VideoMAE encoder init. |
| `k400_mlx_dataset.py` | PyAV-based K400 dataset and a threaded streaming loader. Built for a *live* download directory — re-scans each epoch and skips partial files. |

## Requirements

- Python 3.9+
- Apple Silicon: `pip install mlx` — or Linux + NVIDIA: `pip install "mlx[cuda12]"`
- `pip install numpy av`
- Optional, only for `--init_videomae`: `pip install torch transformers`

## Quick start

No dataset needed — these run on synthetic data:

```bash
# Model: smoke test, numerical path-equivalence check, and KDA benchmark
python nope_gdn_mlx.py          # forward/backward smoke test
python nope_gdn_mlx.py verify   # confirm all compute paths agree numerically
python nope_gdn_mlx.py bench    # benchmark the KDA recurrence paths

# Training loop on synthetic clips
python train_mlx_k400.py --debug --steps 10
```

## Training on Kinetics-400

Point the trainer at a directory of `.mp4` clips named `{youtube_id}_{start:06d}_{end:06d}.mp4`
and the official K400 annotations CSV (columns: `label, youtube_id, time_start, time_end`):

```bash
python train_mlx_k400.py \
  --video_root /path/to/k400/train \
  --annotations_csv /path/to/annotations/train.csv \
  --epochs 5 \
  --init_videomae            # optional: warm-start the encoder from VideoMAE-base
```

- **Live downloads** — the dataset re-scans the directory every epoch, so you can start training
  before the full dataset has finished downloading.
- **`--gdn_path`** selects the GDN training kernel:
  - `chunkwise_kda_vjp` — default; fastest at full-model scale, lowest backward-state memory
  - `metal_vjp` — legacy step-by-step Metal backward
  - `compiled` — non-Metal fallback (CPU / CUDA)
- Checkpoints are written as safetensors; resume with `--ckpt path.safetensors`.

Sanity-check the data pipeline on its own (edit the paths at the bottom of the file first):

```bash
python k400_mlx_dataset.py
```

