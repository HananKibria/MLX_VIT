"""
Local MLX inference harness for the NoPE-GDN SSv2 checkpoint (best_model_25.pt,
8x96, mean-pool head, gdn_temporal_only=True, EMA shadow weights).

Runs the weakness-localization experiments on an M4 without any training.
Metal KDA kernels overflow threadgroup memory at head_dim=96, so every GDN
layer is forced onto the pure-MLX 'compiled' recurrence path.
"""
import os, sys, time, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mlx.core as mx
import nope_gdn_mlx as M
import ssv2_mlx_dataset as D

SSV2      = "/Users/hanan/Downloads/ssv2"
VIDEO_DIR = os.path.join(SSV2, "20bn-something-something-v2")
ANN       = os.path.join(SSV2, "annotations", "labels")
PT_CKPT   = "/Users/hanan/Downloads/final2/best_model_25.pt"
MLX_W     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "best_model_25_ema.mlx.safetensors")


def build_model(head_type="mean", gdn_temporal_only=True,
                compute_path="chunkwise_kda", chunk_size=16):
    # chunkwise_kda @ chunk_size=16 is the FASTEST VERIFIED path at head_dim=96
    # (~55ms vs 'compiled' 303ms = 5.5x; numerically identical, max|Δ|=5e-8).
    # The Metal simdgroup kernel (metal_sg, the channel-wise default) overflows the
    # M4's 32KB threadgroup at D=96; 'metal' (61ms) and 'compiled' (303ms) also work.
    m = M.NoPEGDNClassifier(
        img_size=224, num_frames=32, tubelet_size=(2, 16, 16),
        encoder_dim=768, encoder_depth=12, encoder_heads=12,
        processor_dim=768, processor_depth=4, processor_heads=8,
        gdn_ratio=3, chunk_size=chunk_size, channel_wise_decay=True,
        allow_neg_eigval=False, factorized_attention=True,
        num_classes=174, gdn_temporal_only=gdn_temporal_only, head_type=head_type)
    for blk in m.backbone.processor_blocks:
        if blk.block_type == "gdn":
            blk.layer.compute_path = compute_path
    return m


def get_model(head_type="mean", gdn_temporal_only=True, verbose=True):
    m = build_model(head_type, gdn_temporal_only)
    if os.path.exists(MLX_W):
        m.load_weights(MLX_W)
        if verbose: print(f"loaded cached MLX weights: {MLX_W}")
    else:
        import torch
        ck = torch.load(PT_CKPT, map_location="cpu", weights_only=False)
        M.load_pytorch_ssv2_checkpoint(m, ck["ema_state"]["shadow"],
                                       strict=True, verbose=False)
        m.save_weights(MLX_W)
        if verbose: print(f"converted PT->MLX (EMA shadow), cached to {MLX_W}")
    mx.eval(m.parameters())
    return m


def val_items(n=None, seed=0):
    id2lab, classes = D.load_ssv2_label_map(ANN, split="validation")
    items = sorted(id2lab.items())          # deterministic
    rng = random.Random(seed)
    rng.shuffle(items)
    if n: items = items[:n]
    return items, classes


def decode_eval(vid, num_frames=32):
    p = os.path.join(VIDEO_DIR, f"{vid}.webm")
    return D.decode_clip(p, num_frames=num_frames, img_size=224,
                         train=False, normalize=True)   # (T,H,W,C) float32


def evaluate(model, n=200, batch=4, seed=0, num_frames=32, frame_op=None):
    """Top-1/Top-5 on n seeded val clips. frame_op(clip)->clip optional input hook."""
    items, _ = val_items(n, seed)
    top1 = top5 = tot = 0
    t0 = time.time()
    buf_x, buf_y = [], []
    def flush():
        nonlocal top1, top5, tot
        if not buf_x: return
        x = mx.array(np.stack(buf_x))            # (B,T,H,W,C)
        logits = np.array(model(x))
        mx.eval(logits) if isinstance(logits, mx.array) else None
        top5_idx = np.argsort(-logits, axis=1)[:, :5]
        for i, y in enumerate(buf_y):
            top1 += int(top5_idx[i, 0] == y)
            top5 += int(y in top5_idx[i])
            tot += 1
        buf_x.clear(); buf_y.clear()
    for j, (vid, lab) in enumerate(items):
        try:
            clip = decode_eval(vid, num_frames)
            if frame_op is not None: clip = frame_op(clip)
            buf_x.append(clip); buf_y.append(lab)
        except Exception as e:
            continue
        if len(buf_x) == batch: flush()
        if (j + 1) % 50 == 0:
            r = 100.0 * top1 / max(tot, 1)
            print(f"  [{j+1}/{len(items)}] top1={r:.2f}% ({tot} scored, {time.time()-t0:.0f}s)")
    flush()
    return dict(top1=100.0*top1/max(tot,1), top5=100.0*top5/max(tot,1), n=tot,
                secs=time.time()-t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", default="mean")
    ap.add_argument("--temporal_only", type=int, default=1)
    a = ap.parse_args()
    m = get_model(head_type=a.head, gdn_temporal_only=bool(a.temporal_only))
    print(f"\nval accuracy (n={a.n}, head={a.head}, gdn_temporal_only={a.temporal_only}):")
    r = evaluate(m, n=a.n, batch=a.batch, seed=a.seed)
    print(f"\n  Top-1: {r['top1']:.2f}%   Top-5: {r['top5']:.2f}%   "
          f"(N={r['n']}, {r['secs']:.0f}s)")
