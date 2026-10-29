"""Cache LeWM-encoded latents for expert HDF5 datasets.

Loads the LeWM model checkpoint as a pickled module (matches what eval.py
does via stable_worldmodel.policy.AutoCostModel), uses model.backbone (HF
ViT) to encode each frame, and writes a new HDF5 with:

    latent      (N, D)         float16 (mean-pooled patch tokens by default)
    ep_idx, ep_len, ep_offset, valid_len, ep_success, subtask_id, ...
                                copied from source if present.

Usage (on a GPU node, after `module load cuda/12.6.1`):

    python scripts/cache_latents.py --dataset cube --limit 500 --batch_size 64
    python scripts/cache_latents.py --dataset cube
    python scripts/cache_latents.py --dataset scene --pool patches
"""

import argparse
import os
import sys
import time
from pathlib import Path

# The pickled checkpoint references `jepa.JEPA` (a module that lives in the
# project root, NOT in stable_worldmodel). Add project root to sys.path so
# torch.load can resolve the class regardless of where this script is run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hdf5plugin  # noqa: F401  — registers Blosc/LZ4 codecs for compressed h5 datasets
import h5py
import numpy as np
import torch
from tqdm import tqdm

# Match eval.py import order — these pull in the same chain that eval.py uses
import stable_pretraining as spt
import stable_worldmodel as swm  # noqa: F401  (needed so torch.load can resolve classes)
import jepa  # noqa: F401  (provides jepa.JEPA class for unpickling)


_HOME = os.environ["STABLEWM_HOME"]
CKPT_PATH = os.path.join(_HOME, 'checkpoints/cube_wm/lewm_object.ckpt')

DATASETS = {
    'cube':  os.path.join(_HOME, 'datasets/cube_single_expert_c0.3.h5'),
    'scene': os.path.join(_HOME, 'datasets/scene_expert_c0.3.h5'),
}

OUTPUT_DIR = os.path.join(_HOME, 'latents')


def get_imagenet_stats(device, dtype):
    """Match the normalization used in eval.py:
    transforms.Normalize(**spt.data.dataset_stats.ImageNet)
    """
    stats = spt.data.dataset_stats.ImageNet
    mean = torch.tensor(stats['mean'], device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(stats['std'], device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def load_lewm(ckpt_path, device):
    """Load the pickled JEPA module from a *_object.ckpt file.

    Returns (full_model, encoder, projector). The downstream latent used by
    LeWM is `projector(encoder(pixels).last_hidden_state[:, 0])` (see jepa.py).
    """
    print(f'Loading checkpoint: {ckpt_path}')
    model = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    # Fix ViT attributes for cross-version pickle compatibility
    import transformers
    for m in model.modules():
        if hasattr(m, 'config') and isinstance(m.config, transformers.PretrainedConfig):
            fresh = type(m.config)()
            for k, v in fresh.__dict__.items():
                if k not in m.config.__dict__:
                    setattr(m.config, k, v)
        if 'ViTSelfAttention' in type(m).__name__ and not hasattr(m, 'dropout'):
            m.dropout = torch.nn.Dropout(getattr(m, 'dropout_prob', 0.0))
    print(f'  loaded module type: {type(model).__name__}')
    print(f'  children: {[n for n, _ in model.named_children()]}')

    if not hasattr(model, 'encoder'):
        raise RuntimeError(f'No .encoder attribute on {type(model).__name__}')
    encoder = model.encoder
    projector = getattr(model, 'projector', None)
    print(f'  encoder type: {type(encoder).__name__}')
    print(f'  projector type: {type(projector).__name__ if projector is not None else "None"}')

    encoder = encoder.to(device).eval()
    encoder.requires_grad_(False)
    if projector is not None:
        projector = projector.to(device).eval()
        projector.requires_grad_(False)
    return model, encoder, projector


@torch.no_grad()
def encode_batch(encoder, projector, pixels_uint8, device, dtype, mean, std, pool):
    """pixels_uint8: (B, H, W, 3) uint8 -> latent np array.

    pool semantics:
      'lewm'     -> projector(cls_token)   (matches jepa.py JEPA.encode, RECOMMENDED)
      'cls'      -> raw cls token (no projector)
      'mean'     -> mean of patch tokens (no projector)
      'cls_mean' -> concat(cls, mean_patches)
      'patches'  -> full patch sequence (B, P, D)
    """
    x = torch.from_numpy(pixels_uint8).to(device, non_blocking=True)
    x = x.to(dtype) / 255.0
    x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
    x = (x - mean) / std

    out = encoder(x, interpolate_pos_encoding=True)
    hs = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out

    if pool == 'lewm':
        z = hs[:, 0]                                  # CLS
        if projector is not None:
            z = projector(z)                          # projected
    elif pool == 'cls':
        z = hs[:, 0]
    elif pool == 'mean':
        z = hs[:, 1:].mean(dim=1)
    elif pool == 'cls_mean':
        z = torch.cat([hs[:, 0], hs[:, 1:].mean(dim=1)], dim=-1)
    elif pool == 'patches':
        z = hs[:, 1:]
    else:
        raise ValueError(f'unknown pool: {pool}')

    return z.float().cpu().numpy().astype(np.float16)


def copy_field(src, dst, name, optional=False):
    if name in src:
        dst.create_dataset(name, data=src[name][:])
        return True
    if not optional:
        print(f'  [warn] required field missing: {name}')
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cube', 'scene', 'pusht'], required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pool', choices=['lewm', 'mean', 'cls', 'cls_mean', 'patches'],
                        default='lewm',
                        help='lewm = projector(cls_token), matches jepa.py JEPA.encode (recommended)')
    parser.add_argument('--ckpt', type=str, default=CKPT_PATH)
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--limit', type=int, default=None, help='encode only first N frames (debug)')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='shard start (inclusive) — run K parallel jobs on disjoint ranges')
    parser.add_argument('--end_frame', type=int, default=None, help='shard end (exclusive)')
    parser.add_argument('--prefetch', type=int, default=4,
                        help='reader-thread queue depth (overlaps h5 read/decompress with GPU)')
    parser.add_argument('--fp32', action='store_true',
                        help='force float32 compute (some ViT ckpts produce NaN in fp16 half)')
    args = parser.parse_args()

    src_path = args.source or DATASETS[args.dataset]
    out_path = args.output or str(Path(OUTPUT_DIR) / f'{args.dataset}_latents_{args.pool}.h5')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f'Source:  {src_path}')
    print(f'Output:  {out_path}')
    print(f'Device:  {args.device}')
    print(f'Pool:    {args.pool}')
    print(f'Batch:   {args.batch_size}')

    # Load model
    use_half = args.device.startswith('cuda') and not args.fp32
    dtype = torch.float16 if use_half else torch.float32
    _, encoder, projector = load_lewm(args.ckpt, args.device)
    if use_half:
        encoder = encoder.half()
        if projector is not None:
            projector = projector.half()
    mean, std = get_imagenet_stats(args.device, dtype)

    # Open source / output
    with h5py.File(src_path, 'r') as src:
        pixels = src['pixels']
        N = pixels.shape[0] if args.limit is None else min(pixels.shape[0], args.limit)
        s0 = max(0, int(args.start_frame))
        s1 = N if args.end_frame is None else min(N, int(args.end_frame))
        print(f'Total frames: {N}  (shard [{s0}, {s1}))')

        # Probe latent shape with a small forward
        probe = encode_batch(encoder, projector, pixels[:2], args.device, dtype, mean, std, args.pool)
        print(f'Probe latent shape: {probe.shape}, dtype: {probe.dtype}')
        print(f'Probe latent stats: mean={probe.mean():.3f}, std={probe.std():.3f}, '
              f'min={probe.min():.3f}, max={probe.max():.3f}')

        latent_shape = (s1 - s0,) + probe.shape[1:]
        with h5py.File(out_path, 'w') as dst:
            chunks = (min(1024, N),) + probe.shape[1:]
            latent_ds = dst.create_dataset(
                'latent', shape=latent_shape, dtype=np.float16,
                chunks=chunks, compression='lzf',
            )

            copy_field(src, dst, 'ep_idx')
            copy_field(src, dst, 'ep_len')
            copy_field(src, dst, 'ep_offset')
            copy_field(src, dst, 'valid_len', optional=True)
            copy_field(src, dst, 'ep_success', optional=True)
            copy_field(src, dst, 'subtask_id', optional=True)
            copy_field(src, dst, 'subtask_end_flag', optional=True)
            copy_field(src, dst, 'ep_physical_success', optional=True)
            # density-proxy corpus fields (carried so train/validate scripts never
            # reopen the 1.2M-frame pixel file). All optional → cube/scene unaffected.
            copy_field(src, dst, 'action', optional=True)
            copy_field(src, dst, 'episode_idx', optional=True)
            copy_field(src, dst, 'step_idx', optional=True)
            copy_field(src, dst, 'human_demo', optional=True)

            t0 = time.time()
            # Producer thread reads+decompresses the NEXT slices while the GPU
            # encodes the current one (the raw loop serializes IO and compute —
            # the h5 per-frame decompress starves the GPU).
            import queue as _q, threading as _th
            Q = _q.Queue(maxsize=max(1, int(args.prefetch)))

            def _reader():
                for start in range(s0, s1, args.batch_size):
                    end = min(start + args.batch_size, s1)
                    Q.put((start, np.asarray(pixels[start:end])))
                Q.put(None)

            _t = _th.Thread(target=_reader, daemon=True); _t.start()
            done = 0
            with tqdm(total=s1 - s0, desc='encoding') as bar:
                while True:
                    item = Q.get()
                    if item is None:
                        break
                    start, px = item
                    z = encode_batch(encoder, projector, px, args.device, dtype, mean, std, args.pool)
                    latent_ds[start - s0:start - s0 + len(z)] = z
                    done += len(z); bar.update(len(z))
            dt = time.time() - t0
            print(f'Encoded {done} frames in {dt:.1f}s ({done / max(dt, 1e-9):.0f} fps)')

            dst.attrs['source'] = src_path
            dst.attrs['shard_start'] = s0
            dst.attrs['shard_end'] = s1
            dst.attrs['ckpt'] = args.ckpt
            dst.attrs['pool'] = args.pool
            dst.attrs['latent_shape'] = list(probe.shape[1:])
            dst.attrs['encoder'] = 'lewm_jepa_cube'

    print(f'Done. Latents saved to {out_path}')


if __name__ == '__main__':
    main()
