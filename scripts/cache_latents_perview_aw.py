"""Cache perview-AW (agentview + wrist) latents for MV diffusion HDF5 datasets.

Loads a perview JEPA `*_object.ckpt` (saved by trainer w/ tokens.agent and
tokens.wrist enabled), uses model.encoder (ViT) + model.projector (192-d MLP)
to encode each agentview frame and each wrist frame, then concatenates
along the last dim to produce a (N, 384) latent.

This mirrors `scripts/cache_latents.py` but:
  - Reads BOTH `pixels` (agentview) and `wrist_pixels` from the source h5.
  - Output latent dim is 192 + 192 = 384 (concat in canonical [A; W] order).
  - Uses the SAME ViT encoder and SAME projector for both views (perview JEPA
    shares them by design — see jepa_perview.py:387-396).

Usage (sbatch only — login node will bus-error on torch import):

    python scripts/cache_latents_perview_aw.py \\
        --source "$STABLEWM_HOME/datasets/cube_multiview_diffusion_c0.3_postpad_fullscale.h5" \\
        --output "$STABLEWM_HOME/cube/latents/cube_multiview_diffusion_aw_c0.3_postpad_fullscale.h5" \\
        --ckpt "$STABLEWM_HOME/checkpoints/cube_wm/lewm_pv_aw_cube_scratch_epoch_11_object.ckpt" \\
        --batch_size 64
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import torch
from tqdm import tqdm

import stable_pretraining as spt
import stable_worldmodel as swm  # noqa: F401  needed for unpickle class resolution
import jepa  # noqa: F401  base JEPA class
import jepa_perview  # noqa: F401  PerViewJEPA class for unpickle


def get_imagenet_stats(device, dtype):
    stats = spt.data.dataset_stats.ImageNet
    mean = torch.tensor(stats['mean'], device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(stats['std'], device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def load_perview(ckpt_path, device):
    """Load pickled PerViewJEPA module.

    Returns (encoder, projector). Both are shared between agent and wrist.
    """
    print(f'Loading perview ckpt: {ckpt_path}')
    model = torch.load(ckpt_path, weights_only=False, map_location='cpu')

    # Cross-version pickle compatibility (mirrors cache_latents.py)
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
    if hasattr(model, 'enabled_views'):
        print(f'  enabled_views    : {model.enabled_views}')
    print(f'  children: {[n for n, _ in model.named_children()]}')

    if not hasattr(model, 'encoder') or not hasattr(model, 'projector'):
        raise RuntimeError(
            f'Expected .encoder and .projector on {type(model).__name__}; got '
            f'{[n for n, _ in model.named_children()]}'
        )
    if model.projector is None:
        raise RuntimeError('model.projector is None — perview AW requires projected embeddings.')

    encoder = model.encoder.to(device).eval().requires_grad_(False)
    projector = model.projector.to(device).eval().requires_grad_(False)
    return encoder, projector


@torch.no_grad()
def encode_view_batch(encoder, projector, pixels_uint8, device, dtype, mean, std):
    """pixels_uint8: (B, H, W, 3) uint8 → (B, 192) projected CLS embedding."""
    x = torch.from_numpy(pixels_uint8).to(device, non_blocking=True)
    x = x.to(dtype) / 255.0
    x = x.permute(0, 3, 1, 2)        # BHWC -> BCHW
    x = (x - mean) / std
    cls = encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]   # (B, hidden)
    z = projector(cls)               # (B, 192)
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
    parser.add_argument('--source', required=True, help='source MV diffusion h5 with pixels + wrist_pixels')
    parser.add_argument('--output', required=True, help='destination latent h5 (latent dim = 384)')
    parser.add_argument('--ckpt', required=True, help='perview JEPA *_object.ckpt')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--limit', type=int, default=None, help='encode first N frames only (debug)')
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f'Source : {args.source}')
    print(f'Output : {args.output}')
    print(f'Ckpt   : {args.ckpt}')
    print(f'Device : {args.device}')
    print(f'Batch  : {args.batch_size}')

    use_half = args.device.startswith('cuda')
    dtype = torch.float16 if use_half else torch.float32
    encoder, projector = load_perview(args.ckpt, args.device)
    if use_half:
        encoder = encoder.half()
        projector = projector.half()
    mean, std = get_imagenet_stats(args.device, dtype)

    with h5py.File(args.source, 'r') as src:
        if 'pixels' not in src or 'wrist_pixels' not in src:
            raise RuntimeError(
                f"Source h5 missing 'pixels' or 'wrist_pixels' (got: {list(src.keys())})"
            )
        agent = src['pixels']
        wrist = src['wrist_pixels']
        assert agent.shape == wrist.shape, (
            f'pixels {agent.shape} != wrist_pixels {wrist.shape}'
        )
        N = agent.shape[0] if args.limit is None else min(agent.shape[0], args.limit)
        print(f'Total frames: {N}')

        # Probe shape via tiny forward
        z_a = encode_view_batch(encoder, projector, agent[:2], args.device, dtype, mean, std)
        z_w = encode_view_batch(encoder, projector, wrist[:2], args.device, dtype, mean, std)
        if z_a.shape[1] != z_w.shape[1]:
            raise RuntimeError(f'A latent dim {z_a.shape[1]} != W latent dim {z_w.shape[1]}')
        D_view = z_a.shape[1]
        D_total = 2 * D_view
        print(f'Per-view latent dim : {D_view}    Concat dim : {D_total}')
        print(f'Probe agent stats   : mean={z_a.mean():.3f}, std={z_a.std():.3f}')
        print(f'Probe wrist stats   : mean={z_w.mean():.3f}, std={z_w.std():.3f}')

        with h5py.File(args.output, 'w') as dst:
            chunks = (min(1024, N), D_total)
            latent_ds = dst.create_dataset(
                'latent', shape=(N, D_total), dtype=np.float16,
                chunks=chunks, compression='lzf',
            )

            copy_field(src, dst, 'ep_idx')
            copy_field(src, dst, 'ep_len')
            copy_field(src, dst, 'ep_offset')
            copy_field(src, dst, 'valid_len', optional=True)
            copy_field(src, dst, 'ep_success', optional=True)
            copy_field(src, dst, 'ep_physical_success', optional=True)
            copy_field(src, dst, 'subtask_id', optional=True)
            copy_field(src, dst, 'subtask_end_flag', optional=True)

            t0 = time.time()
            for start in tqdm(range(0, N, args.batch_size), desc='encoding AW'):
                end = min(start + args.batch_size, N)
                z_a = encode_view_batch(encoder, projector, agent[start:end], args.device, dtype, mean, std)
                z_w = encode_view_batch(encoder, projector, wrist[start:end], args.device, dtype, mean, std)
                latent_ds[start:end] = np.concatenate([z_a, z_w], axis=-1)
            dt = time.time() - t0
            print(f'Encoded {N} frames in {dt:.1f}s ({N / max(dt, 1e-9):.0f} fps)')

            dst.attrs['source'] = args.source
            dst.attrs['ckpt'] = args.ckpt
            dst.attrs['encoder'] = 'perview_lewm_aw'
            dst.attrs['per_view_dim'] = int(D_view)
            dst.attrs['layout'] = 'concat[agent;wrist]'

    print(f'Done. Latents saved to {args.output}')


if __name__ == '__main__':
    main()
