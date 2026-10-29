import os
from functools import partial
from pathlib import Path

# Patch torchvision v2 transforms for compatibility with stable_pretraining
# (older torchvision uses _transform, newer uses transform)
from torchvision.transforms import v2 as _v2_transforms
if not hasattr(_v2_transforms.Resize, 'transform') and hasattr(_v2_transforms.Resize, '_transform'):
    for _cls in [_v2_transforms.Resize, _v2_transforms.ColorJitter, _v2_transforms.RandomRotation,
                 _v2_transforms.GaussianBlur, _v2_transforms.RandomAffine]:
        if hasattr(_cls, '_transform') and not hasattr(_cls, 'transform'):
            _cls.transform = _cls._transform

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg

# repo-root utils.py is shadowed by the (empty) utils/ package dir — load the
# module explicitly by file path instead of `from utils import ...`.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_wm_train_utils", Path(__file__).resolve().parent / "utils.py")
_wm_train_utils = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_wm_train_utils)
get_column_normalizer = _wm_train_utils.get_column_normalizer
get_img_preprocessor = _wm_train_utils.get_img_preprocessor
ModelObjectCallBack = _wm_train_utils.ModelObjectCallBack

# Agent-encoding auxiliary loss: decode physical state from the latent to FORCE
# the WM to retain agent (pusher) info — the diagnosed binding constraint
# (slippery WM under-encodes agent: probe 71px vs plain 4px → phantom-push).
#
# RESULT (2026-06, INVESTIGATED-BUT-UNADOPTED): the aux loss did reshape the
# latent — agent decode probe dropped to ~1px (vs scr20's poor agent encoding) —
# and lifted GT-MPCC at reduced CEM (auxall slippery-050 SR 75% vs scr20 37.5%,
# n=16 so ±12pp noisy). BUT the aux-trained WM was WORSE than the baseline WM on
# DTG-MPCC (diffusion-MPCC), so the aux WM is NOT used in the final pipeline.
# Agent-encoding turned out task-dependent (helps GT-MPCC, hurts DTG-MPCC), not a
# clean win. This loss + the auxagent/auxall train configs are kept only to
# document the lever; default OFF leaves the WM bit-exact. (See memory
# project_slippery_baselines_corpus_2026_06_26.)
#
# state layout (pusht): [agent_x, agent_y, block_x, block_y, block_angle, ...].
# spec: target -> (state slice, head out_dim, kind). angle uses sin/cos.
AUX_SPEC = {
    "agent": (slice(0, 2), 2, "xy"),
    "block": (slice(2, 4), 2, "xy"),
    "angle": (slice(4, 5), 2, "sincos"),
}
AUX_XY_MU, AUX_XY_SD = 256.0, 150.0   # arena-center normalization (px → conditioned)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss. Optional delta-normalized variant (+loss.delta_norm=true):
    # raw MSE is dominated by large-|Δz| samples (noise-play moves 5-10× more
    # latent distance than demos), so the mixture optimum sacrifices small-
    # motion precision — exactly the demo-region regression seen at G2.
    # Normalizing per (sample,step) by ‖Δz‖² optimizes RELATIVE error, putting
    # demo-pace and noise-pace transitions on equal footing.
    if cfg.loss.get("delta_norm", False):
        prev_emb = emb[:, :-n_preds] if n_preds > 0 else emb
        delta_sq = (tgt_emb - prev_emb[:, -tgt_emb.shape[1]:]).pow(2).mean(
            dim=-1, keepdim=True)
        err_sq = (pred_emb - tgt_emb).pow(2).mean(dim=-1, keepdim=True)
        # Inverse-delta weights, batch-normalized to mean 1 and clamped:
        # relative-error emphasis WITHOUT near-zero-delta (stationary/padded)
        # samples dominating — raw 1/(d+1e-4) gave them ~1e4× weight, which
        # destroyed the warm start (raw err stuck at 0.2 vs 0.03, job 9789943).
        # Mean-1 weights also keep the loss on the raw-MSE scale (LR semantics).
        w = 1.0 / (delta_sq + 1e-3)
        w = (w / w.mean().clamp(min=1e-8)).clamp(max=20.0).detach()
        output["pred_loss"] = (w * err_sq).mean()
        output["pred_loss_raw"] = err_sq.mean().detach()
    else:
        output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    # ── Agent-encoding aux loss + monitoring. Default OFF → WM bit-exact:
    # monitor heads read emb.detach() (no WM gradient); supervised targets feed
    # gradient into the WM via emb (current) + pred_emb (future) to FORCE
    # agent/contact encoding. Logs decode error in interpretable px/deg.
    aux_cfg = cfg.loss.get("aux_state", None)
    if aux_cfg is not None and hasattr(self.model, "aux_heads") and "state" in batch:
        state = torch.nan_to_num(batch["state"].float(), 0.0)        # (B,T,7)
        heads = self.model.aux_heads
        sup = set(aux_cfg.get("targets", [])) if aux_cfg.get("enabled", False) else set()
        lam = float(aux_cfg.get("weight", 1.0))
        abd = (state[..., 0:2] - state[..., 2:4]).pow(2).sum(-1).sqrt()   # agent-block dist (B,T)
        near = abd < 50.0
        on_s = (stage == "train"); on_e = not on_s   # exactly one True (val/sanity → epoch)
        sup_loss = pred_emb.new_zeros(())
        mon_loss = pred_emb.new_zeros(())
        for t, head in heads.items():
            sl, _odim, kind = AUX_SPEC[t]
            gt = state[..., sl]
            if kind == "xy":
                tgt = (gt - AUX_XY_MU) / AUX_XY_SD
            else:                                                    # sincos angle
                a = gt[..., 0]
                tgt = torch.stack([torch.sin(a), torch.cos(a)], dim=-1)
            if t in sup:                                             # gradient → WM
                sup_loss = sup_loss + (head(emb) - tgt).pow(2).mean() \
                    + (head(pred_emb) - tgt[:, n_preds:]).pow(2).mean()
            else:                                                    # monitor only (detached)
                mon_loss = mon_loss + (head(emb.detach()) - tgt).pow(2).mean()
            with torch.no_grad():                                    # interpretable logging
                p = head(emb)
                if kind == "xy":
                    e_px = ((p - tgt) * AUX_XY_SD).pow(2).sum(-1).sqrt()
                    self.log(f"{stage}/{t}_err_px", e_px.mean(), on_step=on_s, on_epoch=on_e, sync_dist=True)
                    if t == "agent":
                        for msk, nm in ((near, "near"), (~near, "far")):
                            v = e_px[msk].mean() if msk.any() else e_px.new_zeros(())
                            self.log(f"{stage}/agent_err_{nm}_px", v, on_step=on_s, on_epoch=on_e, sync_dist=True)
                else:
                    da = torch.atan2(p[..., 0], p[..., 1]) - gt[..., 0]
                    da = torch.abs((da + torch.pi) % (2 * torch.pi) - torch.pi) * (180.0 / torch.pi)
                    self.log(f"{stage}/{t}_err_deg", da.mean(), on_step=on_s, on_epoch=on_e, sync_dist=True)
        if sup:
            output["aux_sup_loss"] = sup_loss
            output["loss"] = output["loss"] + lam * sup_loss
        output["aux_mon_loss"] = mon_loss                            # trains monitor heads (WM detached)
        output["loss"] = output["loss"] + mon_loss

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

class FreezeEvalCallback(pl.Callback):
    """Keep frozen encoder/projector in eval mode every epoch (BN stats frozen).

    Module-level class (NOT an instance-attribute closure) so the model object
    stays picklable — a local `_train` monkeypatch broke ModelObjectCallBack's
    per-epoch torch.save ("Can't pickle local object ... _train")."""

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.model.encoder.eval()
        pl_module.model.projector.eval()


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    # Optional fixed held-out validation file (+data.val_dataset_name=<name>):
    # train on ALL windows of the training file and validate on a separate,
    # episode-disjoint file normalized with the TRAINING set's normalizers.
    # The data-scaling sweep uses it so every subset size is scored on one
    # common, leak-free val set (the default 90/10 split is window-level, so
    # overlapping windows of one episode land on both sides). Unset = original.
    val_name = cfg.data.get("val_dataset_name", None)
    if val_name:
        _vkw = {k: cfg.data.dataset[k] for k in cfg.data.dataset}
        _vkw["name"] = val_name
        val_set = swm.data.HDF5Dataset(**_vkw, transform=None)
        val_set.transform = transform
        train_set = dataset
        print(f"[val] external held-out val '{val_name}': {len(val_set)} windows; "
              f"training on all {len(train_set)} windows of '{cfg.data.dataset.name}'")
    else:
        train_set, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
        )

    # Optional fixed-length epochs (+steps_per_epoch=N): each epoch draws N
    # batches with replacement, so validation and the per-epoch object dump run
    # every N optimizer steps whatever the dataset size (a tiny subset would
    # otherwise run hundreds of epochs, each paying a val pass and a checkpoint
    # write). The LR schedule follows estimated_stepping_batches = N*max_epochs.
    spe = cfg.get("steps_per_epoch", None)
    if spe:
        _sampler = torch.utils.data.RandomSampler(
            train_set, replacement=True,
            num_samples=int(spe) * int(cfg.loader.batch_size), generator=rnd_gen)
        train = torch.utils.data.DataLoader(
            train_set, **cfg.loader, sampler=_sampler, drop_last=True)
        print(f"[steps_per_epoch] {int(spe)} steps/epoch, sampled with replacement "
              f"from {len(train_set)} training windows")
    else:
        train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    # -- optional fine-tune from an existing *_object.ckpt --
    # +finetune_from=<path to lewm_object.ckpt>  +freeze_encoder=true
    # The object ckpt IS the pickled JEPA module — use it directly as the model
    # (guarantees identical architecture and skips vit_hf construction, whose
    # `from transformers import TimmWrapperModel ...` import fails in this env).
    ft_path = cfg.get("finetune_from", None)
    if ft_path:
        world_model = torch.load(str(ft_path), map_location="cpu", weights_only=False)
        assert isinstance(world_model, JEPA), f"unexpected ckpt object: {type(world_model)}"
        # HF ViT compat patch — the ckpt was pickled with a NEWER transformers
        # than this env's 4.46.3. Canonical shim copied from policy/mpc.py:75-99
        # (same as MPCCPolicy / DPPolicy — what makes eval work with this ckpt):
        # 1) fill missing PretrainedConfig fields from a fresh instance,
        # 2) re-attach the dropout submodule newer ViTSelfAttention dropped.
        import transformers as _hf_transformers
        for _m in world_model.modules():
            if hasattr(_m, "config") and isinstance(_m.config, _hf_transformers.PretrainedConfig):
                _fresh = type(_m.config)()
                for _k, _v in _fresh.__dict__.items():
                    if _k not in _m.config.__dict__:
                        setattr(_m.config, _k, _v)
            if "ViTSelfAttention" in type(_m).__name__ and not hasattr(_m, "dropout"):
                _m.dropout = torch.nn.Dropout(getattr(_m, "dropout_prob", 0.0))
        print(f"[finetune] using pickled JEPA from {ft_path} as the model "
              f"({sum(p.numel() for p in world_model.parameters())/1e6:.2f}M params); "
              f"HF ViT compat shim applied (policy/mpc.py parity)")
    else:
        try:
            encoder = spt.backbone.utils.vit_hf(
                cfg.encoder_scale,
                patch_size=cfg.patch_size,
                image_size=cfg.img_size,
                pretrained=False,
                use_mask_token=False,
            )
        except ImportError:
            # spt.vit_hf needs `transformers.TimmWrapperModel` (absent in this
            # env's 4.46.3). Build the same ViT directly; arch parity with the
            # v1 ckpt is verified by state_dict key/shape comparison (see
            # preverify_scratch job / memory project_wm_v2ft_pipeline).
            from transformers import ViTConfig as _VC, ViTModel as _VM
            _sizes = {
                "tiny":  dict(hidden_size=192, num_hidden_layers=12,
                              num_attention_heads=3, intermediate_size=768),
                "small": dict(hidden_size=384, num_hidden_layers=12,
                              num_attention_heads=6, intermediate_size=1536),
            }
            _vcfg = _VC(
                **_sizes[cfg.encoder_scale],
                patch_size=cfg.patch_size,
                image_size=cfg.img_size,
            )
            encoder = _VM(_vcfg, add_pooling_layer=False, use_mask_token=True)
            print(f"[scratch] built ViT-{cfg.encoder_scale} via local ViTModel "
                  f"fallback (spt.vit_hf ImportError)")

        hidden_dim = encoder.config.hidden_size
        embed_dim = cfg.wm.get("embed_dim", hidden_dim)
        effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

        predictor = ARPredictor(
            num_frames=cfg.wm.history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **cfg.predictor,
        )

        action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

        projector = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )

        predictor_proj = MLP(
            input_dim=hidden_dim,
            output_dim=embed_dim,
            hidden_dim=2048,
            norm_fn=torch.nn.BatchNorm1d,
        )

        world_model = JEPA(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=predictor_proj,
        )

    # freeze_encoder freezes encoder+projector (params AND BatchNorm running
    # stats, via persistent eval mode) so the 192-d latent space is bit-preserved
    # — existing refs/decoders/baselines stay comparable; only the dynamics side
    # (predictor/action_encoder/pred_proj) learns from the new corpus.
    freeze_encoder = bool(cfg.get("freeze_encoder", False))
    if freeze_encoder:
        assert ft_path, "freeze_encoder without finetune_from would freeze random weights"
        for _m in (world_model.encoder, world_model.projector):
            for _p in _m.parameters():
                _p.requires_grad_(False)
            _m.eval()
        # BN-stats freezing is re-applied each epoch by FreezeEvalCallback
        # (added to trainer callbacks below) — an instance-level train()
        # monkeypatch here would make the model unpicklable.
        _n_tr = sum(p.numel() for p in world_model.parameters() if p.requires_grad)
        _n_all = sum(p.numel() for p in world_model.parameters())
        print(f"[finetune] froze encoder+projector (eval via FreezeEvalCallback); "
              f"trainable {_n_tr/1e6:.2f}M / {_n_all/1e6:.2f}M params")

    # -- optional cold dynamics (+reinit_dynamics=true, needs finetune_from) --
    # finetune_from uses the WHOLE donor JEPA, so its predictor / action_encoder /
    # pred_proj start from dynamics already trained on the donor's full play
    # corpus. This replaces those three with fresh modules built by the same
    # constructors as the from-scratch branch above, while the donor's
    # encoder+projector (and so the latent space every DTG / warm-ring cache
    # lives in) are kept. Shapes are checked against the donor so the only
    # difference is the initialization.
    if bool(cfg.get("reinit_dynamics", False)):
        assert ft_path, "reinit_dynamics needs finetune_from (source of the encoder)"
        torch.manual_seed(int(cfg.seed))
        _hid = world_model.encoder.config.hidden_size
        _emb = cfg.wm.get("embed_dim", _hid)
        _fresh = {
            "predictor": ARPredictor(
                num_frames=cfg.wm.history_size, input_dim=_emb,
                hidden_dim=_hid, output_dim=_hid, **cfg.predictor),
            "action_encoder": Embedder(
                input_dim=cfg.data.dataset.frameskip * cfg.wm.action_dim, emb_dim=_emb),
            "pred_proj": MLP(
                input_dim=_hid, output_dim=_emb, hidden_dim=2048,
                norm_fn=torch.nn.BatchNorm1d),
        }
        for _name, _new in _fresh.items():
            _old_sd = getattr(world_model, _name).state_dict()
            _new_sd = _new.state_dict()
            _bad = sorted(k for k in set(_old_sd) | set(_new_sd)
                          if k not in _old_sd or k not in _new_sd
                          or _old_sd[k].shape != _new_sd[k].shape)
            assert not _bad, (f"reinit_dynamics: fresh {_name} does not match the "
                              f"donor's architecture: {_bad[:6]}")
            setattr(world_model, _name, _new)
        _n_tr = sum(p.numel() for p in world_model.parameters() if p.requires_grad)
        print(f"[reinit_dynamics] predictor/action_encoder/pred_proj re-initialized "
              f"(shapes match the donor); encoder+projector kept from {ft_path}; "
              f"trainable {_n_tr/1e6:.2f}M params")

    # Aux decode heads (agent/block/angle ← latent). Attached as submodules of
    # the JEPA model so the 'model' optimizer below trains them. Monitor heads
    # always read DETACHED latent (zero WM gradient → WM bit-exact); supervised
    # targets (loss.aux_state.targets, only when enabled) feed gradient into the
    # WM. Present only when the config declares loss.aux_state → other tasks/
    # baselines without the block are 100% untouched.
    if cfg.loss.get("aux_state", None) is not None:
        _emb_d = cfg.wm.get("embed_dim", 192)
        _heads = torch.nn.ModuleDict(
            {_t: torch.nn.Linear(_emb_d, AUX_SPEC[_t][1]) for _t in AUX_SPEC})
        world_model.aux_heads = _heads
        print(f"[aux] decode heads attached: {list(_heads)} (emb_d={_emb_d}); "
              f"supervised={list(cfg.loss.aux_state.get('targets', [])) if cfg.loss.aux_state.get('enabled', False) else '[]'}")

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    _callbacks = [object_dump_callback]
    if freeze_encoder:
        _callbacks.append(FreezeEvalCallback())

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=_callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
