"""Train per-view LeWM on OGBench Cube (agent + wrist + proprio tokens, no fusion)."""

import os
from functools import partial
from pathlib import Path

# torchvision compat shim (same as multiview trainer)
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
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa_perview import (
    PerViewJEPA, PerViewARPredictor, ProprioEncoder, GraspAuxHead, VIEW_NAMES,
)
from module import Embedder, MLP, SIGReg
# repo-root utils.py is shadowed by the (empty) utils/ package dir — load the
# module explicitly by file path instead of `from utils import ...`. Same shim
# as train.py:25-34. le-wm has no utils/__init__.py, so the plain import worked
# there and only broke on the port into le_wm_v2 (job 11582698).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_wm_train_utils", Path(__file__).resolve().parent / "utils.py")
_wm_train_utils = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_wm_train_utils)
get_column_normalizer = _wm_train_utils.get_column_normalizer
get_img_preprocessor = _wm_train_utils.get_img_preprocessor
ModelObjectCallBack = _wm_train_utils.ModelObjectCallBack


# ---------------------------------------------------------------------------
# Custom transform: compute grasp_label BEFORE normalizers
# ---------------------------------------------------------------------------

class GraspLabelTransform:
    """Computes grasp_label from raw inputs and adds it to batch.

    Must run BEFORE normalizer transforms (which would alter contact/distance
    semantics).
    """

    def __init__(self, contact_key, contact_threshold, eff_key, cube_key, prox_threshold):
        self.contact_key = contact_key
        self.contact_threshold = float(contact_threshold)
        self.eff_key = eff_key
        self.cube_key = cube_key
        self.prox_threshold = float(prox_threshold)

    def __call__(self, sample):
        if self.contact_key not in sample:
            return sample
        contact = sample[self.contact_key]
        if not torch.is_tensor(contact):
            contact = torch.as_tensor(contact)
        contact = contact.float()
        if contact.dim() > 0 and contact.shape[-1] == 1:
            contact = contact.squeeze(-1)
        eff = torch.as_tensor(sample[self.eff_key]).float()
        cube = torch.as_tensor(sample[self.cube_key]).float()
        dist = (cube - eff).norm(dim=-1)
        label = ((contact > self.contact_threshold) & (dist < self.prox_threshold)).float()
        sample["grasp_label"] = label
        return sample


# ---------------------------------------------------------------------------
# Warm-start helper
# ---------------------------------------------------------------------------

class FreezeEvalCallback(pl.Callback):
    """Keep a frozen encoder/projector in eval mode every epoch (BN stats frozen).

    Ported verbatim from train.py:157-167. Module-level class (NOT an
    instance-attribute closure) so the model object stays picklable — a local
    `_train` monkeypatch breaks ModelObjectCallBack's per-epoch torch.save
    ("Can't pickle local object ... _train")."""

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.model.encoder.eval()
        pl_module.model.projector.eval()


def load_warmstart(model: nn.Module, ckpt_path: str, modules_to_load: dict) -> None:
    """Load specified submodules from a Lightning/spt checkpoint into `model`."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[warmstart] no ckpt at '{ckpt_path}' — full scratch.")
        return

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, nn.Module):
        # HF ViT cross-version pickle shim (train.py:213-225 parity). The object
        # ckpt was pickled with a newer transformers than this env's 4.46.3:
        # 1) fill missing PretrainedConfig fields from a fresh instance,
        # 2) re-attach the dropout submodule newer ViTSelfAttention dropped.
        # Only reachable on the *_object.ckpt path, which the state-dict path
        # never touched — so it arrived with that change, not before it.
        import transformers as _hf
        for _m in state.modules():
            if hasattr(_m, "config") and isinstance(_m.config, _hf.PretrainedConfig):
                _fresh = type(_m.config)()
                for _k, _v in _fresh.__dict__.items():
                    if _k not in _m.config.__dict__:
                        setattr(_m.config, _k, _v)
            if "ViTSelfAttention" in type(_m).__name__ and not hasattr(_m, "dropout"):
                _m.dropout = torch.nn.Dropout(getattr(_m, "dropout_prob", 0.0))
        # *_object.ckpt is the pickled module, not a state dict. Needed because
        # the cube champion is selected per-EPOCH (epoch_11), and only the
        # object dumps are per-epoch — *_weights.ckpt is the last epoch only, so
        # warm-starting from it would freeze a different encoder than the one
        # the champion eval actually used.
        print(f"[warmstart] ckpt is a pickled {type(state).__name__}; using .state_dict()")
        state = state.state_dict()
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Strip 'model.' prefix from spt.Module wrapper
    cleaned = {}
    for k, v in state.items():
        if k.startswith("model."):
            cleaned[k[len("model."):]] = v
        else:
            cleaned[k] = v

    # Filter to only modules we want to load
    targets = [name for name, load in modules_to_load.items() if load]
    filtered = {}
    for k, v in cleaned.items():
        for t in targets:
            if k.startswith(f"{t}."):
                filtered[k] = v
                break

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[warmstart] loaded {len(filtered)} keys from {ckpt_path}")
    print(f"[warmstart]   targets enabled: {targets}")
    print(f"[warmstart]   missing (expected for scratch modules): {len(missing)}")
    if len(unexpected) > 0:
        print(f"[warmstart]   unexpected (ignored): {len(unexpected)} (e.g. {unexpected[:3]})")


# ---------------------------------------------------------------------------
# Loss / forward
# ---------------------------------------------------------------------------

def lejepa_forward(self, batch, stage, cfg):
    """Per-view JEPA forward + multi-component loss."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    sigreg_w = cfg.loss.sigreg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    # 1) Encode all enabled views
    output = self.model.encode(batch)
    enabled_views = self.model.enabled_views

    # 2) Predict next-step per-view tokens
    ctx_per_view = {v: output[f"emb_{v}"][:, :ctx_len] for v in enabled_views}
    tgt_per_view = {v: output[f"emb_{v}"][:, n_preds:] for v in enabled_views}

    pred_per_view = self.model.predict(ctx_per_view, output["act_emb"][:, :ctx_len])

    # 3) Per-view L2 prediction loss (weighted sum)
    pred_loss_total = 0.0
    per_view_loss_log = {}
    for v in enabled_views:
        weight = float(cfg.tokens[v].loss_weight)
        loss_v = (pred_per_view[v] - tgt_per_view[v]).pow(2).mean()
        pred_loss_total = pred_loss_total + weight * loss_v
        per_view_loss_log[f"{stage}/loss_pred_{v}"] = loss_v.detach()

    # 4) Per-view SIGReg (computed on full sequence, not just context)
    if cfg.loss.sigreg.get("per_view", True):
        sigreg_loss = sum(
            self.sigreg(output[f"emb_{v}"].transpose(0, 1)) for v in enabled_views
        )
    else:
        all_emb = output["emb"]   # (B, T, D_concat)
        sigreg_loss = self.sigreg(all_emb.transpose(0, 1))

    # 5) Auxiliary grasp head (optional)
    aux_grasp_loss = output["emb"].new_tensor(0.0)
    aux_grasp_log = {}
    if (cfg.aux_heads.grasp.enabled and self.model.grasp_aux_head is not None
            and "grasp_label" in batch):
        # Source: RAW encoder output for wrist (avoid proprio leak)
        source_view = cfg.aux_heads.grasp.source_view
        src = output[f"cls_emb_{source_view}"]    # (B, T, D)
        logits = self.model.grasp_aux_head(src)   # (B, T)
        labels = batch["grasp_label"].float()     # (B, T)
        if labels.dim() == 3 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
        pos_weight = torch.tensor(
            float(cfg.aux_heads.grasp.pos_weight), device=logits.device
        )
        bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        aux_grasp_loss = float(cfg.aux_heads.grasp.weight) * bce

        # Diagnostic metrics
        with torch.no_grad():
            preds = (logits > 0).float()
            tp = ((preds == 1) & (labels == 1)).float().sum()
            fp = ((preds == 1) & (labels == 0)).float().sum()
            fn = ((preds == 0) & (labels == 1)).float().sum()
            tn = ((preds == 0) & (labels == 0)).float().sum()
            recall = tp / (tp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
            aux_grasp_log[f"{stage}/grasp_bce"] = bce.detach()
            aux_grasp_log[f"{stage}/grasp_acc"] = acc.detach()
            aux_grasp_log[f"{stage}/grasp_recall"] = recall.detach()
            aux_grasp_log[f"{stage}/grasp_precision"] = precision.detach()

            # Batch-AUC (Mann-Whitney U on logits vs labels). Cheap for small batches.
            pos_logits = logits[labels == 1]
            neg_logits = logits[labels == 0]
            if pos_logits.numel() > 0 and neg_logits.numel() > 0:
                auc = (pos_logits[:, None] > neg_logits[None, :]).float().mean()
                aux_grasp_log[f"diag/grasp_auc"] = auc.detach()

    # 6) Total
    total_loss = pred_loss_total + sigreg_w * sigreg_loss + aux_grasp_loss

    output["pred_loss"] = pred_loss_total
    output["sigreg_loss"] = sigreg_loss
    output["aux_grasp_loss"] = aux_grasp_loss
    output["loss"] = total_loss

    # 7) AdaLN warmup tick (only during training)
    if stage == "train" and hasattr(self.model.predictor, "step_warmup"):
        self.model.predictor.step_warmup()

    # 8) Diagnostic: AdaLN weight norm (action conditioning strength indicator)
    #    At init weights = 0 (AdaLN-zero). Should grow during early training.
    #    If still ~0 after epoch 1, predictor is action-blind.
    diag_log = {}
    if stage == "train" and self.global_step % 50 == 0:
        with torch.no_grad():
            adaln_ws = []
            for block in self.model.predictor.blocks:
                for m in block.adaln.modules_per_view:
                    adaln_ws.append(m[-1].weight.detach().norm())
            if adaln_ws:
                diag_log["diag/adaln_w_norm_mean"] = torch.stack(adaln_ws).mean()
                diag_log["diag/adaln_gate_scale"] = torch.tensor(
                    float(self.model.predictor._gate_scale()), device=output["loss"].device
                )

    # Log
    self.log_dict(
        {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k},
        on_step=True, sync_dist=True,
    )
    self.log_dict(per_view_loss_log, on_step=True, sync_dist=True)
    if aux_grasp_log:
        self.log_dict(aux_grasp_log, on_step=True, sync_dist=True)
    if diag_log:
        self.log_dict(diag_log, on_step=True, sync_dist=True)

    return output


# ---------------------------------------------------------------------------
# Custom spt.Module subclass with per-module LR groups
# ---------------------------------------------------------------------------

class PerViewSPTModule(spt.Module):
    """spt.Module + custom configure_optimizers with per-module LR multipliers."""

    def __init__(self, *, lr_groups, base_lr, weight_decay, max_epochs, cosine_tmax=None,
                 **kwargs):
        kwargs.pop("optim", None)
        super().__init__(**kwargs)
        self._lr_groups = dict(lr_groups)
        self._base_lr = float(base_lr)
        self._weight_decay = float(weight_decay)
        self._max_epochs = int(max_epochs)
        # spt.Module runs manual optimization and steps every scheduler once per
        # optimizer step (stable_pretraining/module.py training_step), so the
        # 'interval: epoch' below is ignored and T_max counts STEPS: with the
        # default T_max = max_epochs the LR cycles between 0 and base_lr with a
        # period of 2*max_epochs steps (14 steps for the 7-epoch agg_cube runs).
        # cosine_tmax (+lr_cosine_tmax) pins T_max so runs with a different
        # max_epochs keep the same LR dynamics. Unset = original behavior.
        self._cosine_tmax = int(cosine_tmax) if cosine_tmax else self._max_epochs

    def configure_optimizers(self):
        seen = set()
        param_groups = []
        for name, mult in self._lr_groups.items():
            module = getattr(self.model, name, None)
            if module is None:
                continue
            params = [p for p in module.parameters() if p.requires_grad and id(p) not in seen]
            if not params:
                continue
            param_groups.append({
                "params": params,
                "lr": self._base_lr * float(mult),
                "name": name,
            })
            for p in params:
                seen.add(id(p))

        # Catch any params not covered by the named groups (e.g. SIGReg buffers; unlikely)
        leftover = [p for p in self.parameters() if p.requires_grad and id(p) not in seen]
        if leftover:
            param_groups.append({"params": leftover, "lr": self._base_lr, "name": "leftover"})

        # Print a summary of param groups for sanity
        for g in param_groups:
            n_params = sum(p.numel() for p in g["params"])
            print(f"[optim] group '{g['name']}': lr={g['lr']:.2e}  "
                  f"n_tensors={len(g['params'])}  n_params={n_params:,}")

        opt = torch.optim.AdamW(param_groups, weight_decay=self._weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self._cosine_tmax)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

    def on_before_optimizer_step(self, optimizer):
        """Diagnostic: log grad norms per module group every 50 steps.

        Key indicator: if `diag/action_encoder_grad_norm` stays at 0 past step
        ~200, predictor has failed to develop action conditioning (AdaLN-zero
        never broke symmetry) — model will be action-blind at MPC time.
        """
        if self.global_step % 50 != 0:
            return
        targets = ("encoder", "action_encoder", "predictor", "projector",
                   "proprio_encoder", "grasp_aux_head")
        with torch.no_grad():
            for name in targets:
                module = getattr(self.model, name, None)
                if module is None:
                    continue
                grads = [p.grad.detach() for p in module.parameters() if p.grad is not None]
                if not grads:
                    continue
                norm = torch.cat([g.flatten() for g in grads]).norm()
                self.log(f"diag/grad_norm_{name}", norm, on_step=True, sync_dist=True)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def train_perview(cfg):
    """Shared training entry point. Called by both cube and scene wrappers."""
    ##############################
    ##         dataset          ##
    ##############################
    ds_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    names = ds_cfg.pop('names', None)
    if names is None:
        primary = swm.data.HDF5Dataset(**ds_cfg, transform=None)
        sources = [primary]
    else:
        ds_cfg.pop('name', None)
        sources = [swm.data.HDF5Dataset(name=n, transform=None, **ds_cfg) for n in names]
        primary = sources[0]

    # ---- Build transforms ----
    grasp_label_xform = None
    if cfg.aux_heads.grasp.enabled:
        lcfg = cfg.aux_heads.grasp.label
        eff_key, cube_key = lcfg.proximity_pair
        grasp_label_xform = GraspLabelTransform(
            contact_key=lcfg.contact_key,
            contact_threshold=lcfg.contact_threshold,
            eff_key=eff_key,
            cube_key=cube_key,
            prox_threshold=lcfg.proximity_threshold,
        )

    transforms = []
    # Compute grasp label FIRST (before normalizers mess with raw values)
    if grasp_label_xform is not None:
        transforms.append(grasp_label_xform)

    # Image preprocessors (always agent + wrist if loaded)
    transforms.append(get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size))
    if cfg.tokens.wrist.enabled or 'wrist_pixels' in cfg.data.dataset.keys_to_load:
        transforms.append(get_img_preprocessor(source='wrist_pixels', target='wrist_pixels', img_size=cfg.img_size))

    # Normalizers for non-image columns
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col == "pixels" or col == "wrist_pixels":
                continue
            normalizer = get_column_normalizer(primary, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", primary.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    for src in sources:
        src.transform = transform

    dataset = sources[0] if len(sources) == 1 else torch.utils.data.ConcatDataset(sources)
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    # Optional fixed held-out validation file (+data.val_dataset_name=<name>), as in
    # train.py: train on every window of the training file, validate on a separate
    # episode-disjoint file with the TRAINING normalizers. Unset = original split.
    val_name = cfg.data.get("val_dataset_name", None)
    if val_name:
        assert names is None, "val_dataset_name supports a single training source"
        _vkw = dict(ds_cfg)
        _vkw["name"] = val_name
        val_set = swm.data.HDF5Dataset(**_vkw, transform=None)
        val_set.transform = transform
        train_set = dataset
        print(f"[val] external held-out val '{val_name}': {len(val_set)} windows; "
              f"training on all {len(train_set)} windows of '{ds_cfg.get('name')}'")
    else:
        train_set, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
        )

    # h5py fork-safety (required when resuming — see feedback_h5py_fork_deadlock memory)
    def _h5_safe_worker_init(_worker_id):
        import torch.utils.data as _td
        info = _td.get_worker_info()
        if info is None:
            return
        roots = getattr(info.dataset, "datasets", [info.dataset])
        seen = set()
        stack = list(roots)
        while stack:
            obj = stack.pop()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if hasattr(obj, "h5_file"):
                obj.h5_file = None
            for attr in ("dataset", "datasets"):
                child = getattr(obj, attr, None)
                if child is None:
                    continue
                stack.extend(child if isinstance(child, (list, tuple)) else [child])

    # Optional fixed-length epochs (+steps_per_epoch=N), as in train.py: N batches
    # drawn with replacement per epoch, so validation and the per-epoch object dump
    # run every N optimizer steps whatever the dataset size. Unset = original.
    spe = cfg.get("steps_per_epoch", None)
    if spe:
        _sampler = torch.utils.data.RandomSampler(
            train_set, replacement=True,
            num_samples=int(spe) * int(cfg.loader.batch_size), generator=rnd_gen)
        train = torch.utils.data.DataLoader(
            train_set, **cfg.loader, sampler=_sampler, drop_last=True,
            worker_init_fn=_h5_safe_worker_init,
        )
        print(f"[steps_per_epoch] {int(spe)} steps/epoch, sampled with replacement "
              f"from {len(train_set)} training windows")
    else:
        train = torch.utils.data.DataLoader(
            train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen,
            worker_init_fn=_h5_safe_worker_init,
        )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False,
        worker_init_fn=_h5_safe_worker_init,
    )

    ##############################
    ##       model / optim      ##
    ##############################
    try:
        encoder = spt.backbone.utils.vit_hf(
            cfg.encoder_scale,
            patch_size=cfg.patch_size,
            image_size=cfg.img_size,
            pretrained=False,
            use_mask_token=False,
        )
    except ImportError:
        # spt.vit_hf needs `transformers.TimmWrapperModel`, absent in this env's
        # 4.46.3. train.py:238-257 already carries this fallback; the per-view
        # trainer predates it, which is why the straight port failed (job
        # 11583048). Arch parity with the v1 ckpt is verified by state_dict
        # key/shape comparison — and here additionally by the [warmstart] key
        # count, since we load the champion's encoder into this module.
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

    # Enabled views in canonical order
    enabled_views = tuple(v for v in VIEW_NAMES if cfg.tokens[v].enabled)
    n_views = len(enabled_views)
    print(f"[model] enabled views: {enabled_views} (n_views={n_views})")

    # Per-view predictor — input/output dim = embed_dim (192)
    predictor = PerViewARPredictor(
        history_size=cfg.wm.history_size,
        n_views=n_views,
        depth=cfg.predictor.depth,
        heads=cfg.predictor.heads,
        mlp_dim=cfg.predictor.mlp_dim,
        input_dim=embed_dim,
        output_dim=hidden_dim,                  # match pred_proj input
        dim_head=cfg.predictor.dim_head,
        dropout=cfg.predictor.dropout,
        emb_dropout=cfg.predictor.emb_dropout,
        view_pos_init_scale=cfg.predictor.view_pos_init_scale,
        attention_mask=cfg.predictor.attention_mask,
        token_order=cfg.predictor.token_order,
        adaln_warmup_steps=cfg.predictor.adaln_warmup_steps,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)

    # Proprio encoder (only if enabled)
    proprio_encoder = None
    if "proprio" in enabled_views:
        proprio_input_dim = sum(primary.get_dim(k) for k in cfg.tokens.proprio.source_keys)
        print(f"[model] proprio input dim: {proprio_input_dim} "
              f"(from {list(cfg.tokens.proprio.source_keys)})")
        proprio_encoder = ProprioEncoder(input_dim=proprio_input_dim, output_dim=embed_dim)

    # Grasp aux head (only if enabled and wrist is enabled)
    grasp_aux_head = None
    if cfg.aux_heads.grasp.enabled and "wrist" in enabled_views:
        grasp_aux_head = GraspAuxHead(
            input_dim=hidden_dim,                        # cls_emb is hidden_dim
            hidden_dim=cfg.aux_heads.grasp.head_hidden_dim,
        )

    # View config dict for model
    view_config = {v: dict(cfg.tokens[v]) for v in VIEW_NAMES}

    # Grasp label config (used by model to sanity-check labels)
    grasp_label_cfg = OmegaConf.to_container(cfg.aux_heads.grasp.label, resolve=True)

    world_model = PerViewJEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
        proprio_encoder=proprio_encoder,
        grasp_aux_head=grasp_aux_head,
        view_config=view_config,
        proprio_keys=tuple(cfg.tokens.proprio.source_keys) if "proprio" in enabled_views else (),
        grasp_label_cfg=grasp_label_cfg,
    )

    # ---- Warm-start ----
    ws = cfg.warmstart
    load_warmstart(world_model, ws.ckpt_path, OmegaConf.to_container(ws.modules, resolve=True))

    # ---- Optional encoder freeze (ported from train.py:299-312) ----
    # `lr_groups.encoder=0.0` is NOT equivalent: the projector's BatchNorm1d
    # keeps updating its running stats in train mode, so the 192-d latent space
    # drifts even though no weight moves. Experiment 6 compares WMs whose cost
    # constants were tuned in one space, so the space must be bit-preserved.
    freeze_encoder = bool(cfg.get("freeze_encoder", False))
    if freeze_encoder:
        assert ws.ckpt_path, "freeze_encoder without warmstart.ckpt_path would freeze random weights"
        for _m in (world_model.encoder, world_model.projector):
            for _p in _m.parameters():
                _p.requires_grad_(False)
            _m.eval()
        _n_tr = sum(p.numel() for p in world_model.parameters() if p.requires_grad)
        _n_all = sum(p.numel() for p in world_model.parameters())
        print(f"[freeze] encoder+projector frozen (eval re-applied per epoch by "
              f"FreezeEvalCallback); trainable {_n_tr/1e6:.2f}M / {_n_all/1e6:.2f}M params")

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = PerViewSPTModule(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        lr_groups=OmegaConf.to_container(cfg.lr_groups, resolve=True),
        base_lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        max_epochs=cfg.trainer.max_epochs,
        cosine_tmax=cfg.get("lr_cosine_tmax", None),
    )

    ##############################
    ##         training         ##
    ##############################
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


@hydra.main(version_base=None, config_path="./config/train", config_name="perview_lewm_cube")
def run(cfg):
    train_perview(cfg)


if __name__ == "__main__":
    run()
