import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

from module import VisualizationDecoder

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer


def denormalize_images(images: torch.Tensor):
    """Invert the ImageNet normalization used by the image preprocessing pipeline."""
    mean = torch.tensor(dt.dataset_stats.ImageNet["mean"], device=images.device).view(
        1, -1, 1, 1
    )
    std = torch.tensor(dt.dataset_stats.ImageNet["std"], device=images.device).view(
        1, -1, 1, 1
    )
    return images * std + mean


def attach_decoder_from_checkpoint(model, checkpoint_path, map_location="cpu"):
    """Attach a saved decoder (VisualizationDecoder or VQVAEDecoder) to a world model."""
    from module import VQVAEDecoder
    import inspect

    DECODER_REGISTRY = {
        "visualization": VisualizationDecoder,
        "vqvae": VQVAEDecoder,
    }

    payload = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
    if not isinstance(payload, dict) or "state_dict" not in payload or "config" not in payload:
        raise ValueError("Decoder checkpoint must contain 'state_dict' and 'config'.")

    config = dict(payload["config"])
    decoder_type = config.pop("type", "visualization")
    decoder_cls = DECODER_REGISTRY.get(decoder_type, VisualizationDecoder)

    valid_params = set(inspect.signature(decoder_cls.__init__).parameters.keys()) - {"self"}
    config = {k: v for k, v in config.items() if k in valid_params}

    decoder = decoder_cls(**config)
    decoder.load_state_dict(payload["state_dict"])
    model.decoder = decoder
    model.decoder_input_key = payload.get("input_key", "emb")
    return model

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")
