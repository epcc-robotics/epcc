"""Faithful port of the UNISafe ensemble uncertainty module (CMU-IntentLab/UNISafe,
arXiv 2505.00779). Math copied verbatim from dreamerv3_torch/ensemble/{ensemble_linear,
penn,utility}.py; only the training/optimizer glue is replaced (torch.optim.Adam instead
of dreamer's tools.Optimizer) in predictor.py."""
from .ensemble_linear import EnsembleLinear
from .penn import EnsembleStochasticLinear
from .utility import JensenRenyiDivergence
from .predictor import EnsemblePredictor

__all__ = ["EnsembleLinear", "EnsembleStochasticLinear", "JensenRenyiDivergence", "EnsemblePredictor"]
