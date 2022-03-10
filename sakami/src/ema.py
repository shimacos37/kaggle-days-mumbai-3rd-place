from copy import deepcopy

import torch
from torch import nn


class ModelEma(object):
    """Model for Exponential Moving Average.

    Parameters
    ----------
    model : nn.Module
        Model for training.
    decay : float, optional
        Rate of previous weight, by default 0.9
    n : int, optional
        Interval steps between weight update, by default 1
    """

    def __init__(self, model: nn.Module, decay: float = 0.9, n: int = 1):
        # make a copy of the model for accumulating moving average of weights
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        self.n = n
        self.count = self.n

        self.ema_model
        self.ema_has_module = hasattr(self.ema_model, "module")
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def _update(self, model):
        # correct a mismatch in state dict keys
        needs_module = hasattr(model, "module") and not self.ema_has_module
        with torch.no_grad():
            state_dict = model.state_dict()
            for k, ema_v in self.ema_model.state_dict().items():
                if needs_module:
                    k = "module." + k
                model_v = state_dict[k].detach()
                ema_v.copy_(ema_v * self.decay + model_v * (1.0 - self.decay))

    def update(self, model):
        self.count -= 1
        if self.count == 0:
            self._update(model)
            self.count = self.n
