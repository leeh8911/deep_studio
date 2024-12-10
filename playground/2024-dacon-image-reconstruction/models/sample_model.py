import torch
from torch import nn

from deep_studio.model_layer.model_registry import MODEL_REGISTRY


@MODEL_REGISTRY.register
class CustomModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image
