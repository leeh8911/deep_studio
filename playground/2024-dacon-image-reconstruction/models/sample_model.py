from typing import Dict, List

import torch
from torch import nn

from deep_studio.model_layer.model_registry import MODEL_REGISTRY


@MODEL_REGISTRY.register
class CustomModel(nn.Module):
    def __init__(self, embed_layers: List[Dict], **kwargs):
        super().__init__()

        self.embed_layers = nn.ModuleList()
        for embed_layer in embed_layers:
            self.embed_layers.append(MODEL_REGISTRY.build(**embed_layer))

    def forward(self, image: torch.Tensor) -> torch.Tensor:

        for embed_layer in self.embed_layers:
            image = embed_layer(image)
        return image
