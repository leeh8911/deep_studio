from deep_studio.model_layer.model_interface import BaseModelInterface
from deep_studio.common_layer.types import Loss, Image, Metrics
from deep_studio.model_layer.model_registry import (
    MODEL_REGISTRY,
    MODEL_INTERFACE_REGISTRY,
    LOSS_REGISTRY,
)

import torch


@MODEL_INTERFACE_REGISTRY.register
class Dacon2024ImageReconstructionModelInterface(BaseModelInterface):
    def __init__(self, model, criterion, **kwargs):
        super().__init__()

        self.model = MODEL_REGISTRY.build(**model)
        self.criterion = LOSS_REGISTRY.build(**criterion)

    def forward(self, image: Image) -> Image:
        return self.model(image)

    def forward_train(self, image: Image, target_image: Image) -> (Loss, Metrics):
        pred_image = self(image)
        loss = self.criterion(pred_image, target_image)
        return loss, None

    def forward_test(self, image: Image) -> (Image, Metrics):
        pred_image = self(image)
        return pred_image, None
