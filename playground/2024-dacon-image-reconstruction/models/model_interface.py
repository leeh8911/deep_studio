from deep_studio.model_layer.model_interface import BaseModelInterface
from deep_studio.common_layer.types import Loss, Image, Metrics
from deep_studio.model_layer.model_registry import (
    MODEL_REGISTRY,
    MODEL_INTERFACE_REGISTRY,
    LOSS_REGISTRY,
)


@MODEL_INTERFACE_REGISTRY.build
class ModelInterface(BaseModelInterface):
    def __init__(self, **kwargs):
        self.model = MODEL_REGISTRY.build(**kwargs.get("model"))
        self.criterion = LOSS_REGISTRY.build(**kwargs.get("loss"))

    def forward_train(self, image: Image, target_image: Image) -> (Loss, Metrics):
        pred_image = self.model(image)
        loss = self.criterion(pred_image, target_image)
        return loss, None

    def forward_test(self, image: Image) -> (Image, Metrics):
        pred_image = self.model(image)
        return pred_image, None
