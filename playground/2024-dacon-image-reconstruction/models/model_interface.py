from deep_studio.model_layer.model_interface import BaseModelInterface
from deep_studio.common_layer.types import Loss, Image, Metrics


class ModelInterface(BaseModelInterface):
    def __init__(self, **kwargs):
        pass

    def forward_train(self, image: Image, target_image: Image) -> (Loss, Metrics):
        pass

    def forward_test(self, image: Image) -> (Image, Metrics):
        pass
