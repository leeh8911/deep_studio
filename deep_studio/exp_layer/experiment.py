from argparse import ArgumentParser


from deep_studio.common_layer.config import Config

from deep_studio.data_layer.data_registry import DATALOADER_REGISTRY

from deep_studio.model_layer.model_registry import (
    MODEL_INTERFACE_REGISTRY,
    MODEL_REGISTRY,
)


class Experiment:
    """학습 및 검증을 위한 실험 클래스"""

    def __init__(self):

        parser = ArgumentParser(description="실험 및 검증을 위한 실험 프로젝트")
        parser.add_argument(
            "--config", type=str, help="configuration file location for training"
        )

        args = parser.parse_args()
        config_path = args.config

        self.config = Config.from_file(config_path)

        self.model = MODEL_INTERFACE_REGISTRY.build(
            **self.config["cfg"]["model_interface"]
        )

        self.dataloader = DATALOADER_REGISTRY.build(**self.config["cfg"]["dataloader"])

    def run(self):
        pass

    def visualization(self):
        pass
