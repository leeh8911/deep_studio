from argparse import ArgumentParser
import random
import os

import numpy as np
import torch


from deep_studio.common_layer.config import Config
from deep_studio.data_layer.data_registry import DATALOADER_REGISTRY
from deep_studio.model_layer.model_registry import (
    MODEL_INTERFACE_REGISTRY,
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
)


class Experiment:
    """학습 및 검증을 위한 실험 클래스"""

    def __init__(self):

        parser = ArgumentParser(description="실험 및 검증을 위한 실험 프로젝트")
        parser.add_argument(
            "--config", type=str, help="configuration file location for training"
        )

        args = parser.parse_args()
        config_path = args.config  # {project_path}/configs/config_file

        self.workspace = os.path.join(*config_path.split("/")[:-2])
        self.config = Config.from_file(config_path)

        self.seed = self.config["cfg"]["seed"] if "seed" in self.config["cfg"] else 0
        self.__set_seed(self.seed)

        self.device = self.__device_check(self.config["cfg"]["device"])

        self.train = self.config["cfg"]["train"]

        self.model = MODEL_INTERFACE_REGISTRY.build(
            **self.config["cfg"]["model_interface"]
        )
        self.optimizer = OPTIMIZER_REGISTRY.build(**self.config["cfg"]["optimizer"])
        if "scheduler" in self.config["cfg"]:
            self.scheduler = self.config["cfg"]["scheduler"]
        else:
            self.scheduler = None

        self.dataloader = DATALOADER_REGISTRY.build(**self.config["cfg"]["dataloader"])

    def __device_check(self, device):
        if device == "cuda" and torch.cuda.is_available():
            return device

        return "cpu"

    def __set_seed(self, seed):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run(self):

        for idx, data in enumerate(self.dataloader):
            input_image, gt_image = data
            loss, _ = self.model.forward_train(input_image, gt_image)

    def save(self):
        pass

    def load(self):
        pass

    def visualization(self):
        pass
