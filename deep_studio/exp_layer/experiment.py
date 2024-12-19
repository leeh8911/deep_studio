from argparse import ArgumentParser
import random
from typing import Tuple, Union
import os
from datetime import datetime
from pathlib import Path
import logging
import sys


import numpy as np
import torch
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
from sklearn.model_selection import train_test_split

from deep_studio.common_layer.config import Config
from deep_studio.data_layer.data_registry import DATALOADER_FACTORY_REGISTRY
from deep_studio.model_layer.model_registry import (
    MODEL_INTERFACE_REGISTRY,
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
)
from deep_studio.exp_layer.runner import TrainRunner, ValidationRunner, TestRunner

__all__ = ["Experiment"]


class Experiment:
    """학습 및 검증을 위한 실험 클래스"""

    def __init__(self):
        self.logger = logging.getLogger("ExperimentLogger")
        self.exp_time = datetime.now()

        # 설정 초기화
        self.__initialize_config()

        # 시드 설정
        self.seed = self.config["cfg"].get("seed", 0)
        self.__set_seed(self.seed)

        # 디바이스 설정
        self.device = self.__device_check(self.config["cfg"]["device"])
        self.logger.info("Set device: %s", self.device)

        # 경로 설정
        self.__initialize_paths()

        # 모델, 옵티마이저, 스케줄러 설정
        self.__initialize_components()

        # 체크포인트 로드
        if self.checkpoint:
            self.load_checkpoint(self.checkpoint)

        self.train_runner = None
        self.validation_runner = None
        self.test_runner = None

    def __initialize_config(self):
        """Argument Parser 및 Config 초기화"""
        parser = ArgumentParser(description="실험 및 검증을 위한 실험 프로젝트")
        parser.add_argument(
            "--config", type=str, help="configuration file location for training"
        )
        parser.add_argument(
            "--checkpoint", default=None, type=str, help="checkpoint file path"
        )

        args = parser.parse_args()
        self.checkpoint = args.checkpoint

        config_path = args.config
        self.config_name = Path(config_path).stem
        self.workspace = Path(config_path).parent.parent
        self.config = Config.from_file(config_path)

    def __initialize_paths(self):
        """Experiment 디렉토리 및 경로 초기화"""
        self.exp_base_dir = self.workspace / "experiment"
        self.exp_base_dir.mkdir(exist_ok=True)

        self.exp_dir = self.exp_base_dir / (
            self.exp_time.strftime("%y%m%d_%H%M%S") + "_" + self.config_name
        )
        self.exp_dir.mkdir()

        self.checkpoint_dir = self.exp_dir / "checkpoint"
        self.output_dir = self.exp_dir / "output"

        self.checkpoint_period = self.config["cfg"]["checkpoint_period"]
        self.max_epoch = self.config["cfg"]["max_epoch"]
        self.current_epoch = 0

    def __initialize_components(self):
        """모델, 옵티마이저, 스케줄러 초기화"""
        self.model = MODEL_INTERFACE_REGISTRY.build(
            **self.config["cfg"]["model_interface"]
        )
        self.model.to(self.device)

        self.optimizer = OPTIMIZER_REGISTRY.build(
            **self.config["cfg"]["optimizer"], params=self.model.parameters()
        )

        self.scheduler = None
        if "scheduler" in self.config["cfg"]:
            raise NotImplementedError("Scheduler setting is not implemented")

        self.dataloader_factory = DATALOADER_FACTORY_REGISTRY.build(
            **self.config["cfg"]["dataloader"]
        )

    def __device_check(self, device):
        """디바이스 설정 확인"""
        return device if device == "cuda" and torch.cuda.is_available() else "cpu"

    def __set_seed(self, seed):
        """시드 설정"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self):
        best_val_loss = float("inf")

        self.train_runner = TrainRunner(
            self.model,
            self.dataloader_factory.get("train"),
            self.optimizer,
            self.device,
        )
        self.validation_runner = ValidationRunner(
            self.model, self.dataloader_factory.get("validation"), self.device
        )

        # TODO(sangwon): 추후 리팩터링 필요
        self.test_runner = TestRunner(
            self.model, self.dataloader_factory.get("validation"), self.device
        )

        epoch_pbar = tqdm(range(self.max_epoch), desc="EPOCH", leave=True)
        epoch_pbar.update(self.current_epoch)
        for epoch in epoch_pbar:
            train_loss = self.train_runner.run()
            print(f"Epoch {epoch} Train Loss: {train_loss:.4f}")

            val_loss = self.validation_runner.run()
            print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}")

            if True:  # TODO(sangwon): 추후 리팩터링 필요
                test_results = self.test_runner.run()
                temp_test_results_save_dir = Path(self.exp_dir, "EPOCH_" + str(epoch))
                os.mkdir(temp_test_results_save_dir)
                for idx, image in enumerate(test_results):
                    image = to_pil_image(image)

                    image.save(
                        f"{temp_test_results_save_dir}/TEST_{str(idx).zfill(3)}.png"
                    )

            self.current_epoch = epoch
            if epoch % self.checkpoint_period == 0:
                self.save_checkpoint(f"{epoch}-CHECKPOINT.pth")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint("BEST-CHECKPOINT.pth")

    def test(self, split: str = "test"):
        self.test_runner = TestRunner(
            self.model,
            self.dataloader_factory.get(split),
            self.device,
        )

        return self.test_runner.run()

    def save_checkpoint(self, path: Union[str, Path]):
        self.logger.info("Save checkpoint  %s", path)
        torch.save(
            {
                "epoch": self.current_epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: Union[str, Path]):
        self.logger.info("Load checkpoint %s", path)
        checkpoint = torch.load(path)  # 저장된 체크포인트 불러오기

        # 저장된 상태 복원
        self.current_epoch = checkpoint["epoch"]  # 에폭 정보
        self.model.load_state_dict(checkpoint["model"])  # 모델 가중치 로드
        self.optimizer.load_state_dict(checkpoint["optimizer"])  # 옵티마이저 상태 로드

        if checkpoint["scheduler"]:
            self.scheduler.load_state_dict(
                checkpoint["scheduler"]
            )  # 스케줄러 상태 로드
        self.config = checkpoint["config"]  # 설정 정보 로드

    def visualization(self):
        pass
