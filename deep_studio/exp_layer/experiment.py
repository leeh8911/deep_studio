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
        args = self._parse_arguments()
        
        # 시드 및 디바이스 설정
        self.config = Config.from_file(args.config)
        self.seed = self.config.get("cfg", {}).get("seed", 0)
        self.device = self._setup_device(self.config["cfg"].get("device", "cpu"))
        self._set_seed(self.seed)

        # 경로 설정 및 폴더 생성
        self.workspace, self.exp_dir, self.checkpoint_dir = self._setup_experiment_directory(args.config)
        self.checkpoint_period = self.config["cfg"].get("checkpoint_period", 1)
        self.max_epoch = self.config["cfg"]["max_epoch"]
        self.current_epoch = 0

        # 모델, 옵티마이저, 데이터로더 설정
        self.model, self.optimizer, self.scheduler = self._setup_model_and_optimizer()
        self.dataloader_factory = DATALOADER_FACTORY_REGISTRY.build(
            **self.config["cfg"]["dataloader"]
        )
        self.train_runner, self.validation_runner, self.test_runner = None, None, None

        # 체크포인트 로드
        if args.checkpoint:
            self.load_checkpoint(args.checkpoint)

    def _parse_arguments(self):
        parser = ArgumentParser(description="실험 및 검증을 위한 실험 프로젝트")
        parser.add_argument("--config", type=str, help="configuration file location")
        parser.add_argument("--checkpoint", default=None, type=str, help="checkpoint path")
        return parser.parse_args()

    def _setup_device(self, device: str):
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _setup_experiment_directory(self, config_path: str):
        workspace = Path(config_path).resolve().parent.parent
        exp_base_dir = workspace / "experiment"
        exp_base_dir.mkdir(parents=True, exist_ok=True)

        exp_dir = exp_base_dir / (self.exp_time.strftime("%y%m%d_%H%M%S") + f"_{Path(config_path).stem}")
        exp_dir.mkdir(exist_ok=True)

        checkpoint_dir = exp_dir / "checkpoint"
        checkpoint_dir.mkdir(exist_ok=True)
        
        return workspace, exp_dir, checkpoint_dir

    def _setup_model_and_optimizer(self):
        model = MODEL_INTERFACE_REGISTRY.build(**self.config["cfg"]["model_interface"])
        model.to(self.device)

        optimizer = OPTIMIZER_REGISTRY.build(**self.config["cfg"]["optimizer"], params=model.parameters())
        scheduler = None
        if "scheduler" in self.config["cfg"]:
            raise NotImplementedError("Scheduler setting is not implemented")

        return model, optimizer, scheduler

    def train(self):
        """학습 및 검증 루프"""
        best_val_loss = float("inf")

        self.train_runner = TrainRunner(
            self.model, self.dataloader_factory.get("train"), self.optimizer, self.device
        )
        self.validation_runner = ValidationRunner(
            self.model, self.dataloader_factory.get("validation"), self.device
        )

        epoch_pbar = tqdm(range(self.current_epoch, self.max_epoch), desc="EPOCH", leave=True)
        for epoch in epoch_pbar:
            train_loss = self.train_runner.run()
            val_loss = self.validation_runner.run()

            print(f"Epoch {epoch} Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            self.current_epoch = epoch

            # 체크포인트 저장
            self._save_checkpoint_if_needed(val_loss, best_val_loss)
            best_val_loss = min(best_val_loss, val_loss)

    def _save_checkpoint_if_needed(self, val_loss, best_val_loss):
        if self.current_epoch % self.checkpoint_period == 0:
            self.save_checkpoint(self.checkpoint_dir / f"{self.current_epoch}-CHECKPOINT.pth")

        if val_loss < best_val_loss:
            self.save_checkpoint(self.exp_dir / "BEST-CHECKPOINT.pth")

    def save_checkpoint(self, path: Union[str, Path]):
        self.logger.info(f"Saving checkpoint at {path}")
        torch.save({
            "epoch": self.current_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: Union[str, Path]):
        self.logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path)
        self.current_epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.config = checkpoint["config"]

    def test(self, split: str = "test"):
        self.test_runner = TestRunner(self.model, self.dataloader_factory.get(split), self.device)
        return self.test_runner.run()