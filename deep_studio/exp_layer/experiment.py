from argparse import ArgumentParser
import random
from typing import Union
import os
from datetime import datetime
from pathlib import Path
import logging

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 추가

from deep_studio.common_layer.config import Config
from deep_studio.data_layer.data_registry import DATALOADER_FACTORY_REGISTRY
from deep_studio.model_layer.model_registry import (
    MODEL_INTERFACE_REGISTRY,
    OPTIMIZER_REGISTRY,
)
from deep_studio.exp_layer.runner import TrainRunner, ValidationRunner, TestRunner

__all__ = ["Experiment"]


class Experiment:
    """
    Experiment 클래스: 모델 학습, 검증 및 테스트를 위한 실험 환경을 설정 및 실행합니다.
    
    주요 기능:
        - 설정 파일(config) 로드
        - 실험 폴더 및 체크포인트 경로 생성
        - 모델, 옵티마이저, 데이터로더 초기화
        - TensorBoard 로그 기록
        - 학습 루프와 체크포인트 관리
    """

    def __init__(self):
        """
        Experiment 클래스의 생성자.
        - 명령줄 인자를 파싱하고 설정(config)을 로드합니다.
        - 실험 환경 설정 (시드, 디바이스, 경로 등)을 초기화합니다.
        - TensorBoard SummaryWriter를 설정합니다.
        """
        self.logger = logging.getLogger("ExperimentLogger")
        self.exp_time = datetime.now()

        # 명령줄 인자 처리 및 설정 파일 로드
        args = self._parse_arguments()
        self._initialize_config(args)
        self.seed = self.config.get("cfg", {}).get("seed", 0)
        self.device = self._setup_device(self.config["cfg"].get("device", "cpu"))
        self._set_seed(self.seed)

        # 경로 설정 및 TensorBoard 설정
        self._initialize_paths()
        self.writer = SummaryWriter(log_dir=str(self.exp_base_dir), comment=self.exp_name)

        # 모델 및 옵티마이저 초기화
        self._initialize_components()

        # 체크포인트 설정
        self.checkpoint_period = self.config["cfg"].get("checkpoint_period", 1)
        self.max_epoch = self.config["cfg"]["max_epoch"]
        self.current_epoch = 0

        if args.checkpoint:
            self.load_checkpoint(args.checkpoint)

    def _parse_arguments(self):
        """
        명령줄 인자를 파싱합니다.
        
        Returns:
            argparse.Namespace: 파싱된 인자 객체
        """
        parser = ArgumentParser(description="실험 및 검증을 위한 실험 프로젝트")
        parser.add_argument("--config", type=str, help="Configuration file location")
        parser.add_argument("--checkpoint", default=None, type=str, help="Checkpoint file path")
        return parser.parse_args()

    def _setup_device(self, device: str):
        """
        디바이스를 설정합니다.
        
        Args:
            device (str): 설정 파일에서 지정된 디바이스 이름
        
        Returns:
            str: 사용할 디바이스 ('cpu', 'cuda', 'mps')
        """
        if device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.logger.info("Set device: %s", device)
        return device

    def _set_seed(self, seed: int):
        """
        시드를 설정합니다.
        
        Args:
            seed (int): 설정 파일에서 지정된 시드 값
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _initialize_config(self, args):
        """
        설정 파일을 로드하고 작업 경로를 설정합니다.
        
        Args:
            args (argparse.Namespace): 파싱된 명령줄 인자
        """
        self.checkpoint = args.checkpoint
        config_path = args.config
        self.config_name = Path(config_path).stem
        self.workspace = Path(config_path).parent.parent
        self.config = Config.from_file(config_path)

    def _initialize_paths(self):
        """
        실험 결과를 저장할 경로를 초기화합니다.
        """
        self.exp_name = self.exp_time.strftime("%y%m%d_%H%M%S") + "_" + self.config_name
        self.exp_base_dir = self.workspace / "logs"
        self.exp_base_dir.mkdir(exist_ok=True)

        self.exp_dir = self.exp_base_dir / self.exp_name
        self.exp_dir.mkdir(exist_ok=True)

        self.checkpoint_dir = self.exp_dir / "checkpoint"
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _initialize_components(self):
        """
        모델, 옵티마이저 및 데이터로더를 초기화합니다.
        """
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

    def train(self):
        """
        모델 학습 및 검증 루프를 실행합니다.
        체크포인트를 주기적으로 저장하고 TensorBoard 로그를 기록합니다.
        """
        best_val_loss = float("inf")
        self.train_runner = TrainRunner(
            self.model, self.dataloader_factory.get("train"), self.optimizer, self.device, writer=self.writer
        )
        self.validation_runner = ValidationRunner(
            self.model, self.dataloader_factory.get("validation"), self.device, writer=self.writer
        )

        epoch_pbar = tqdm(range(self.current_epoch, self.max_epoch), desc="EPOCH", leave=True)
        for epoch in epoch_pbar:
            train_loss = self.train_runner.run(epoch)
            val_loss = self.validation_runner.run(epoch)

            self.current_epoch = epoch
            if epoch % self.checkpoint_period == 0:
                self.save_checkpoint(self.checkpoint_dir / f"{epoch}-CHECKPOINT.pth")

            if val_loss["total_loss"] < best_val_loss:
                best_val_loss = val_loss["total_loss"]
                self.save_checkpoint(self.exp_dir / "BEST-CHECKPOINT.pth")

        self.writer.close()

    def save_checkpoint(self, path: Union[str, Path]):
        """
        모델의 상태와 학습 정보를 체크포인트에 저장합니다.
        
        Args:
            path (Union[str, Path]): 체크포인트 파일의 경로
        """
        self.logger.info("Saving checkpoint at %s", path)
        torch.save({
            "epoch": self.current_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: Union[str, Path]):
        """
        체크포인트 파일에서 모델과 학습 상태를 불러옵니다.
        
        Args:
            path (Union[str, Path]): 체크포인트 파일의 경로
        """
        self.logger.info("Loading checkpoint from %s", path)
        checkpoint = torch.load(path)
        self.current_epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            self.scheduler.load_state_dict(checkpoint["scheduler"])
