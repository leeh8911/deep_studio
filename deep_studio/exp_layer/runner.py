from abc import ABC, abstractmethod
from typing import Union, Optional
from pathlib import Path

from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

__all__ = ["BaseRunner", "TrainRunner", "ValidationRunner", "TestRunner"]


# BaseRunner 추상 클래스
class BaseRunner(ABC):
    def __init__(
        self,
        model,
        dataloader,
        device,
        save_temp_output: Optional[Union[str, Path]] = None,
        writer: Optional[SummaryWriter] = None,
    ):
        """
        BaseRunner 초기화
        Args:
            model: 학습/검증/테스트에 사용할 모델
            dataloader: 데이터로더
            device: 사용할 디바이스 (cuda or cpu)
            save_temp_output: 임시 결과 저장 경로
            writer: TensorBoard SummaryWriter 객체
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.save_temp_output = save_temp_output
        self.writer = writer  # TensorBoard Writer

    @abstractmethod
    def run(self, epoch: int = 0):
        pass


# TrainRunner
class TrainRunner(BaseRunner):
    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        device,
        save_temp_output: Optional[Union[str, Path]] = None,
        writer: Optional[SummaryWriter] = None,
    ):
        """
        TrainRunner 초기화
        Args:
            model: 학습할 모델
            dataloader: 학습 데이터로더
            optimizer: 옵티마이저
            device: 사용할 디바이스
            save_temp_output: 결과 저장 경로
            writer: TensorBoard SummaryWriter 객체
        """
        super().__init__(model, dataloader, device, save_temp_output, writer)
        self.optimizer = optimizer

    def run(self, epoch: int = 0):
        """
        학습 과정 실행 및 TensorBoard 로그 기록
        Args:
            epoch (int): 현재 Epoch 번호
        Returns:
            dict: 평균 Loss 및 Metrics
        """
        self.model.train()
        total_losses = {}
        total_metrics = {}
        total_loss_sum = 0.0

        train_pbar = tqdm(self.dataloader, desc=f"TRAIN EPOCH {epoch}", leave=False)

        for step, (input_image, gt_image) in enumerate(train_pbar):
            input_image, gt_image = (
                input_image.to(self.device),
                gt_image.to(self.device),
            )

            # Loss 및 Metrics 계산
            losses, metrics = self.model.forward_train(input_image, gt_image)
            loss_sum = sum(losses.values())

            # 옵티마이저 업데이트
            self.optimizer.zero_grad()
            loss_sum.backward()
            self.optimizer.step()

            # Loss 및 Metrics 누적
            for key, value in losses.items():
                total_losses[key] = total_losses.get(key, 0.0) + value.item()
            for key, value in (metrics or {}).items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value
            total_loss_sum += loss_sum.item()

            # TensorBoard Step 단위 기록
            if self.writer:
                self.writer.add_scalar(
                    "Train/Step_Total_Loss",
                    loss_sum.item(),
                    epoch * len(self.dataloader) + step,
                )
                for key, value in losses.items():
                    self.writer.add_scalar(
                        f"Train/Step_{key}",
                        value.item(),
                        epoch * len(self.dataloader) + step,
                    )
                for key, value in metrics.items():
                    self.writer.add_scalar(
                        f"Train/Step_{key}", value, epoch * len(self.dataloader) + step
                    )

            # 진행 상태 업데이트
            train_pbar.set_postfix({"Total_Loss": f"{loss_sum.item():.4f}"})

        # 평균 Loss 및 Metrics 계산
        avg_losses = {k: v / len(self.dataloader) for k, v in total_losses.items()}
        avg_metrics = {k: v / len(self.dataloader) for k, v in total_metrics.items()}
        avg_total_loss = total_loss_sum / len(self.dataloader)

        # TensorBoard Epoch 단위 기록
        if self.writer:
            self.writer.add_scalar("Train/Epoch_Total_Loss", avg_total_loss, epoch)
            for key, value in avg_losses.items():
                self.writer.add_scalar(f"Train/Epoch_{key}", value, epoch)
            for key, value in avg_metrics.items():
                self.writer.add_scalar(f"Train/Epoch_{key}", value, epoch)

        return {"total_loss": avg_total_loss, **avg_losses, **avg_metrics}


# ValidationRunner
class ValidationRunner(BaseRunner):
    def run(self, epoch: int = 0):
        """
        Validation 과정 실행 및 TensorBoard 로그 기록
        Args:
            epoch (int): 현재 Epoch 번호
        Returns:
            dict: Validation 평균 Loss 및 Metrics
        """
        self.model.eval()
        total_losses = {}
        total_metrics = {}
        total_loss_sum = 0.0

        val_pbar = tqdm(self.dataloader, desc=f"VALIDATION EPOCH {epoch}", leave=False)

        with torch.no_grad():
            for step, (input_image, gt_image) in enumerate(val_pbar):
                input_image, gt_image = (
                    input_image.to(self.device),
                    gt_image.to(self.device),
                )

                # Loss 및 Metrics 계산
                losses, metrics = self.model.forward_train(input_image, gt_image)
                loss_sum = sum(losses.values())

                # Loss 및 Metrics 누적
                for key, value in losses.items():
                    total_losses[key] = total_losses.get(key, 0.0) + value.item()
                for key, value in (metrics or {}).items():
                    total_metrics[key] = total_metrics.get(key, 0.0) + value
                total_loss_sum += loss_sum.item()

                val_pbar.set_postfix({"Total_Loss": f"{loss_sum.item():.4f}"})

        # 평균 Loss 및 Metrics 계산
        avg_losses = {k: v / len(self.dataloader) for k, v in total_losses.items()}
        avg_metrics = {k: v / len(self.dataloader) for k, v in total_metrics.items()}
        avg_total_loss = total_loss_sum / len(self.dataloader)

        # TensorBoard Epoch 단위 기록
        if self.writer:
            self.writer.add_scalar("Validation/Epoch_Total_Loss", avg_total_loss, epoch)
            for key, value in avg_losses.items():
                self.writer.add_scalar(f"Validation/Epoch_{key}", value, epoch)
            for key, value in avg_metrics.items():
                self.writer.add_scalar(f"Validation/Epoch_{key}", value, epoch)

        return {"total_loss": avg_total_loss, **avg_losses, **avg_metrics}


# TestRunner
class TestRunner(BaseRunner):
    def run(self, epoch: int = 0):
        """
        Test 과정 실행
        Args:
            epoch (int): 현재 Epoch 번호
        Returns:
            torch.Tensor: 예측된 이미지 결과
        """
        self.model.eval()
        test_pbar = tqdm(self.dataloader, desc=f"TEST EPOCH {epoch}", leave=True)

        results = []
        with torch.no_grad():
            for input_image, _ in test_pbar:
                input_image = input_image.to(self.device)

                # 예측 수행
                pred_image = self.model.forward_test(input_image)

                results.append(pred_image.detach().cpu())

        return torch.cat(results, dim=0)
