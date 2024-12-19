from abc import ABC, abstractmethod
from typing import Union, Optional
from pathlib import Path

from tqdm import tqdm

import torch

__all__ = ["BaseRunner", "TrainRunner", "ValidationRunner", "TestRunner"]


# BaseRunner 추상 클래스
class BaseRunner(ABC):

    def __init__(
        self,
        model,
        dataloader,
        device,
        save_temp_output: Optional[Union[str, Path]] = None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.save_temp_output = save_temp_output

    @abstractmethod
    def run(self):
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
    ):
        super().__init__(model, dataloader, device, save_temp_output)
        self.optimizer = optimizer

    def run(self):
        self.model.train()
        total_loss = 0
        train_pbar = tqdm(self.dataloader, desc="TRAIN", leave=False)

        for input_image, gt_image in train_pbar:

            input_image, gt_image = input_image.to(self.device), gt_image.to(
                self.device
            )

            losses, _ = self.model.forward_train(input_image, gt_image)
            loss = sum(losses.values())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update({"total_loss": loss})
            total_loss += loss.item()
            train_pbar.set_postfix({k: v.item() for k, v in losses.items()})

        return total_loss / len(self.dataloader)


# ValidationRunner
class ValidationRunner(BaseRunner):
    def run(self):
        self.model.eval()
        total_loss = 0
        val_pbar = tqdm(self.dataloader, desc="VALIDATION", leave=False)

        with torch.no_grad():
            for input_image, gt_image in val_pbar:
                input_image, gt_image = input_image.to(self.device), gt_image.to(
                    self.device
                )

                losses, _ = self.model.forward_train(input_image, gt_image)
                loss = sum(losses.values())
                total_loss += loss.item()
                val_pbar.set_postfix({k: v.item() for k, v in losses.items()})

        return total_loss / len(self.dataloader)


# TestRunner
class TestRunner(BaseRunner):
    def run(self):
        self.model.eval()
        total_loss = 0
        test_pbar = tqdm(self.dataloader, desc="TEST", leave=True)

        results = []
        with torch.no_grad():
            for input_image, _ in test_pbar:
                input_image = input_image.to(self.device)

                pred_image, _ = self.model.forward_test(input_image)

                results.append(pred_image.detach().cpu())

        return torch.cat(results, dim=0)
