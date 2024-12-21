from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Grayscale
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from deep_studio.model_layer.model_interface import BaseModelInterface
from deep_studio.common_layer.types import Loss, Image, Metrics
from deep_studio.model_layer.model_registry import (
    MODEL_REGISTRY,
    MODEL_INTERFACE_REGISTRY,
    LOSS_REGISTRY,
)


# Metric Classes
class SSIMMetric(nn.Module):
    def __init__(self):
        super(SSIMMetric, self).__init__()

    def forward(self, true, pred):
        true_np = true.permute(0, 2, 3, 1).cpu().numpy()
        pred_np = pred.permute(0, 2, 3, 1).cpu().numpy()
        batch_size = true.shape[0]
        ssim_values = []
        for i in range(batch_size):
            ssim_value = ssim(
                true_np[i],
                pred_np[i],
                channel_axis=-1,
                data_range=pred_np[i].max() - pred_np[i].min(),
            )
            ssim_values.append(ssim_value)
        return torch.tensor(ssim_values, device=true.device)


class MaskedSSIMMetric(nn.Module):
    def __init__(self):
        super(MaskedSSIMMetric, self).__init__()

    def forward(self, true, pred, mask):
        true_np = true.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
        pred_np = pred.permute(0, 2, 3, 1).cpu().numpy()
        mask_np = mask.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, 1)

        batch_size = true.shape[0]
        masked_ssim_values = []

        for i in range(batch_size):
            # 마스크를 채널 차원으로 반복하여 true 및 pred와 호환되게 함
            mask_3d = mask_np[i].repeat(3, axis=2) > 0  # (H, W, C)

            # 마스크가 적용된 이미지 생성 (마스크 영역만 남김, 나머지는 0으로 설정)
            true_masked = true_np[i] * mask_3d
            pred_masked = pred_np[i] * mask_3d

            # SSIM 계산 (전체 이미지에서 마스킹된 영역만 평가)
            ssim_value = ssim(
                true_masked,
                pred_masked,
                channel_axis=-1,
                data_range=pred_masked.max() - pred_masked.min(),
            )
            masked_ssim_values.append(ssim_value)

        # Torch tensor로 변환하여 반환
        return torch.tensor(masked_ssim_values, device=true.device)


class HistogramSimilarityMetric(nn.Module):
    def __init__(self):
        super(HistogramSimilarityMetric, self).__init__()

    def forward(self, true, pred):
        true_np = true.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pred_np = pred.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        batch_size = true.shape[0]
        hist_similarity_values = []
        for i in range(batch_size):
            true_hsv = cv2.cvtColor(true_np[i], cv2.COLOR_BGR2HSV)
            pred_hsv = cv2.cvtColor(pred_np[i], cv2.COLOR_BGR2HSV)
            hist_true = cv2.calcHist([true_hsv], [0], None, [180], [0, 180])
            hist_pred = cv2.calcHist([pred_hsv], [0], None, [180], [0, 180])
            hist_true = cv2.normalize(hist_true, hist_true).flatten()
            hist_pred = cv2.normalize(hist_pred, hist_pred).flatten()
            similarity = cv2.compareHist(hist_true, hist_pred, cv2.HISTCMP_CORREL)
            hist_similarity_values.append(similarity)
        return torch.tensor(hist_similarity_values, device=true.device)


@MODEL_INTERFACE_REGISTRY.register
class Dacon2024ImageReconstructionModelInterface(BaseModelInterface):
    def __init__(self, model, criterions, **kwargs):
        super().__init__()

        self.model = MODEL_REGISTRY.build(**model)
        self.to_gray_scale = Grayscale()
        self.criterion = {
            key: LOSS_REGISTRY.build(**criterion)
            for key, criterion in criterions.items()
        }

        # Metrics
        self.ssim_metric = SSIMMetric()
        self.masked_ssim_metric = MaskedSSIMMetric()
        self.histogram_metric = HistogramSimilarityMetric()

    def forward(self, image: Image) -> Image:
        return self.model(image)

    def __loss(self, results, image, target_image):
        """Loss 계산"""
        mask_prediction = 1 - results["mask_prediction"]
        if mask_prediction.dim() == 3:
            mask_prediction = mask_prediction.unsqueeze(1)

        none_mask_prediction = 1 - mask_prediction

        color_loss = self.criterion["color_criterion"](results["image"], target_image)
        recon_loss = self.criterion["recon_criterion"](
            self.to_gray_scale(results["image"]), self.to_gray_scale(target_image)
        )

        masked_color_loss = (
            self.criterion["masked_color_criterion"](results["image"], target_image)
            * none_mask_prediction
        )
        masked_recon_loss = (
            self.criterion["masked_recon_criterion"](
                self.to_gray_scale(results["image"]), self.to_gray_scale(target_image)
            )
            * none_mask_prediction
        )

        masked_color_loss = masked_color_loss.mean()
        masked_recon_loss = masked_recon_loss.mean()
        mask_prediction_loss = 0.1 * (image * mask_prediction).mean()

        losses = {
            "color_loss": color_loss,
            "recon_loss": recon_loss,
            "masked_color_loss": masked_color_loss,
            "masked_recon_loss": masked_recon_loss,
            "mask_prediction_loss": mask_prediction_loss,
        }
        return losses

    def __metric(self, reconstructed_image, target_image, mask):
        """Metrics 계산"""
        ssim_value = self.ssim_metric(reconstructed_image, target_image).mean()
        masked_ssim_value = self.masked_ssim_metric(
            reconstructed_image, target_image, mask
        ).mean()
        histogram_similarity_value = self.histogram_metric(
            reconstructed_image, target_image
        ).mean()

        metrics = {
            "ssim": ssim_value.item(),
            "masked_ssim": masked_ssim_value.item(),
            "histogram_similarity": histogram_similarity_value.item(),
        }
        return metrics

    def forward_train(self, image: Image, target_image: Image) -> (Any, Loss, Metrics):
        results = self(image)

        # Loss 계산
        losses = self.__loss(results, image, target_image)

        # Metric 계산
        metrics = self.__metric(
            results["image"].detach(),
            target_image.detach(),
            results["mask_prediction"].detach(),
        )

        return results, losses, metrics

    def forward_test(self, image: Image) -> Image:
        results = self(image)
        return results["image"]
