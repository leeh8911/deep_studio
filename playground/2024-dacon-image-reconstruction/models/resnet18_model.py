from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F

from deep_studio.model_layer.model_registry import MODEL_REGISTRY


def upsampler(in_channels: int, out_channels: int, scale: int = 2) -> nn.Sequential:
    """
    입력 대비 출력 이미지 크기를 scale 배로 업샘플링하는 함수입니다.
    - ConvTranspose2d + BatchNorm2d + ReLU 조합

    Args:
        in_channels (int): 입력 채널 수
        out_channels (int): 출력 채널 수
        scale (int): 업스케일링 팩터 (2, 4, ...)

    Returns:
        nn.Sequential: 업샘플링 레이어
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels * (scale**2), kernel_size=3, padding=1),
        nn.PixelShuffle(scale),
        nn.ReLU(inplace=True),
    )


class MaskPredictor(nn.Module):
    """
    Attention Score 기반으로 마스크를 학습 가능한 Conv 기반 헤드로 예측합니다.
    """

    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.conv(x)


@MODEL_REGISTRY.register
class BACKBONE_ResNet18_TRANSFORMER_DECODER(nn.Module):

    def __init__(
        self, color_encoder: Dict, backbone: Dict, heads: List[Dict], **kwargs
    ):
        super().__init__()
        self.color_encoder = MODEL_REGISTRY.build(**color_encoder)
        self.backbone = MODEL_REGISTRY.build(**backbone)

        # Upsampling layers
        self.upsampling_layers = nn.ModuleDict(
            {
                "16_to_32": upsampler(512, 256, 2),
                "32_to_64": upsampler(256, 128, 2),
                "64_to_128": upsampler(128, 64, 2),
                "128_to_512": upsampler(64, 64, 4),
            }
        )

        # Transformer Decoder for cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=512, nhead=8, batch_first=True
        )
        self.cross_attention = nn.TransformerDecoder(decoder_layer, num_layers=3)

        # Learnable embedding for Key, Value
        self.learnable_embedding = nn.Parameter(torch.randn(1, 512, 1, 1))

        # Mask Prediction Head
        self.mask_predictor = MaskPredictor(in_channels=512)

        # Final Head layer
        self.head = nn.Sequential(*[MODEL_REGISTRY.build(**head) for head in heads])

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            image (torch.Tensor): 입력 이미지 텐서. (B, C, H, W)

        Returns:
            Dict[str, torch.Tensor]: 최종 이미지와 마스크 예측 결과
        """
        # 1. Color Encoding 및 Backbone Feature 추출
        encoded_image = self.color_encoder(image)
        features = self.backbone(encoded_image)
        image_128, image_64, image_32, image_16 = features

        # 2. Cross-Attention 수행
        refined_image_16 = self._apply_cross_attention(image_16)
        mask_prediction = self.mask_predictor(refined_image_16)
        mask_prediction = F.interpolate(
            mask_prediction, scale_factor=32, mode="bilinear", align_corners=False
        )

        # 3. Upsampling 및 Feature 결합
        image_32 = image_32 + self.upsampling_layers["16_to_32"](refined_image_16)
        image_64 = image_64 + self.upsampling_layers["32_to_64"](image_32)
        image_128 = image_128 + self.upsampling_layers["64_to_128"](image_64)
        upsampled_image = self.upsampling_layers["128_to_512"](image_128)

        # 4. Final Head Layer
        output_image = self.head(upsampled_image)

        return {"image": output_image, "mask_prediction": mask_prediction}

    def _apply_cross_attention(self, image_16: torch.Tensor) -> torch.Tensor:
        """
        Cross Attention을 수행하여 Refined Feature Map을 반환합니다.

        Args:
            image_16 (torch.Tensor): Feature Map (B, C, H, W)

        Returns:
            torch.Tensor: Attention을 적용한 Feature Map
        """
        B, C, H, W = image_16.shape

        # Use learnable embedding as Key and Value
        memory = self.learnable_embedding.expand(B, -1, H, W)

        # Flatten for Transformer Decoder
        tgt = image_16.view(B, C, H * W).transpose(1, 2)  # Query는 원본
        memory = memory.view(B, C, H * W).transpose(
            1, 2
        )  # Key, Value는 Learnable Embedding

        # Cross-Attention 수행
        refined_image_16 = self.cross_attention(tgt=tgt, memory=memory)
        refined_image_16 = refined_image_16.transpose(1, 2).view(B, C, H, W)
        return refined_image_16
