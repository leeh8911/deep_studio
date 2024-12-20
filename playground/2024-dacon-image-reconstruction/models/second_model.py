from typing import Dict, List

import torch
from torch import nn

from deep_studio.model_layer.model_registry import MODEL_REGISTRY


@MODEL_REGISTRY.register
class SecondModel(nn.Module):

    def __init__(
        self,
        embed_layers: List[Dict],
        downsample_layers: List[Dict],
        num_encoder_transformer_layers: int,
        encoder_transformer_layer: Dict,
        num_decoder_transformer_layers: int,
        decoder_transformer_layer: Dict,
        upsample_layers: List[Dict],
        head: Dict,
        **kwargs
    ):
        super().__init__()

        # Embedding layers
        self.embed_layers = nn.ModuleList()
        for embed_layer in embed_layers:
            self.embed_layers.append(MODEL_REGISTRY.build(**embed_layer))

        # Downsampling layers (with MaxPool2d)
        self.downsample_layers = nn.ModuleList()
        for downsample_layer in downsample_layers:
            self.downsample_layers.append(
                nn.Sequential(
                    MODEL_REGISTRY.build(**downsample_layer),
                    nn.MaxPool2d(2, return_indices=True),
                )
            )

        # Transformer Encoder
        self.transformer_encoder_layer = MODEL_REGISTRY.build(
            **encoder_transformer_layer
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=num_encoder_transformer_layers
        )

        # Transformer Encoder
        self.transformer_decoder_layer = MODEL_REGISTRY.build(
            **decoder_transformer_layer
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer, num_layers=num_decoder_transformer_layers
        )

        # Upsampling layers (with MaxUnpool2d)
        self.upsample_layers = nn.ModuleList()
        for upsample_layer in upsample_layers:
            self.upsample_layers.append(
                nn.Sequential(MODEL_REGISTRY.build(**upsample_layer), nn.MaxUnpool2d(2))
            )

        # Final Head layer
        self.head = MODEL_REGISTRY.build(**head)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        indices_list = []  # MaxPool에서 반환된 인덱스를 저장
        sizes = []  # 원본 사이즈 저장 (Unpooling 용)
        residuals = []  # Residual 연결을 저장

        image[image == 0] += torch.rand_like(image)[image == 0]
        # Embedding layers
        for embed_layer in self.embed_layers:
            image = embed_layer(image)

        # Downsampling layers with Residuals
        for downsample_layer in self.downsample_layers:
            residuals.append(image)  # Residual 저장
            sizes.append(image.size())
            image, indices = downsample_layer(image)
            indices_list.append(indices)

        # Flatten for Transformer
        b, c, h, w = image.size()
        image = image.view(b, c, -1).permute(2, 0, 1)  # (seq_len, batch, channels)
        noise = torch.rand_like(image)

        # Transformer Encoder
        encode_image = self.transformer_encoder(image)
        decode_image = self.transformer_decoder(noise, encode_image)
        image = encode_image + decode_image

        # Reshape back to 2D
        image = image.permute(1, 2, 0).contiguous().view(b, c, h, w)

        # Upsampling layers with Residuals
        for upsample_layer, residual, indices, size in zip(
            self.upsample_layers,
            reversed(residuals),
            reversed(indices_list),
            reversed(sizes),
        ):
            image = upsample_layer[0](image)  # CNBR Layer
            image = upsample_layer[1](image, indices, output_size=size)  # MaxUnpool2d
            image = image + residual  # Residual 연결 추가

        # Final Head layer
        image = self.head(image)

        return image
