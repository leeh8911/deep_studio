"""dataset.py
2024 dacon image reconstruction dataset
"""

from pathlib import Path
from typing import Optional, Tuple, Callable

from PIL import Image
import torch

from deep_studio.data_layer.data_registry import DATASET_REGISTRY


@DATASET_REGISTRY.register
class Dacon2024ImageReconstructionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        image_folder: str,
        gt_folder: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):

        self.root = root
        self.input_path = Path(self.root, image_folder)
        self.gt_path = Path(self.root, gt_folder) if gt_folder else None
        self.transform = transform

        assert self.input_path.exists()
        assert self.gt_path.exists() or self.gt_path is None

        self.input_path_list = sorted(self.input_path.glob("*.png"))

    def __len__(self) -> int:
        return len(self.input_path_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_image_dir = self.input_path_list[idx]
        input_image = Image.open(input_image_dir)
        gt_image = None

        if self.gt_path:
            image_name = input_image_dir.split("/")[-1]
            gt_image = Image.open(Path(self.gt_path, image_name))

        if self.transform:
            input_image, gt_image = self.transform(input_image, gt_image)

        return input_image, gt_image
