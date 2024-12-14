"""dataset.py
2024 dacon image reconstruction dataset
"""

from pathlib import Path
from typing import Optional, Tuple, Callable, List, Dict

from PIL import Image
import torch
import torchvision
from torchvision.transforms import Compose

from deep_studio.data_layer.data_registry import DATASET_REGISTRY, TRANSFORM_REGISTRY


@TRANSFORM_REGISTRY.register
class Dacon2024ImageReconstructionTransform:
    def __init__(self, compose: List[Dict], **kwargs):

        compose_list = []
        for elm in compose:
            compose_list.append(TRANSFORM_REGISTRY.build(**elm))

        self.compose_list = compose_list

    def __call__(
        self, input_image: Image, gt_image: Optional[Image] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        images = (input_image, gt_image)
        for elm in self.compose_list:
            images = elm(*images)

        return images


@TRANSFORM_REGISTRY.register
class ToTensor:
    def __init__(self):
        self.input_totensor = torchvision.transforms.ToTensor()
        self.gt_totensor = torchvision.transforms.ToTensor()

    def __call__(
        self, input_image: Image, gt_image: Optional[Image] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.input_totensor(input_image), (
            self.gt_totensor(gt_image) if gt_image else None
        )


@DATASET_REGISTRY.register
class Dacon2024ImageReconstructionDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root: str,
        image_folder: str,
        gt_folder: Optional[str] = None,
        transforms: Optional[Callable] = None,
    ):

        self.root = root
        self.input_path = Path(self.root, image_folder)
        self.gt_path = Path(self.root, gt_folder) if gt_folder else None
        self.transforms = TRANSFORM_REGISTRY.build(**transforms)

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
            image_name = input_image_dir.parts[-1]
            gt_image = Image.open(Path(self.gt_path, image_name))

        if self.transforms:
            input_image, gt_image = self.transforms(input_image, gt_image)

        return input_image, gt_image
