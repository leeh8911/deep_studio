from typing import Dict

import torch.utils.data.DataLoader
from sklearn.model_selection import train_test_split

from deep_studio.data_layer.data_registry import DATALOADER_REGISTRY, DATASET_REGISTRY


@DATALOADER_REGISTRY.register
class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: Dict, split: Dict[str, float], **kwargs):
        dataset = DATASET_REGISTRY.build(**dataset)
