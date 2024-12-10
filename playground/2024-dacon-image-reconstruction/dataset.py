"""dataset.py
2024 dacon image reconstruction dataset
"""

from torch.utils.data import Dataset

from deep_studio.data_layer.data_registry import DATA_REGISTRY


@DATA_REGISTRY.register
class Dacon2024ImageReconstruction(Dataset):
    def __init__(self):
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> Image:
        pass
