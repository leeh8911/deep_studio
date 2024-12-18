from typing import Dict, Union

from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from deep_studio.data_layer.data_registry import (
    DATALOADER_FACTORY_REGISTRY,
    DATASET_REGISTRY,
)


@DATALOADER_FACTORY_REGISTRY.register
class DataLoaderFactory:

    def __init__(
        self, dataset_dict: Dict, split: Dict[str, float], seed: int = 42, **kwargs
    ):
        self.dataset_dict = dataset_dict
        self.split = split
        self.seed = seed
        self.kwargs = kwargs
        self.loaded_datasets = {}  # Lazy loading을 위해 캐싱된 데이터셋
        self.dataloader_cache = {}  # Lazy loading된 DataLoader 캐시

    def _lazy_load_dataset(self, split_key: str) -> DataLoader:
        """
        데이터셋을 필요한 시점에 생성하고 DataLoader를 반환.

        Args:
            split_key: "train", "validation", "test" 중 하나

        Returns:
            DataLoader
        """
        # 메모리 관리: 불필요한 데이터 삭제
        self._clear_other_dataloaders(split_key)

        if split_key not in self.dataloader_cache:
            if split_key == "test":
                # Test 데이터셋은 나눌 필요 없이 그대로 사용
                dataset = DATASET_REGISTRY.build(**self.dataset_dict["test"])
                self.dataloader_cache["test"] = DataLoader(
                    dataset, **self.kwargs["test"]
                )
            else:
                # Train/Validation 분할 수행
                full_dataset = DATASET_REGISTRY.build(**self.dataset_dict["train"])
                indices = np.arange(len(full_dataset))

                train_indices, val_indices = train_test_split(
                    indices,
                    test_size=self.split["validation"]
                    / (self.split["train"] + self.split["validation"]),
                    random_state=self.seed,
                )

                # Subset 생성
                train_dataset = Subset(full_dataset, train_indices)
                val_dataset = Subset(full_dataset, val_indices)

                # DataLoader 생성 및 캐싱
                self.dataloader_cache["train"] = DataLoader(
                    train_dataset, **self.kwargs["train"]
                )
                self.dataloader_cache["validation"] = DataLoader(
                    val_dataset, **self.kwargs["validation"]
                )

        return self.dataloader_cache[split_key]

    def _clear_other_dataloaders(self, current_key: str):
        """
        현재 로드된 split 이외의 데이터로더를 메모리에서 삭제.

        Args:
            current_key: 현재 로드하려는 데이터로더 키 ("train", "validation", "test")
        """
        if current_key == "test":
            keys_to_delete = {"train", "validation"}
        else:
            keys_to_delete = {"test"}

        for key in keys_to_delete:
            if key in self.dataloader_cache:
                del self.dataloader_cache[key]  # 메모리에서 삭제
                print(f"Deleted {key} DataLoader from memory.")

    def get(self, split: str) -> DataLoader:
        """
        데이터로더 반환. 필요한 경우 데이터셋을 생성합니다.

        Args:
            split: "train", "validation", "test" 중 하나

        Returns:
            DataLoader
        """
        if split not in ["train", "validation", "test"]:
            raise ValueError(
                f"Invalid split '{split}'. Must be 'train', 'validation', or 'test'."
            )

        return self._lazy_load_dataset(split)
