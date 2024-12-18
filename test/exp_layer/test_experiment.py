"""test_experiment.py
deep_studio/exp_layer/experiment.py의 테스트를 수행하는 스크립트입니다.
"""

import pytest

import os
import sys
from pathlib import Path
import shutil

import torch

from deep_studio.exp_layer import Experiment
from deep_studio.model_layer.model_registry import MODEL_INTERFACE_REGISTRY
from deep_studio.data_layer.data_registry import DATASET_REGISTRY, DATALOADER_FACTORY_REGISTRY
from deep_studio.model_layer.model_interface import BaseModelInterface

TEST_EXPERIMENT_FOLDER_PATH = Path("./temporal_folder").absolute()
TEST_EXPERIMENT_CONFIG_FOLDER_PATH = Path(TEST_EXPERIMENT_FOLDER_PATH, "configs")
TEST_EXPERIMENT_CONFIG_PATH = Path(TEST_EXPERIMENT_CONFIG_FOLDER_PATH, "temporal_config.py")

@pytest.fixture
def generate_temporal_folder():
    TEST_EXPERIMENT_FOLDER_PATH.mkdir(exist_ok=True)
    TEST_EXPERIMENT_CONFIG_FOLDER_PATH.mkdir(exist_ok=True)
    
    yield
    
    shutil.rmtree(TEST_EXPERIMENT_FOLDER_PATH)
    
@pytest.fixture
def generate_temporal_classes(generate_temporal_folder):
    @MODEL_INTERFACE_REGISTRY.register
    class TemporalModelInterface(BaseModelInterface):
        def __init__(self):
            super().__init__()
            self.empty_layer = torch.nn.Linear(1,1)
        def forward_train(self):
            pass
        def forward_test(self):
            pass
        
    @DATASET_REGISTRY.register
    class TemporalDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.x = torch.rand(10, 2)
            self.y = torch.rand(10)
        def __len__(self):
            return len(self.x)
        
        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]
        
    
    
    
@pytest.fixture
def generate_config(generate_temporal_classes):
    config_file = """
cfg = {
    "device": "cpu",
    "seed": 42,
    "max_epoch": 1,
    "checkpoint_period": 1,
    "model_interface": {"name": "TemporalModelInterface"},
    "optimizer": {"name": "torch.optim.SGD", "lr": 1e-4},
    "dataloader": {"name": "DataLoaderFactory", "split": {"train": 0.9, "validation": 0.1}, "dataset_dict": {"train": {"name": "TemporalDataset"}}, "train": {"batch_size":1, "shuffle": True}, "validation": {"batch_size":1, "shuffle":False}}
}
"""
    with open(TEST_EXPERIMENT_CONFIG_PATH, "w", encoding='utf-8') as f:
        f.write(config_file)
        
    yield
    if os.path.exists(TEST_EXPERIMENT_CONFIG_PATH):
        os.remove(TEST_EXPERIMENT_CONFIG_PATH)
    
def test_experiment_initialize(generate_config):
    """ Experiment 실행 시, 저장이 필요한 파일들이 제대로 된 위치에 추가되는지를 확인함.
    playground/project-name/    # 프로젝트의 최상위 경로
    │
    └── experiment/             # Experiment 저장 디렉토리
        └── {YYMMDD_HHMMSS}_{config_file_name}/   # 특정 실험 실행 시 생성되는 디렉토리
            ├── EPOCH_{num}/                    # 각 에폭의 결과를 저장
            │   ├── checkpoint.pth              # 체크포인트 파일
            │   └── result...                   # 결과 파일들
            │
            └── BEST_CHECKPOINT.pth             # 가장 좋은 체크포인트

    """
    sys.argv = [
        "test_experiment",  # 프로그램 이름
        "--config", str(TEST_EXPERIMENT_CONFIG_PATH),
    ]

    # Experiment 실행
    exp = Experiment()

    # Workspace 및 폴더 트리 확인
    exp_dir = exp.exp_dir
    checkpoint_dir = exp.checkpoint_dir

    # 1. 상위 폴더가 제대로 생성되었는지 확인
    assert exp.workspace == TEST_EXPERIMENT_FOLDER_PATH
    assert exp_dir.exists()
    assert checkpoint_dir.exists()

    # 2. 체크포인트 저장 테스트
    exp.train()  # train 메서드 실행 → 체크포인트 파일 저장
    
    best_checkpoint = exp_dir / "BEST-CHECKPOINT.pth"
    epoch_0_checkpoint = checkpoint_dir / "0-CHECKPOINT.pth"

    assert best_checkpoint.exists(), "BEST-CHECKPOINT.pth 파일이 존재해야 합니다."
    assert epoch_0_checkpoint.exists(), "0-CHECKPOINT.pth 파일이 존재해야 합니다."

    # 3. 폴더 구조 검증
    print(f"Experiment Directory: {exp_dir}")
    print(f"Checkpoint Directory: {checkpoint_dir}")
    print("BEST-CHECKPOINT.pth와 0-CHECKPOINT.pth가 생성되었습니다.")
