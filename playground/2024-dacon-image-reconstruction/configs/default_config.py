"""실험에 사용될 Configuration 파일
"""

DATA_ROOT = "F:/datasets/2024_dacon_image_restoration"

TRAIN_DATALOADER = {
    "name": "DataLoader",
    "dataset": {
        "name": "Dacon2024ImageReconstructionDataset",
        "root": DATA_ROOT,
        "image_folder": "train_input",
        "gt_folder": "train_gt",
        "transforms": {
            "name": "Dacon2024ImageReconstructionTransform",
            "compose": [{"name": "ToTensor"}],
        },
    },
    "batch_size": 8,
    "shuffle": True,
}
TEST_DATALOADER = {
    "name": "DataLoader",
    "dataset": {
        "name": "Dacon2024ImageReconstructionDataset",
        "root": DATA_ROOT,
        "input_path": "test_input",
        "gt_path": None,
    },
    "batch_size": 8,
    "shuffle": False,
}

cfg = {
    "train": True,
    "seed": 42,
    "device": "cuda",
    "dataloader": TRAIN_DATALOADER,
    "model_interface": {
        "name": "Dacon2024ImageReconstructionModelInterface",
        "model": {"name": "CustomModel"},
        "loss": {"name": "torch.nn.L1Loss"},
    },
}
