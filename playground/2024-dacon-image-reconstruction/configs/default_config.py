"""실험에 사용될 Configuration 파일
"""

DATA_ROOT = "F:/datasets/2024_dacon_image_restoration"

TRAIN_DATALOADER = {
    "dataset": {
        "name": "Dacon2024ImageReconstructionDataset",
        "root": DATA_ROOT,
        "input_path": "train_input",
        "gt_path": "train_gt",
    },
    "batch_size": 8,
}
TEST_DATALOADER = {
    "dataset": {
        "name": "Dacon2024ImageReconstructionDataset",
        "root": DATA_ROOT,
        "input_path": "test_input",
        "gt_path": None,
    },
    "batch_size": 8,
}

cfg = {
    "train": True,
    "seed": 42,
    "dataloader": {
        "dataset": {
            "name": "Dacon2024ImageReconstructionDataset",
            "root": DATA_ROOT,
            "input_path": "train_input",
            "gt_path": "train_gt",
        },
        "batch_size": 8,
    },
    "model_interface": {
        "name": "Dacon2024ImageReconstructionModelInterface",
        "model": {"name": "CustomModel"},
        "loss": {"name": "torch.nn.L1Loss"},
    },
}
