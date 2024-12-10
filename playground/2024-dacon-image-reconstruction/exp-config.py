"""실험에 사용될 Configuration 파일
"""

DATA_ROOT = "F:/datasets/2024_dacon_image_restoration"

MODEL = {"name"}


cfg = {
    "dataloader": {
        "dataset": {"name": "Dacon2024ImageReconstruction", "root": DATA_ROOT},
        "batch_size": 8,
        "shuffle": True,
        "transforms": [],
    },
    "model_interface": {
        "name": "Dacon2024ImageReconstructionModelInterface",
        "model": {"name": "CustomModel"},
        "loss": {"name": "torch.nn.L1Loss"},
    },
}
