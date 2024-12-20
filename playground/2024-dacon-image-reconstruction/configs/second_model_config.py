"""실험에 사용될 Configuration 파일
"""

__all__ = ["cfg"]

DATA_ROOT = "F:/datasets/2024_dacon_image_restoration"

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"

SEED = 42

MAX_EPOCH = 16
BATCH_SIZE = 8

DATALOADER_FACTORY = {
    "name": "DataLoaderFactory",
    "dataset_dict": {
        TRAIN: {
            "name": "Dacon2024ImageReconstructionDataset",
            "root": DATA_ROOT,
            "image_folder": "train_input",
            "gt_folder": "train_gt",
            "transforms": {
                "name": "Dacon2024ImageReconstructionTransform",
                "compose": [{"name": "ToTensor"}],
            },
        },
        TEST: {
            "name": "Dacon2024ImageReconstructionDataset",
            "root": DATA_ROOT,
            "image_folder": "test_input",
            "gt_folder": None,
            "transforms": {
                "name": "Dacon2024ImageReconstructionTransform",
                "compose": [{"name": "ToTensor"}],
            },
        },
    },
    "seed": SEED,
    "split": {TRAIN: 0.9, VALIDATION: 0.1},
    TRAIN: {"batch_size": BATCH_SIZE, "shuffle": True},
    VALIDATION: {"batch_size": BATCH_SIZE, "shuffle": False},
    TEST: {"batch_size": BATCH_SIZE, "shuffle": False},
}

EMBED_SIZE = 16
cfg = {
    "seed": SEED,
    "device": "cuda",
    "max_epoch": MAX_EPOCH,
    "checkpoint_dir": "./checkpoint",
    "output_dir": "./output",
    "temp_output_dir": "./temp_output",
    "checkpoint_period": 1,
    "dataloader": DATALOADER_FACTORY,
    "model_interface": {
        "name": "Dacon2024ImageReconstructionModelInterface",
        "model": {
            "name": "SecondModel",
            "embed_layers": [
                {
                    "name": "CNBR",
                    "in_channels": 1,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 7,
                },
                {
                    "name": "CNBR",
                    "in_channels": EMBED_SIZE,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 5,
                },
                {
                    "name": "CNBR",
                    "in_channels": EMBED_SIZE,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 5,
                },
                {
                    "name": "CNBR",
                    "in_channels": EMBED_SIZE,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 5,
                },
                {
                    "name": "CNBR",
                    "in_channels": EMBED_SIZE,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 3,
                },
                {
                    "name": "CNBR",
                    "in_channels": EMBED_SIZE,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 3,
                },
            ],
            "downsample_layers": [
                {
                    "name": "CNBR",
                    "in_channels": EMBED_SIZE,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 3,
                },
                {
                    "name": "CNBR",
                    "in_channels": EMBED_SIZE,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 3,
                },
                {
                    "name": "CNBR",
                    "in_channels": EMBED_SIZE,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 3,
                },
                {
                    "name": "CNBR",
                    "in_channels": EMBED_SIZE,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 3,
                },
                {
                    "name": "CNBR",
                    "in_channels": EMBED_SIZE,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 3,
                },
                {
                    "name": "CNBR",
                    "in_channels": EMBED_SIZE,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 3,
                },
                {
                    "name": "CNBR",
                    "in_channels": EMBED_SIZE,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 3,
                },
            ],
            "num_encoder_transformer_layers": 1,
            "encoder_transformer_layer": {
                "name": "torch.nn.TransformerEncoderLayer",
                "d_model": EMBED_SIZE,
                "nhead": 4,
            },
            "num_decoder_transformer_layers": 1,
            "decoder_transformer_layer": {
                "name": "torch.nn.TransformerDecoderLayer",
                "d_model": EMBED_SIZE,
                "nhead": 4,
            },
            "upsample_layers": [
                {
                    "name": "CNBR",
                    "in_channels": EMBED_SIZE,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 3,
                },
                {
                    "name": "CNBR",
                    "in_channels": EMBED_SIZE,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 3,
                },
                {
                    "name": "CNBR",
                    "in_channels": EMBED_SIZE,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 3,
                },
                {
                    "name": "CNBR",
                    "in_channels": EMBED_SIZE,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 3,
                },
                {
                    "name": "CNBR",
                    "in_channels": EMBED_SIZE,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 3,
                },
                {
                    "name": "CNBR",
                    "in_channels": EMBED_SIZE,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 3,
                },
                {
                    "name": "CNBR",
                    "in_channels": EMBED_SIZE,
                    "out_channels": EMBED_SIZE,
                    "kernel_size": 3,
                },
            ],
            "head": {
                "name": "CNBR",
                "in_channels": EMBED_SIZE,
                "out_channels": 3,
                "kernel_size": 1,
            },
        },
        "criterions": {
            "color_criterion": {"name": "torch.nn.L1Loss"},
            "recon_criterion": {"name": "torch.nn.L1Loss"},
        },
    },
    "optimizer": {"name": "torch.optim.AdamW", "lr": 1e-3},
}
