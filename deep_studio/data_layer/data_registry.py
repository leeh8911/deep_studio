"""data_registry.py
"""

from deep_studio.common_layer.registry import make_registry

DATASET_REGISTRY = make_registry("DATASET_REGISTRY")
DATALOADER_REGISTRY = make_registry("DATALOADER_REGISTRY")
TRANSFORM_REGISTRY = make_registry("TRANSFORM_REGISTRY")
