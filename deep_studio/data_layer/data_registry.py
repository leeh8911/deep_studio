"""data_registry.py
"""

from deep_studio.common_layer.registry import make_registry

DATASET_REGISTRY = make_registry("DATASET_REGISTRY")

DATALOADER_FACTORY_REGISTRY = make_registry("DATALOADER_FACTORY_REGISTRY")

TRANSFORM_REGISTRY = make_registry("TRANSFORM_REGISTRY")
TRANSFORM_REGISTRY.set_allow_ext_modules(["torchvision"])
