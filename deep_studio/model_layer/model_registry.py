"""model_registry.py
"""

from deep_studio.common_layer.registry import make_registry

MODEL_INTERFACE_REGISTRY = make_registry("MODEL_INTERFACE_REGISTRY")

MODEL_REGISTRY = make_registry("MODEL_REGISTRY")
MODEL_REGISTRY.set_allow_ext_modules(["torch"])

LOSS_REGISTRY = make_registry("LOSS_REGISTRY")
LOSS_REGISTRY.set_allow_ext_modules(["torch"])

OPTIMIZER_REGISTRY = make_registry("OPTIMIZER_REGISTRY")
OPTIMIZER_REGISTRY.set_allow_ext_modules(["torch"])
