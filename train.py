from pytorchkpn.registry import MODEL_REGISTRY
from pytorchkpn.config import get_cfg
import pytorchkpn.models as models

cfg = get_cfg()

print(MODEL_REGISTRY.get(cfg.MODEL.NAME))