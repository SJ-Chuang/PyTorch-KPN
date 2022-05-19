from pytorchkpn.registry import MODEL_REGISTRY
from torchvision import models
import torch.nn as nn

__all__ = ["LRASPP"]

@MODEL_REGISTRY.register()
class LRASPP(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self._model = models.segmentation.lraspp_mobilenet_v3_large(num_classes=1)
    
    def forward(self, x):
        return self._model(x)["out"]