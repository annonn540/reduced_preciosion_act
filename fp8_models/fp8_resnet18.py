from torchvision.models import resnet18
import torch.nn as nn
from fp8_models.act import FP8ReLU, FP8Sigmoid, FP8Tanh, FP8GELU, FP8LeakyReLU, FP8PReLU, FP8Swish, FP8Softplus

class FP8ResNet18(nn.Module):
    def __init__(self, num_classes=10, activation="relu"):
        super().__init__()
        activation_map = {
            "relu": FP8ReLU,
            "sigmoid": FP8Sigmoid,
            "tanh": FP8Tanh,
            "gelu": FP8GELU,
            "leaky_relu": FP8LeakyReLU,
            "prelu": FP8PReLU,
            "swish": FP8Swish,
            "softplus": FP8Softplus
        }
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}. Choose from {list(activation_map.keys())}")
        ActivationClass = activation_map[activation]
        self.base_model = resnet18(num_classes=num_classes)
        def replace_relu(module):
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(module, name, ActivationClass())
                else:
                    replace_relu(child)
        replace_relu(self.base_model)
    def forward(self, x):
        return self.base_model(x)
