from torchvision.models import resnet18
import torch.nn as nn
from fp6_models.act import FP6ReLU, FP6Sigmoid, FP6Tanh, FP6GELU, FP6LeakyReLU, FP6PReLU, FP6Swish, FP6Softplus

class FP6ResNet18(nn.Module):
    def __init__(self, num_classes=10, activation="relu"):
        super().__init__()
        activation_map = {
            "relu": FP6ReLU,
            "sigmoid": FP6Sigmoid,
            "tanh": FP6Tanh,
            "gelu": FP6GELU,
            "leaky_relu": FP6LeakyReLU,
            "prelu": FP6PReLU,
            "swish": FP6Swish,
            "softplus": FP6Softplus
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