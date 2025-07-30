from torchvision.models import resnet18
import torch.nn as nn
from fp4_models.act import FP4ReLU, FP4Sigmoid, FP4Tanh, FP4GELU, FP4LeakyReLU, FP4PReLU, FP4Swish, FP4Softplus

class FP4ResNet18(nn.Module):
    def __init__(self, num_classes=10, activation="relu"):
        super().__init__()
        activation_map = {
            "relu": FP4ReLU,
            "sigmoid": FP4Sigmoid,
            "tanh": FP4Tanh,
            "gelu": FP4GELU,
            "leaky_relu": FP4LeakyReLU,
            "prelu": FP4PReLU,
            "swish": FP4Swish,
            "softplus": FP4Softplus
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