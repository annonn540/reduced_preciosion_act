from torchvision.models import resnet18
import torch.nn as nn
from fp3_models.act import FP3ReLU, FP3Sigmoid, FP3Tanh, FP3GELU, FP3LeakyReLU, FP3PReLU, FP3Swish, FP3Softplus

class FP3ResNet18(nn.Module):
    def __init__(self, num_classes=10, activation="relu"):
        super().__init__()
        activation_map = {
            "relu": FP3ReLU,
            "sigmoid": FP3Sigmoid,
            "tanh": FP3Tanh,
            "gelu": FP3GELU,
            "leaky_relu": FP3LeakyReLU,
            "prelu": FP3PReLU,
            "swish": FP3Swish,
            "softplus": FP3Softplus
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