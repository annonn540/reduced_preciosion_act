from torchvision.models import resnet18
import torch.nn as nn
from fp2_models.act import FP2ReLU, FP2Sigmoid, FP2Tanh, FP2GELU, FP2LeakyReLU, FP2PReLU, FP2Swish, FP2Softplus

class FP2ResNet18(nn.Module):
    def __init__(self, num_classes=10, activation="relu"):
        super().__init__()
        activation_map = {
            "relu": FP2ReLU,
            "sigmoid": FP2Sigmoid,
            "tanh": FP2Tanh,
            "gelu": FP2GELU,
            "leaky_relu": FP2LeakyReLU,
            "prelu": FP2PReLU,
            "swish": FP2Swish,
            "softplus": FP2Softplus
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