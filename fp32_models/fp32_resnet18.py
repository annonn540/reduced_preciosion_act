from torchvision.models import resnet18
import torch.nn as nn

class FP32ResNet18(nn.Module):
    def __init__(self, num_classes=10, activation="relu"):
        super().__init__()
        activation_map = {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "leaky_relu": lambda: nn.LeakyReLU(),
            "prelu": lambda: nn.PReLU(),
            "swish": nn.SiLU,
            "softplus": nn.Softplus
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