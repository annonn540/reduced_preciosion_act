import torch.nn as nn
from fp8_models.act import FP8ReLU, FP8Sigmoid, FP8Tanh, FP8GELU, FP8LeakyReLU, FP8PReLU, FP8Swish, FP8Softplus
from torchvision.models import mobilenet_v2

class FP8MobileNetV2(nn.Module):
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
        base_model = mobilenet_v2(num_classes=num_classes)
        self._replace_activations(base_model, ActivationClass)
        self.model = base_model

    def _replace_activations(self, model, activation_class):
        for name, module in model.named_children():
            if isinstance(module, nn.ReLU6) or isinstance(module, nn.ReLU):
                setattr(model, name, activation_class())
            elif len(list(module.children())) > 0:
                self._replace_activations(module, activation_class)
    def forward(self, x):
        return self.model(x)
