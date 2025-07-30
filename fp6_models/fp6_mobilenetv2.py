import torch.nn as nn
from fp6_models.act import FP6ReLU, FP6Sigmoid, FP6Tanh, FP6GELU, FP6LeakyReLU, FP6PReLU, FP6Swish, FP6Softplus
from torchvision.models import mobilenet_v2

class FP6MobileNetV2(nn.Module):
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
