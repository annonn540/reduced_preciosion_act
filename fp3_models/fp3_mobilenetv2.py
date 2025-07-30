import torch.nn as nn
from fp3_models.act import FP3ReLU, FP3Sigmoid, FP3Tanh, FP3GELU, FP3LeakyReLU, FP3PReLU, FP3Swish, FP3Softplus
from torchvision.models import mobilenet_v2

class FP3MobileNetV2(nn.Module):
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
