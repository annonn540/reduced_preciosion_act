import torch.nn as nn
from fp2_models.act import FP2ReLU, FP2Sigmoid, FP2Tanh, FP2GELU, FP2LeakyReLU, FP2PReLU, FP2Swish, FP2Softplus
from torchvision.models import mobilenet_v2

class FP2MobileNetV2(nn.Module):
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
