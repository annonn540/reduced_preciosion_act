import torch.nn as nn
from fp8_models.act import FP8ReLU, FP8Sigmoid, FP8Tanh, FP8GELU, FP8LeakyReLU, FP8PReLU, FP8Swish, FP8Softplus

class FP8MLPMixer(nn.Module):
    def __init__(self, num_classes=10, activation="gelu", hidden_dim=512):
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
            raise ValueError(f"Unsupported activation: {activation}")
        
        act = activation_map[activation]
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
