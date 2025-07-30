import torch.nn as nn
from fp4_models.act import FP4ReLU, FP4Sigmoid, FP4Tanh, FP4GELU, FP4LeakyReLU, FP4PReLU, FP4Swish, FP4Softplus

class FP4MLPMixer(nn.Module):
    def __init__(self, num_classes=10, activation="gelu", hidden_dim=512):
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
