import torch.nn as nn
from fp2_models.act import FP2ReLU, FP2Sigmoid, FP2Tanh, FP2GELU, FP2LeakyReLU, FP2PReLU, FP2Swish, FP2Softplus

class FP2MLPMixer(nn.Module):
    def __init__(self, num_classes=10, activation="gelu", hidden_dim=512):
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
