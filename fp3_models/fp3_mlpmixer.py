import torch.nn as nn
from fp3_models.act import FP3ReLU, FP3Sigmoid, FP3Tanh, FP3GELU, FP3LeakyReLU, FP3PReLU, FP3Swish, FP3Softplus

class FP3MLPMixer(nn.Module):
    def __init__(self, num_classes=10, activation="gelu", hidden_dim=512):
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
