import torch.nn as nn

class FP32MLPMixer(nn.Module):
    def __init__(self, num_classes=10, activation="gelu", hidden_dim=512):
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