import torch
import torch.nn as nn
from fp2_models.act import FP2ReLU, FP2Sigmoid, FP2Tanh, FP2GELU, FP2LeakyReLU, FP2PReLU, FP2Swish, FP2Softplus

class FP2CNN(nn.Module):
    def __init__(self, num_classes=10, activation="relu", use_softmax=False, softmax_precision="fp2"):
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
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            ActivationClass(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            ActivationClass(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            ActivationClass(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            ActivationClass(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            ActivationClass(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            ActivationClass(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            ActivationClass(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            ActivationClass(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if use_softmax:
            from softmax_layers.softmax_layers import get_softmax_layer
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                ActivationClass(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
                get_softmax_layer(softmax_precision, dim=1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                ActivationClass(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x