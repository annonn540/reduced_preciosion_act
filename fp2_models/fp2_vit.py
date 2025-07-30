import torch
import torch.nn as nn
from fp2_models.act import FP2ReLU, FP2Sigmoid, FP2Tanh, FP2GELU, FP2LeakyReLU, FP2PReLU, FP2Swish, FP2Softplus

class FP2VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=None, num_classes=10, dim=128, depth=4, heads=4, mlp_dim=256, activation="gelu", input_shape=None):
        super().__init__()

        if input_shape is not None and isinstance(input_shape, (list, tuple)) and len(input_shape) == 3:
            img_size = input_shape[1]
            channels = input_shape[0]
        else:
            channels = 3

        if patch_size is None:
            if img_size == 28:
                patch_size = 7
            elif img_size == 32:
                patch_size = 8
            elif img_size == 64:
                patch_size = 8
            else:
                patch_size = max(4, img_size // 8)

        assert img_size % patch_size == 0
        
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
        Act = activation_map[activation]

        num_patches = (img_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer = nn.Sequential(*[
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    Act(),
                    nn.Linear(mlp_dim, dim)
                )
            )
            for _ in range(depth)
        ])

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        
        x = x.unfold(2, p, p).unfold(3, p, p).contiguous()
        x = x.view(B, C, -1, p, p).permute(0, 2, 1, 3, 4).reshape(B, -1, C * p * p)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        x = self.dropout(x)

        for block in self.transformer:
            norm1, attn, norm2, mlp = block
            x = x + attn(norm1(x), norm1(x), norm1(x))[0]
            x = x + mlp(norm2(x))

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)