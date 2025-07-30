import torch
import torch.nn as nn
from fp8_models.act import FP8ReLU, FP8Sigmoid, FP8Tanh, FP8GELU, FP8LeakyReLU, FP8PReLU, FP8Swish, FP8Softplus

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, activation_class):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            activation_class(),
            nn.Linear(mlp_dim, dim)
        )
        
    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        norm_x = self.norm2(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out
        
        return x

class FP8VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=8, num_classes=10, dim=128, depth=4, heads=4, mlp_dim=256, activation="gelu"):
        super().__init__()
        assert img_size % patch_size == 0
        
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
        
        Act = activation_map[activation]
        
        num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(0.1)
        
        # Create transformer blocks
        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, Act)
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
        
        x = x + self.pos_embed
        x = self.dropout(x)
        
        for block in self.transformer:
            x = block(x)
        
        x = self.to_cls_token(x[:, 0])
        
        return self.mlp_head(x)