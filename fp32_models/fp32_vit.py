import torch
import torch.nn as nn

class FP32VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=8, num_classes=10, dim=128, depth=4, heads=4, mlp_dim=256, activation="gelu"):
        super().__init__()

        assert img_size % patch_size == 0
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
        Act = activation_map[activation]

        num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

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
        p = int((x.size(2) * x.size(3)) ** 0.5 // ((self.pos_embed.size(1)-1)**0.5))
        x = x.unfold(2, p, p).unfold(3, p, p).contiguous()
        x = x.view(B, 3, -1, p, p).permute(0, 2, 1, 3, 4).reshape(B, -1, 3 * p * p)
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