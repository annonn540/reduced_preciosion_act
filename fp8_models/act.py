import torch
import torch.nn as nn

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, vals):
        idx = torch.searchsorted(vals, x, right=False)
        idx = torch.clamp(idx, 1, len(vals) - 1)
        high = vals[idx]
        low = vals[idx - 1]
        dist_high = torch.abs(x - high)
        dist_low = torch.abs(x - low)
        return torch.where(dist_low <= dist_high, low, high)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class LearnableFP8Activation(nn.Module):
    def __init__(self):
        super().__init__()
        init_vals = [
            0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0,
            -0.25, -0.5, -0.75, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0, -16.0, -24.0, -32.0, -48.0, -64.0
        ]
        vals = torch.tensor(sorted(set(init_vals + [float(i) for i in range(-128,128)])), dtype=torch.float32)
        self.fp8_values = nn.Parameter(vals[:256])

    def forward(self, x):
        sorted_vals, _ = torch.sort(self.fp8_values)
        return STEQuantize.apply(x, sorted_vals)

class FP8ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp8 = LearnableFP8Activation()

    def forward(self, x):
        x_q = self.fp8(x)
        zero_fp8 = torch.tensor(0.0, device=x.device)
        x_q = torch.where(x_q < 0, zero_fp8, x_q)
        return x_q

class FP8Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp8 = LearnableFP8Activation()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.fp8(self.sigmoid(x))

class FP8Tanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp8 = LearnableFP8Activation()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.fp8(self.tanh(x))

class FP8GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp8 = LearnableFP8Activation()
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fp8(self.gelu(x))

class FP8LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.fp8 = LearnableFP8Activation()
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        return self.fp8(self.leaky_relu(x))

class FP8PReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp8 = LearnableFP8Activation()
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.fp8(self.prelu(x))

class FP8Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp8 = LearnableFP8Activation()

    def forward(self, x):
        return self.fp8(x * torch.sigmoid(x))

class FP8Softplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp8 = LearnableFP8Activation()
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.fp8(self.softplus(x))
