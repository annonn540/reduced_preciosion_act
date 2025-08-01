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

class LearnableFP6Activation(nn.Module):
    def __init__(self):
        super().__init__()
        init_vals = [
            0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375,
            0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75,
            2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0,
            8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0,
            -0.0625, -0.125, -0.1875, -0.25, -0.3125, -0.375, -0.4375, -0.5,
            -0.625, -0.75, -0.875, -1.0, -1.25, -1.5, -1.75, -2.0,
            -2.5, -3.0, -3.5, -4.0, -5.0, -6.0, -7.0, -8.0,
            -10.0, -12.0, -14.0, -16.0, -20.0, -24.0, -28.0, -32.0
        ]
        vals = torch.tensor(sorted(set(init_vals))[:64], dtype=torch.float32)
        self.fp6_values = nn.Parameter(vals)

    def forward(self, x):
        sorted_vals, _ = torch.sort(self.fp6_values)
        return STEQuantize.apply(x, sorted_vals)

class FP6ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp6 = LearnableFP6Activation()

    def forward(self, x):
        x_q = self.fp6(x)
        zero_fp6 = torch.tensor(0.0, device=x.device)
        x_q = torch.where(x_q < 0, zero_fp6, x_q)
        return x_q

class FP6Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp6 = LearnableFP6Activation()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.fp6(self.sigmoid(x))

class FP6Tanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp6 = LearnableFP6Activation()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.fp6(self.tanh(x))

class FP6GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp6 = LearnableFP6Activation()
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fp6(self.gelu(x))

class FP6LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.fp6 = LearnableFP6Activation()
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        return self.fp6(self.leaky_relu(x))

class FP6PReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.fp6 = LearnableFP6Activation()
        self.prelu = nn.PReLU(num_parameters=num_parameters, init=init)

    def forward(self, x):
        return self.fp6(self.prelu(x))

class FP6Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp6 = LearnableFP6Activation()
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.fp6(self.silu(x))

class FP6Softplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp6 = LearnableFP6Activation()
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.fp6(self.softplus(x))