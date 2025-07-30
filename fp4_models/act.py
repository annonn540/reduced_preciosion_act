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

class LearnableFP4Activation(nn.Module):
    def __init__(self):
        super().__init__()
        init_vals = [
            0.0, 0.25, 0.5, 0.75,
            1.0, 1.5, 2.0, 3.0,
            4.0, 6.0, 8.0, 12.0,
            -0.5, -1.0, -2.0, -4.0
        ]
        vals = torch.tensor(sorted(set(init_vals)), dtype=torch.float32)
        self.fp4_values = nn.Parameter(vals)

    def forward(self, x):
        sorted_vals, _ = torch.sort(self.fp4_values)
        return STEQuantize.apply(x, sorted_vals)

class FP4ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp4 = LearnableFP4Activation()

    def forward(self, x):
        x_q = self.fp4(x)
        zero_fp4 = torch.tensor(0.0, device=x.device)
        x_q = torch.where(x_q < 0, zero_fp4, x_q)
        return x_q

class FP4Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp4 = LearnableFP4Activation()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.fp4(self.sigmoid(x))

class FP4Tanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp4 = LearnableFP4Activation()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.fp4(self.tanh(x))

class FP4GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp4 = LearnableFP4Activation()
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fp4(self.gelu(x))

class FP4LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.fp4 = LearnableFP4Activation()
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        return self.fp4(self.leaky_relu(x))

class FP4PReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.fp4 = LearnableFP4Activation()
        self.prelu = nn.PReLU(num_parameters=num_parameters, init=init)

    def forward(self, x):
        return self.fp4(self.prelu(x))

class FP4Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp4 = LearnableFP4Activation()
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.fp4(self.silu(x))

class FP4Softplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp4 = LearnableFP4Activation()
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.fp4(self.softplus(x))

class FP4Softmax(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.fp4 = LearnableFP4Activation()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        return self.fp4(self.softmax(x))
