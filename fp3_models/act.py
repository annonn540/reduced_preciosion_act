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

class LearnableFP3Activation(nn.Module):
    def __init__(self):
        super().__init__()
        init_vals = [
            -1.0, -0.25, 0.0, 0.25, 
            0.5, 1.0, 2.0, 4.0
        ]
        vals = torch.tensor(sorted(set(init_vals)), dtype=torch.float32)
        self.fp3_values = nn.Parameter(vals)

    def forward(self, x):
        sorted_vals, _ = torch.sort(self.fp3_values)
        return STEQuantize.apply(x, sorted_vals)

class FP3ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp3 = LearnableFP3Activation()

    def forward(self, x):
        x_q = self.fp3(x)
        zero_fp3 = torch.tensor(0.0, device=x.device)
        x_q = torch.where(x_q < 0, zero_fp3, x_q)
        return x_q

class FP3Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp3 = LearnableFP3Activation()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.fp3(self.sigmoid(x))

class FP3Tanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp3 = LearnableFP3Activation()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.fp3(self.tanh(x))

class FP3GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp3 = LearnableFP3Activation()
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fp3(self.gelu(x))

class FP3LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.fp3 = LearnableFP3Activation()
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        return self.fp3(self.leaky_relu(x))

class FP3PReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.fp3 = LearnableFP3Activation()
        self.prelu = nn.PReLU(num_parameters=num_parameters, init=init)

    def forward(self, x):
        return self.fp3(self.prelu(x))

class FP3Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp3 = LearnableFP3Activation()
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.fp3(self.silu(x))

class FP3Softplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp3 = LearnableFP3Activation()
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.fp3(self.softplus(x))