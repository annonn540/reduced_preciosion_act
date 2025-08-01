import torch
import torch.nn as nn
import torch.nn.functional as F

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

class LearnableFPActivation(nn.Module):
    def __init__(self, precision_bits=2):
        super().__init__()
        
        if precision_bits == 2:
            init_vals = [-1.0, 0.0, 1.0, 4.0]
        elif precision_bits == 3:
            init_vals = [-1.0, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
        elif precision_bits == 4:
            init_vals = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, -0.5, -1.0, -2.0, -4.0]
        elif precision_bits == 6:
            init_vals = [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375,
            0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75,
            2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0,
            8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0,
            -0.0625, -0.125, -0.1875, -0.25, -0.3125, -0.375, -0.4375, -0.5,
            -0.625, -0.75, -0.875, -1.0, -1.25, -1.5, -1.75, -2.0,
            -2.5, -3.0, -3.5, -4.0, -5.0, -6.0, -7.0, -8.0,
            -10.0, -12.0, -14.0, -16.0, -20.0, -24.0, -28.0, -32.0]
        elif precision_bits == 8:
            init_vals = [
                0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0,
                -0.25, -0.5, -0.75, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0, -16.0, -24.0, -32.0, -48.0, -64.0
            ]
            vals = torch.tensor(sorted(set(init_vals + [float(i) for i in range(-128,128)])), dtype=torch.float32)
            init_vals = vals[:256].tolist()
        else:
            raise ValueError(f"Unsupported precision: {precision_bits}")
        
        vals = torch.tensor(sorted(set(init_vals)), dtype=torch.float32)
        self.fp_values = nn.Parameter(vals)

    def forward(self, x):
        sorted_vals, _ = torch.sort(self.fp_values)
        return STEQuantize.apply(x, sorted_vals)

class FP2Softmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.fp2 = LearnableFPActivation(precision_bits=2)
        self.dim = dim

    def forward(self, x):
        softmax_out = F.softmax(x, dim=self.dim)
        return self.fp2(softmax_out)

class FP3Softmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.fp3 = LearnableFPActivation(precision_bits=3)
        self.dim = dim

    def forward(self, x):
        softmax_out = F.softmax(x, dim=self.dim)
        return self.fp3(softmax_out)

class FP4Softmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.fp4 = LearnableFPActivation(precision_bits=4)
        self.dim = dim

    def forward(self, x):
        softmax_out = F.softmax(x, dim=self.dim)
        return self.fp4(softmax_out)

class FP6Softmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.fp6 = LearnableFPActivation(precision_bits=6)
        self.dim = dim

    def forward(self, x):
        softmax_out = F.softmax(x, dim=self.dim)
        return self.fp6(softmax_out)

class FP8Softmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.fp8 = LearnableFPActivation(precision_bits=8)
        self.dim = dim

    def forward(self, x):
        softmax_out = F.softmax(x, dim=self.dim)
        return self.fp8(softmax_out)

class FP32Softmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, dim=self.dim)

def get_softmax_layer(precision="fp32", dim=-1):
    softmax_map = {
        "fp2": FP2Softmax,
        "fp3": FP3Softmax,
        "fp4": FP4Softmax,
        "fp6": FP6Softmax,
        "fp8": FP8Softmax,
        "fp32": FP32Softmax
    }
    
    if precision.lower() not in softmax_map:
        raise ValueError(f"Unsupported softmax precision: {precision}. Choose from {list(softmax_map.keys())}")
    
    return softmax_map[precision.lower()](dim=dim)