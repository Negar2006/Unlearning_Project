import torch
import torch.nn as nn
from typing import Literal

# Q1: Per-Channel Symmetric Uniform RTN
class PerChannelRTNLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, bits: int = 8):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = (2 ** (bits - 1)) - 1

        w = original_linear.weight.detach().float()
        
        scale = (torch.amax(torch.abs(w), dim=1, keepdim=True) / self.qmax).clamp(min=1e-8)
        
        q_w = torch.clamp(torch.round(w / scale), self.qmin, self.qmax)
        dequant_w = (q_w * scale).to(original_linear.weight.dtype)

        self.register_buffer("weight", dequant_w)
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

# Q2: Groupwise / Blockwise RTN (Group Size = 64)
class GroupwiseRTNLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, bits: int = 8, group_size: int = 64):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.bits = bits
        self.group_size = group_size
        self.qmin = -(2 ** (bits - 1))
        self.qmax = (2 ** (bits - 1)) - 1

        w = original_linear.weight.detach().float()
        out_f, in_f = w.shape

        if in_f % group_size == 0:
            w_grouped = w.view(out_f, in_f // group_size, group_size)
            scale = (torch.amax(torch.abs(w_grouped), dim=-1, keepdim=True) / self.qmax).clamp(min=1e-8)
            q_w = torch.clamp(torch.round(w_grouped / scale), self.qmin, self.qmax)
            dequant_w = (q_w * scale).view(out_f, in_f).to(original_linear.weight.dtype)
        else:
            scale = (torch.amax(torch.abs(w), dim=1, keepdim=True) / self.qmax).clamp(min=1e-8)
            q_w = torch.clamp(torch.round(w / scale), self.qmin, self.qmax)
            dequant_w = (q_w * scale).to(original_linear.weight.dtype)

        self.register_buffer("weight", dequant_w)
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

def apply_linear_replacement(module: nn.Module, quant_cls, bits: int) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, quant_cls(child, bits=bits))
        else:
            apply_linear_replacement(child, quant_cls, bits=bits)
    return module

def quantize_transformer(
    transformer: nn.Module,
    method: Literal["none", "q1_rtn", "q2_blockwise"],
    precision: Literal["fp16", "int8", "int4"]
) -> nn.Module:
    if precision == "fp16" or method == "none":
        return transformer.to(dtype=torch.float16)

    bits = 8 if precision == "int8" else 4

    if method == "q1_rtn":
        return apply_linear_replacement(transformer, PerChannelRTNLinear, bits=bits)
    elif method == "q2_blockwise":
        return apply_linear_replacement(transformer, GroupwiseRTNLinear, bits=bits)
    else:
        raise ValueError(f"Unsupported method: {method}")