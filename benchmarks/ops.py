"""
Real Transformer Ops — PyTorch reference implementations.

These are the operations Molten needs to fuse and beat.
Based on Qwen3 / Llama architecture.
"""

import torch
import torch.nn.functional as F
import math


def rmsnorm_pytorch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm — used in every Qwen3/Llama layer."""
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x / rms * weight


def silu_gate_pytorch(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """SiLU-gated MLP — the FFN activation in Qwen3/Llama."""
    return F.silu(gate) * x


def rope_pytorch(x: torch.Tensor, freqs_cos: torch.Tensor,
                 freqs_sin: torch.Tensor) -> torch.Tensor:
    """Rotary Position Embedding — applied to Q and K in every layer."""
    # x: (batch, seq, heads, head_dim)
    x_r = x.float().reshape(*x.shape[:-1], -1, 2)
    x0, x1 = x_r[..., 0], x_r[..., 1]
    cos = freqs_cos.unsqueeze(0).unsqueeze(2)  # (1, seq, 1, head_dim/2)
    sin = freqs_sin.unsqueeze(0).unsqueeze(2)
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    out = torch.stack([out0, out1], dim=-1).flatten(-2)
    return out.to(x.dtype)


def fused_rmsnorm_silu_gate_pytorch(x: torch.Tensor, norm_weight: torch.Tensor,
                                     gate: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused: RMSNorm → SiLU gate. Three ops that should be one kernel."""
    normed = rmsnorm_pytorch(x, norm_weight, eps)
    return silu_gate_pytorch(normed, gate)


def softmax_pytorch(x: torch.Tensor) -> torch.Tensor:
    """Standard softmax for attention scores."""
    return F.softmax(x, dim=-1)


def gelu_add_pytorch(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """GELU (tanh approx) + residual add — matches Llama/Qwen3 activation."""
    return F.gelu(x, approximate='tanh') + bias


def attention_score_pytorch(q: torch.Tensor, k: torch.Tensor,
                            scale: float) -> torch.Tensor:
    """Q @ K.T / sqrt(d) + softmax — attention score computation."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    return F.softmax(scores, dim=-1)


# --- Shape factories for benchmarking ---

def make_rmsnorm_inputs(batch: int, seq: int, dim: int,
                        device: str = "cuda", dtype=torch.float32):
    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)
    w = torch.ones(dim, device=device, dtype=dtype)
    return x, w


def make_silu_gate_inputs(batch: int, seq: int, dim: int,
                          device: str = "cuda", dtype=torch.float32):
    x = torch.randn(batch, seq, dim, device=device, dtype=dtype)
    gate = torch.randn(batch, seq, dim, device=device, dtype=dtype)
    return x, gate


def make_rope_inputs(batch: int, seq: int, heads: int, head_dim: int,
                     device: str = "cuda", dtype=torch.float32):
    x = torch.randn(batch, seq, heads, head_dim, device=device, dtype=dtype)
    freqs = torch.randn(seq, head_dim // 2, device=device, dtype=torch.float32)
    freqs_cos = freqs.cos()
    freqs_sin = freqs.sin()
    return x, freqs_cos, freqs_sin


def make_attention_inputs(batch: int, heads: int, seq: int, head_dim: int,
                          device: str = "cuda", dtype=torch.float32):
    q = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(head_dim)
    return q, k, scale
