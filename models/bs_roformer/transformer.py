import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import einops
from typing import Tuple


def choose_low_precision_dtype() -> torch.dtype:
    """
    GPU の機能を調べて BF16 → FP16 → FP32 の順に
    最も高速な演算 dtype を返す。
    """
    if not torch.cuda.is_available():
        return torch.float32  # CPU 実行なら FP32 一択

    # Ampere (sm80) 以降ならほぼ BF16 演算に対応
    if torch.cuda.is_bf16_supported():  # PyTorch 2.1+
        return torch.bfloat16

    major_cc, _ = torch.cuda.get_device_capability()
    # Pascal (sm60) 以降なら FP16 演算ユニットあり
    if major_cc >= 6:
        return torch.float16

    return torch.float32  # それ以前の Maxwell など


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 5.960464477539063e-08):  # 0x1p-24
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        l2_norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        denom = torch.maximum(l2_norm, torch.full_like(l2_norm, self.eps))
        normalized_x = x / denom
        return normalized_x * self.scale * self.gamma


class RotaryEmbedding(nn.Module):
    def __init__(self, cos_emb, sin_emb):
        super().__init__()
        # both (seq_len_for_rotation, dim_head)
        self.cos_emb = cos_emb
        self.sin_emb = sin_emb

    def rotate_half(self, x):
        x = einops.rearrange(x, "... (d r) -> ... d r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return einops.rearrange(x, "... d r -> ... (d r)")

    def forward(self, x):
        # x is (batch_eff, heads, seq_len_for_rotation, dim_head)
        cos_b = self.cos_emb.unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
        sin_b = self.sin_emb.unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)

        term1 = x * cos_b
        term2 = self.rotate_half(x) * sin_b

        sum = term1.to(torch.float32) + term2.to(torch.float32)
        return sum.to(x.dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_hidden_size_factor=4, dropout=0.0):
        super().__init__()
        dim_inner = int(dim * ffn_hidden_size_factor)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads=8,
        head_dim=64,
        shared_qkv_bias=None,
        shared_out_bias=None,
        rotary_embed: RotaryEmbedding | None = None,
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_size = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.norm = RMSNorm(input_dim)
        self.to_qkv = nn.Linear(
            input_dim, self.hidden_size * 3, bias=(shared_qkv_bias is not None)
        )
        if shared_qkv_bias is not None:
            self.to_qkv.bias = shared_qkv_bias

        self.to_gates = nn.Linear(input_dim, num_heads)
        self.to_out = nn.Sequential(
            nn.Linear(self.hidden_size, input_dim, bias=(shared_out_bias is not None)),
            nn.Dropout(dropout),
        )
        if shared_out_bias is not None:
            self.to_out[0].bias = shared_out_bias

        self.rotary_embed = rotary_embed
        self.lowp_dtype = choose_low_precision_dtype()

    def forward(self, x):
        x = self.norm(x)

        q, k, v = einops.rearrange(
            self.to_qkv(x), "b t (qkv h d) -> qkv b h t d", qkv=3, h=self.num_heads
        )

        if self.rotary_embed is not None:
            q = self.rotary_embed(q)
            k = self.rotary_embed(k)

        q = q.to(self.lowp_dtype)
        k = k.to(self.lowp_dtype)
        v = v.to(self.lowp_dtype)
        with sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            fetched = F.scaled_dot_product_attention(q, k, v)

        gates = self.to_gates(x)
        gates = gates.sigmoid()

        out = fetched.float() * einops.rearrange(gates, "b n h -> b h n 1")
        out = einops.rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_hidden_size_factor: int = 4,
        dropout: float = 0.0,
        shared_qkv_bias=None,
        shared_out_bias=None,
        output_norm: bool = False,
        rotary_embed: RotaryEmbedding | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(num_layers):
            attention = MultiHeadAttention(
                input_dim=input_dim,
                head_dim=head_dim,
                num_heads=num_heads,
                shared_qkv_bias=shared_qkv_bias,
                shared_out_bias=shared_out_bias,
                rotary_embed=rotary_embed,
            )
            self.layers.append(
                nn.ModuleList(
                    [
                        attention,
                        FeedForward(
                            dim=input_dim, ffn_hidden_size_factor=ffn_hidden_size_factor
                        ),
                    ]
                )
            )

        self.norm = RMSNorm(input_dim) if output_norm else nn.Identity()

    def forward(self, x):
        # x: [B, T, F]
        for attention, ffn in self.layers:
            x = attention(x) + x
            x = ffn(x) + x

        x = self.norm(x)
        return x
