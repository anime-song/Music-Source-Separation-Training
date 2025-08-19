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
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class RotaryEmbeddings(torch.nn.Module):
    """
    RoPE 用の sin・cos テーブルをキャッシュし、必要に応じて伸張／切り詰めるクラス。

    Args:
        head_dim: 1 ヘッドあたりの埋め込み次元 (必ず偶数にする)
        max_seq_len: 事前に準備しておく最大シーケンス長
        base_theta: 周波数スケーリング係数 (多くの論文では 10000.0)
        learned: True にすると sin, cos をパラメータとして学習させられる
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        base_theta: float = 10000.0,
        learned: bool = False,
        device: torch.device | None = None,
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("head_dim (＝1 ヘッドの次元数) は偶数にしてください。")

        self.head_dim = head_dim
        self.base_theta = base_theta
        self.max_seq_len = max_seq_len

        # 角周波数: θ_k = (θ_base)^(2k / d)
        freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base_theta ** (freqs / head_dim))  # (head_dim/2,)

        # time 方向へアウター積 → (max_seq_len, head_dim/2)
        t = torch.arange(max_seq_len, device=device).float()
        sinusoid_inp = torch.einsum("i,j->ij", t, inv_freq)

        sin, cos = (
            sinusoid_inp.sin(),
            sinusoid_inp.cos(),
        )  # 各が (max_seq_len, head_dim/2)

        if learned:
            self.register_parameter("sin_cached", torch.nn.Parameter(sin))
            self.register_parameter("cos_cached", torch.nn.Parameter(cos))
        else:
            self.register_buffer("sin_cached", sin, persistent=False)
            self.register_buffer("cos_cached", cos, persistent=False)

    def forward(
        self,
        seq_len: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        指定長の sin, cos を返す。

        Returns:
            cos: (seq_len, head_dim/2)
            sin: (seq_len, head_dim/2)
        """
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"要求シーケンス長 {seq_len} は max_seq_len={self.max_seq_len} を超えています。"
            )
        cos = self.cos_cached[:seq_len].to(dtype=dtype, device=device)
        sin = self.sin_cached[:seq_len].to(dtype=dtype, device=device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    偶数次元のテンソルを (…, 2i, 2i+1) → (…, -2i+1, 2i) のように 90° 回転させる。
    具体的には (x_even, x_odd) → (-x_odd, x_even)。
    """
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rotary_embedding(
    query: torch.Tensor,  # (batch, num_heads, seq_len, head_dim)
    key: torch.Tensor,  # (batch, num_heads, seq_len, head_dim)
    cos: torch.Tensor,  # (seq_len, head_dim/2)
    sin: torch.Tensor,  # (seq_len, head_dim/2)
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)

    q_rot = query * cos + rotate_half(query) * sin
    k_rot = key * cos + rotate_half(key) * sin
    return q_rot, k_rot


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

        self.rope = RotaryEmbeddings(
            head_dim=self.head_dim,
            learned=False,
        )
        self.lowp_dtype = choose_low_precision_dtype()

    def forward(self, x):
        x = self.norm(x)

        q, k, v = einops.rearrange(
            self.to_qkv(x), "b t (qkv h d) -> qkv b h t d", qkv=3, h=self.num_heads
        )

        cos, sin = self.rope(q.shape[-2], dtype=q.dtype, device=q.device)
        q, k = apply_rotary_embedding(q, k, cos, sin)

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

        self.norm = RMSNorm(input_dim)

    def forward(self, x):
        # x: [B, T, F]
        for attention, ffn in self.layers:
            x = attention(x) + x
            x = ffn(x) + x

        x = self.norm(x)
        return x
