import torch
import torch.nn as nn
import torch.utils.checkpoint
import librosa
import numpy as np
import einops
from .transformer import Transformer, RMSNorm

from typing import List, Tuple


def checkpoint_bypass(f, *args):
    return f(*args)


def build_mel_band_indices(
    sample_rate: int,
    n_fft: int,
    n_bands: int,
) -> list[np.ndarray[int]]:
    """
    各 Mel バンドがカバーする「周波数ビン番号だけ」を返す。
    返り値: list[np.ndarray]  (len = n_bands)
    """
    # --- Mel フィルタ → 0/1 マスク --------------------------------
    mel_fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_bands)
    mel_fb[0, 0] = 1.0
    mel_fb[-1, -1] = 1.0
    mask = mel_fb > 0  # (B, F) bool

    return [np.where(mask[b])[0].astype(np.int32) for b in range(n_bands)]


def build_band_indices(
    sample_rate: int = 44_100, n_fft: int = 2_048
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    BSRoformer 用バンド分割 (各バンドの bin 数が必ず閾値以上)。

    Returns
    -------
    freqs : np.ndarray
        rfft の周波数 (長さ n_fft//2 + 1) [Hz]
    bands : List[np.ndarray]
        各要素が「そのバンドに含まれるビン番号」の 1‑D 配列
    """
    freqs = np.fft.rfftfreq(n_fft, d=1 / sample_rate)

    # バンドが始まるビンの周波数から必要 bin 数を決定
    def bins_per_band(freq_hz: float) -> int:
        if freq_hz < 1_000:  # 0‑1 kHz
            return 2
        elif freq_hz < 2_000:  # 1‑2 kHz
            return 4
        elif freq_hz < 4_000:  # 2‑4 kHz
            return 12
        elif freq_hz < 8_000:  # 4‑8 kHz
            return 24
        elif freq_hz < 16_000:  # 8‑16 kHz
            return 48
        else:  # ≥16 kHz → 後で 2 分割
            return -1

    freq_indices: List[np.ndarray] = []
    idx = 0
    n_bins = len(freqs)

    while idx < n_bins:
        # 16 kHz 以上は残りを 2 分割して終了
        if freqs[idx] >= 16_000:
            remaining = np.arange(idx, n_bins)
            half = len(remaining) // 2
            freq_indices.append(remaining[:half])
            freq_indices.append(remaining[half:])
            break

        size = bins_per_band(freqs[idx])
        band = np.arange(idx, min(idx + size, n_bins))
        freq_indices.append(band)
        idx += size  # bin数 単位で次へ

    return freqs, freq_indices


class BandSplit(torch.nn.Module):
    def __init__(
        self, hidden_size: int, band_indices, num_channels: int, extra_windows: int
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.band_indices = band_indices

        for i, idx in enumerate(band_indices):
            self.register_buffer(
                f"band_idx_{i}", torch.tensor(idx, dtype=torch.long), persistent=False
            )

        self.num_bands = len(band_indices)
        self.to_features = torch.nn.ModuleList([])
        for i in range(self.num_bands):
            sub_band_freqs = (
                len(band_indices[i]) * num_channels * 2 * (extra_windows + 1)
            )
            self.to_features.append(
                torch.nn.Sequential(
                    RMSNorm(sub_band_freqs),
                    torch.nn.Linear(sub_band_freqs, hidden_size),
                )
            )

    def forward(self, x):
        # x: (B, T, C, F)
        sub_band_list = []
        for i, proj_layer in enumerate(self.to_features):
            band_indices = getattr(self, f"band_idx_{i}")
            # サブバンドインデックスで周波数軸から抜きだす
            sub_band = x[..., band_indices]  # (B, T, C, sub_band_freqs)
            sub_band = einops.rearrange(sub_band, "b t c f -> b t (c f)")
            sub_band = proj_layer(sub_band)  # (B, T, hidden_size)
            sub_band_list.append(sub_band)

        return torch.stack(sub_band_list, dim=-2)  # (B, T, num_bands, hidden_size)


def band_mask_mlp(input_dim, band_dim, hidden_dim, depth):
    layers = []
    # 入力→隠れ→…→隠れ→(band_dim*2) の構造
    dims = [input_dim] + [hidden_dim] * depth + [band_dim * 2]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        # 最終層の後に活性化は入れない
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class MaskEstimator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        band_indices,
        num_channels: int,
        mlp_expansion_factor: int = 4,
        depth: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.band_indices = band_indices

        self.num_bands = len(band_indices)
        self.to_freqs = torch.nn.ModuleList([])
        for i in range(self.num_bands):
            sub_band_freqs = len(band_indices[i]) * num_channels * 2
            self.to_freqs.append(
                nn.Sequential(
                    band_mask_mlp(
                        input_dim=input_dim,
                        band_dim=sub_band_freqs,
                        hidden_dim=input_dim * mlp_expansion_factor,
                        depth=depth,
                    ),
                    nn.GLU(dim=-1),
                )
            )

    def forward(self, x):
        # x: (B, T, K, D)
        out = []
        sub_band_list = torch.unbind(x, dim=2)  # list([B, T, D])
        assert len(sub_band_list) == len(self.to_freqs)

        for i, proj_layer in enumerate(self.to_freqs):
            sub_band = proj_layer(sub_band_list[i])  # (B, T, sub_band_freqs)
            sub_band = einops.rearrange(
                sub_band,
                "b t (c f) -> b t c f",
                c=self.num_channels * 2,
                f=len(self.band_indices[i]),
            )
            out.append(sub_band)

        return torch.concat(out, dim=-1)  # (B, T, C, F)


class BSRoformer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int = 12,
        sample_rate: int = 44100,
        num_channels: int = 2,
        head_dim: int = 64,
        num_heads: int = 8,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_bands: int = 60,
        num_stems: int = 2,
        band_split_type: str = "bs",
        ffn_hidden_size_factor: int = 4,
        dropout: float = 0.0,
        use_shared_bias: bool = True,
        use_gradient_checkpoint=True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.window_size = n_fft
        self.hop_length = hop_length
        self.num_channels = num_channels
        self.num_stems = num_stems
        self.band_split_type = band_split_type
        self.use_gradient_checkpoint = use_gradient_checkpoint

        if band_split_type == "bs":
            _, self.band_indices = build_band_indices(
                sample_rate=sample_rate, n_fft=n_fft
            )
        elif band_split_type == "mel":
            self.band_indices = build_mel_band_indices(
                sampling_rate=sample_rate, n_fft=n_fft, n_bands=n_bands
            )
        else:
            raise NotImplementedError("サポートしていない band_split_type です")

        self.register_buffer(
            "hann_window", torch.hann_window(self.window_size), persistent=False
        )
        self.num_bands = len(self.band_indices)
        self.band_split = BandSplit(
            hidden_size=dim,
            band_indices=self.band_indices,
            num_channels=num_channels,
            extra_windows=0,
        )

        self.layers = nn.ModuleList([])
        if use_shared_bias:
            hidden_size = head_dim * num_heads
            self.shared_qkv_bias = nn.Parameter(torch.ones(hidden_size * 3))  # QKV
            self.shared_out_bias = nn.Parameter(torch.ones(dim))  # OUT

        for _ in range(num_layers):
            time_roformer = Transformer(
                input_dim=dim,
                head_dim=head_dim,
                num_layers=1,
                num_heads=num_heads,
                ffn_hidden_size_factor=ffn_hidden_size_factor,
                dropout=dropout,
                shared_qkv_bias=self.shared_qkv_bias,
                shared_out_bias=self.shared_out_bias,
            )
            band_roformer = Transformer(
                input_dim=dim,
                head_dim=head_dim,
                num_layers=1,
                num_heads=num_heads,
                ffn_hidden_size_factor=ffn_hidden_size_factor,
                dropout=dropout,
                shared_qkv_bias=self.shared_qkv_bias,
                shared_out_bias=self.shared_out_bias,
            )
            self.layers.append(nn.ModuleList([time_roformer, band_roformer]))

        self.final_norm = RMSNorm(dim)

        self.register_buffer(
            "freq_indices",
            torch.tensor(np.concatenate(self.band_indices), dtype=torch.long),
            persistent=False,
        )
        # そのビンが何本のバンドに含まれるかを数える
        counts = np.zeros(n_fft // 2 + 1, dtype=np.float32)
        for idx in self.band_indices:
            counts[idx] += 1
        self.register_buffer(
            "num_bands_per_freq",
            torch.tensor(counts, dtype=torch.float32),
            persistent=False,
        )  # shape (F_total,)
        self.mask_estimators = nn.ModuleList(
            [
                MaskEstimator(
                    input_dim=dim,
                    band_indices=self.band_indices,
                    num_channels=num_channels,
                    mlp_expansion_factor=4,
                )
                for _ in range(num_stems)
            ]
        )

    def to_spectrogram(self, inputs: torch.Tensor) -> torch.Tensor:
        # (B, C, T)
        inputs = einops.rearrange(inputs, "b c t -> (b c) t")

        windows: list[torch.Tensor] = [self.hann_window]

        spectrograms = []
        for win in windows:
            spectrogram = torch.stft(
                input=inputs,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.window_size,
                window=win.to(inputs.device),
                center=True,
                return_complex=True,
            )
            spectrogram = einops.rearrange(
                spectrogram, "(b c) f t -> b c f t", c=self.num_channels
            )
            spectrograms.append(spectrogram)  # list of [B, C, F, T]

        spectrogram = torch.concat(spectrograms, dim=1)
        spectrogram = torch.view_as_real(spectrogram)
        spectrogram = einops.rearrange(spectrogram, "b c f t s -> b (c s) t f", s=2)
        return spectrogram, spectrograms[0]

    def mixture_consistency_projection(
        self,
        source_estimates: torch.Tensor,  # [B, C, N, F, T] complex
        mixture: torch.Tensor,  # [B, C, F, T] complex
        weights: torch.Tensor | None = None,  # [B, C, N, F, T] real, 任意
    ) -> torch.Tensor:
        residual = mixture[:, :, None] - source_estimates.sum(dim=2, keepdim=True)
        if weights is None:
            correction = residual / source_estimates.shape[2]
        else:
            weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)
            correction = weights * residual
        return source_estimates + correction

    def to_recon_audio(
        self,
        mask: torch.Tensor,
        original_spec_complex: torch.Tensor,
        original_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        # mask: [B, C, N, F, T]
        source_estimates = original_spec_complex[:, :, None] * mask

        # mixture consistency
        magnitude = source_estimates.abs().clamp_min(1e-8)
        weights = magnitude**2
        separated_spectrogram = self.mixture_consistency_projection(
            source_estimates=source_estimates,
            mixture=original_spec_complex,
            weights=weights,
        )

        separated_spectrogram = einops.rearrange(
            separated_spectrogram, "b c n f t -> (b c n) f t"
        )
        recon_audio = torch.istft(
            separated_spectrogram,
            n_fft=self.window_size,
            hop_length=self.hop_length,
            win_length=self.window_size,
            window=self.hann_window.to(device),
            center=True,
            return_complex=False,
            length=original_length,
        )  # [B*C, T]
        recon_audio = einops.rearrange(
            recon_audio, "(b c n) t -> b c n t", c=self.num_channels, n=self.num_stems
        )
        recon_audio = einops.rearrange(recon_audio, "b c n t -> b n c t")
        return recon_audio

    def forward(self, x):
        # x: (B, C, T)
        if self.use_gradient_checkpoint or self.training:
            checkpoint = torch.utils.checkpoint.checkpoint
        else:
            checkpoint = checkpoint_bypass

        istft_length = x.shape[-1]
        x, original_spec_complex = self.to_spectrogram(x)  # x: [B, C*S, T, F]
        x = einops.rearrange(x, "b c t f -> b t c f")

        # 勾配が流れない問題に対処
        x = checkpoint(
            self.band_split, x, use_reentrant=False
        )  # (B, T, K, hidden_size)

        for time_roformer, band_roformer in self.layers:
            B, T, K, F = x.shape
            # 時間軸Transformer
            x = einops.rearrange(x, "b t k f -> (b k) t f")  # [B*K, T, F]
            x = checkpoint(time_roformer, x, use_reentrant=False)
            x = einops.rearrange(x, "(b k) t f -> b t k f", k=K)  # [B, T, K, F]

            # バンド軸Transformer
            x = x.reshape(B * T, K, F)  # [B*T, K, F]
            x = checkpoint(band_roformer, x, use_reentrant=False)
            x = x.reshape(B, T, K, F)

        x = self.final_norm(x)

        # 音源分離マスク
        mask_list = []
        for mask_estimator in self.mask_estimators:
            pred_mask = checkpoint(
                mask_estimator, x, use_reentrant=False
            )  # [B, T, C*S, F]
            pred_mask = einops.rearrange(
                pred_mask, "b t (c s) f -> b t c f s", s=2
            ).contiguous()
            pred_mask = torch.view_as_complex(pred_mask)  # [B, T, C, F]
            mask_list.append(pred_mask)

        mask_all = torch.stack(mask_list, dim=-2)  # [B, T, C, N, F]
        if self.band_split_type == "mel":
            # 周波数ビンで重なる部分を加算する
            B, T, C, N, F_concat = mask_all.shape
            F_total = self.n_fft // 2 + 1
            mask_sum = torch.zeros(
                (B, T, C, N, F_total), dtype=mask_all.dtype, device=mask_all.device
            )
            freq_idx = self.freq_indices.view(1, 1, 1, 1, -1).expand(
                B, T, C, N, -1
            )  # [B, T, C, N, F_concat]
            mask_sum.scatter_add_(
                dim=-1,
                index=freq_idx,
                src=mask_all,
            )

            # 重なった本数で割って平均
            denom = self.num_bands_per_freq.clamp(min=1e-8)  # [F_total]
            denom = denom.view(1, 1, 1, 1, -1)  # broadcast
            mask_avg = mask_sum / denom  # [B, T, C, N, F_total]
            mask_avg = einops.rearrange(mask_avg, "b t c n f -> b c n f t")
        else:
            mask_avg = einops.rearrange(mask_all, "b t c n f -> b c n f t")

        recon_audio = self.to_recon_audio(
            mask=mask_avg,
            original_spec_complex=original_spec_complex,
            original_length=istft_length,
            dtype=x.dtype,
            device=x.device,
        )
        return recon_audio
