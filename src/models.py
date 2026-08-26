from __future__ import annotations

import torch
from torch import nn


class CNNFeatureExtractor(nn.Module):
    """Small 1D CNN that extracts local temporal patterns."""

    def __init__(self, in_channels: int = 9, channels: int = 64, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 128, 9] -> [B, 9, 128] -> [B, C, 32]
        return self.net(x.transpose(1, 2))


class CNNBaseline(nn.Module):
    """1D CNN baseline for six-class activity recognition."""

    def __init__(
        self,
        in_channels: int = 9,
        cnn_channels: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.3,
        num_classes: int = 6,
    ) -> None:
        super().__init__()
        self.features = CNNFeatureExtractor(in_channels, cnn_channels, kernel_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(cnn_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class AttentionBlock(nn.Module):
    """Self-attention plus feed-forward residual block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: int,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(attention_dropout),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(attention_dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attention_out, weights = self.attention(
            x,
            x,
            x,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = self.norm1(x + attention_out)
        x = self.norm2(x + self.ffn(x))
        return x, weights if return_attention else None


class CNNMultiHeadAttention(nn.Module):
    """1D CNN followed by one or more Multi-Head Self-Attention blocks."""

    def __init__(
        self,
        in_channels: int = 9,
        cnn_channels: int = 64,
        kernel_size: int = 5,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_attention_layers: int = 1,
        attention_dropout: float = 0.2,
        classifier_dropout: float = 0.3,
        feedforward_dim: int = 256,
        num_classes: int = 6,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.features = CNNFeatureExtractor(in_channels, cnn_channels, kernel_size)
        self.projection = nn.Linear(cnn_channels, embed_dim)
        self.attention_blocks = nn.ModuleList(
            [
                AttentionBlock(
                    embed_dim,
                    num_heads,
                    feedforward_dim,
                    attention_dropout,
                )
                for _ in range(num_attention_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def encode(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # CNN: [B, 128, 9] -> [B, C, 32]
        x = self.features(x)
        # MHA representation: [B, C, 32] -> [B, 32, C] -> [B, 32, E]
        x = x.transpose(1, 2)
        x = self.projection(x)

        attentions: list[torch.Tensor] = []
        for block in self.attention_blocks:
            x, weights = block(x, return_attention=return_attention)
            if weights is not None:
                attentions.append(weights)
        return x, attentions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.encode(x, return_attention=False)
        pooled = sequence.mean(dim=1)
        return self.classifier(pooled)

    @torch.no_grad()
    def attention_weights(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return per-head attention maps for exploratory analysis."""
        _, attentions = self.encode(x, return_attention=True)
        return attentions
