"""
FT-Transformer architecture, factored out of notebooks/04b_neural_network_benchmark.ipynb
for reuse by the serving process. Mirrors the notebook's classes exactly -- must, since
this loads the exact weights that notebook trained (models/ft_transformer.pt).
"""

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """Per-feature linear projection for numeric features + embedding lookup for the
    categorical feature -- each input feature becomes its own d_token-dim token, following
    Gorishniy et al. 2021 (arXiv:2106.11959)."""

    def __init__(self, n_numeric: int, n_categories: int, d_token: int):
        super().__init__()
        self.numeric_weight = nn.Parameter(torch.empty(n_numeric, d_token))
        self.numeric_bias = nn.Parameter(torch.empty(n_numeric, d_token))
        self.cat_embedding = nn.Embedding(n_categories, d_token)
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        num_tokens = x_num.unsqueeze(-1) * self.numeric_weight + self.numeric_bias
        cat_tokens = self.cat_embedding(x_cat).unsqueeze(1)
        cls_tokens = self.cls.expand(x_num.size(0), -1, -1)
        return torch.cat([cls_tokens, num_tokens, cat_tokens], dim=1)


class FTTransformer(nn.Module):
    def __init__(self, n_numeric: int, n_categories: int, d_token: int, n_blocks: int,
                 n_heads: int, dropout: float, ffn_mult: int = 2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_numeric, n_categories, d_token)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * ffn_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token), nn.Linear(d_token, d_token // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_token // 2, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x_num, x_cat)
        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0]
        return self.head(cls_out).squeeze(-1)
