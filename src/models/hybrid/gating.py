"""Adaptive Gating Network for Multi-Branch Hybrid Recommendation (P1 & P2).

Computes dynamic importance weights:
    w_c(z_u, z_i): Collaborative Filtering branch weight
    w_s(z_u):      Sequential (SASRec) branch weight
    w_t(z_i):      Content / Item branch weight
Normalized via Softmax over available branches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGatingNetwork(nn.Module):
    """Mini-MLP gating network dynamically weighting CF, Sequential, and Content signals."""

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 32,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Gating sub-networks for individual branch logits
        # 1. CF gate: depends on interaction between user vector and item vector
        self.gate_cf = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

        # 2. Sequential gate: depends on current user dynamic sequential state
        self.gate_seq = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

        # 3. Content gate: depends on item embedding / topic characteristics
        self.gate_content = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights with Xavier uniform."""
        for module in (self.gate_cf, self.gate_seq, self.gate_content):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(
        self,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
        branch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute branch weights for candidate items.

        Args:
            user_repr: Tensor of shape (1, D) or (B, D) representing user sequential/CF state.
            item_repr: Tensor of shape (N, D) representing candidate item embeddings.
            branch_mask: Optional boolean Tensor of shape (3,) indicating active branches
                         [CF, SEQ, CONTENT]. False means branch is disabled (ablation).

        Returns:
            weights: Tensor of shape (N, 3) where each row sums to 1.0.
        """
        num_candidates = item_repr.shape[0]

        # Expand user representation to align with candidate items
        if user_repr.ndim == 1:
            user_repr = user_repr.unsqueeze(0)
        u_expanded = user_repr.expand(num_candidates, -1)  # (N, D)

        # 1. Logit for CF branch: w_c(z_u, z_i)
        cf_input = torch.cat([u_expanded, item_repr], dim=-1)  # (N, 2D)
        logit_cf = self.gate_cf(cf_input)  # (N, 1)

        # 2. Logit for Sequential branch: w_s(z_u)
        logit_seq = self.gate_seq(u_expanded)  # (N, 1)

        # 3. Logit for Content branch: w_t(z_i)
        logit_content = self.gate_content(item_repr)  # (N, 1)

        # Concatenate logits: shape (N, 3)
        logits = torch.cat([logit_cf, logit_seq, logit_content], dim=-1)

        # Mask disabled branches (Ablation study)
        if branch_mask is not None:
            mask = branch_mask.to(logits.device)
            logits = logits.masked_fill(~mask.unsqueeze(0), -1e9)

        # Softmax normalization across branches
        weights = F.softmax(logits, dim=-1)  # (N, 3)
        return weights
