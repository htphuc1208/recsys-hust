"""Self-Attentive Sequential Recommendation (SASRec) Architecture.

Reference:
    Kang, W. C., & McAuley, J. (2018). Self-attentive sequential recommendation.
    In IEEE International Conference on Data Mining (ICDM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointwiseFeedForward(nn.Module):
    """Two-layer feedforward network with GELU activation and dropout."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.conv1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.dropout(F.gelu(self.conv1(x))))


class SelfAttentionBlock(nn.Module):
    """Transformer block with multi-head self-attention and causal masking."""

    def __init__(self, hidden_dim: int, num_heads: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        self.attention_layernorm = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.feedforward_layernorm = nn.LayerNorm(hidden_dim, eps=1e-8)
        self.feedforward = PointwiseFeedForward(hidden_dim, dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-LN Self-Attention with residual
        norm_x = self.attention_layernorm(x)
        attn_output, _ = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_output)

        # Pre-LN Feedforward with residual
        norm_x = self.feedforward_layernorm(x)
        ff_output = self.feedforward(norm_x)
        return x + self.dropout(ff_output)


class SASRec(nn.Module):
    """Self-Attentive Sequential Recommender model.

    Encodes an ordered sequence of user interaction history into a single
    contextual user representation vector using causal self-attention.
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        max_seq_len: int = 50,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.padding_idx = num_items  # Item IDs are 0 .. num_items - 1; num_items is PAD

        self.item_embedding = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=embedding_dim,
            padding_idx=self.padding_idx,
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=embedding_dim,
        )
        self.emb_dropout = nn.Dropout(p=dropout_rate)
        self.emb_layernorm = nn.LayerNorm(embedding_dim, eps=1e-8)

        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    hidden_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layernorm = nn.LayerNorm(embedding_dim, eps=1e-8)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize embeddings and layer weights."""
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[self.padding_idx].fill_(0.0)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate upper-triangular causal attention mask to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def encode_sequence(self, input_seqs: torch.Tensor) -> torch.Tensor:
        """Encode an input sequence of item IDs into the last-step user state vector.

        Args:
            input_seqs: Tensor of shape (batch_size, seq_len) with item IDs in [0, num_items]
                        (where num_items is padding_idx).

        Returns:
            user_vectors: Tensor of shape (batch_size, embedding_dim).
        """
        batch_size, seq_len = input_seqs.shape
        device = input_seqs.device

        # Lookup item embeddings
        seq_embs = self.item_embedding(input_seqs)  # (batch_size, seq_len, embedding_dim)

        # Lookup positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embs = self.position_embedding(positions)

        x = self.emb_layernorm(self.emb_dropout(seq_embs + pos_embs))

        # Masks
        causal_mask = self._generate_causal_mask(seq_len, device)
        key_padding_mask = input_seqs == self.padding_idx  # (batch_size, seq_len)

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        x = self.final_layernorm(x)

        # Extract sequence representation at last position
        # If all tokens are padding, return zeros
        non_pad_mask = input_seqs != self.padding_idx
        # For right-padded sequences or standard sequences, use the last non-padded index
        # If left-padded, the last position is non-pad
        last_indices = non_pad_mask.sum(dim=1) - 1
        last_indices = torch.clamp(last_indices, min=0)

        batch_indices = torch.arange(batch_size, device=device)
        user_repr = x[batch_indices, last_indices]  # (batch_size, embedding_dim)

        # Zero out completely empty sequences
        has_history = non_pad_mask.any(dim=1, keepdim=True)
        user_repr = user_repr * has_history.float()
        return user_repr

    def forward(
        self,
        input_seqs: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute logits for positive and negative target items given history sequences.

        Args:
            input_seqs: Tensor of shape (batch_size, seq_len)
            pos_items: Tensor of shape (batch_size,)
            neg_items: Tensor of shape (batch_size,)

        Returns:
            pos_logits: Tensor of shape (batch_size,)
            neg_logits: Tensor of shape (batch_size,)
        """
        user_repr = self.encode_sequence(input_seqs)  # (batch_size, embedding_dim)

        pos_embs = self.item_embedding(pos_items)  # (batch_size, embedding_dim)
        neg_embs = self.item_embedding(neg_items)  # (batch_size, embedding_dim)

        pos_logits = (user_repr * pos_embs).sum(dim=-1)
        neg_logits = (user_repr * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict_candidates(
        self,
        input_seqs: torch.Tensor,
        candidate_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute recommendation scores for candidate items given interaction history sequences.

        Args:
            input_seqs: Tensor of shape (1, seq_len) or (batch_size, seq_len)
            candidate_item_ids: Tensor of shape (num_candidates,) or (batch_size, num_candidates)

        Returns:
            scores: Tensor of shape (batch_size, num_candidates)
        """
        user_repr = self.encode_sequence(input_seqs)  # (B, D)
        if candidate_item_ids.ndim == 1:
            cand_embs = self.item_embedding(candidate_item_ids)  # (N, D)
            scores = torch.matmul(user_repr, cand_embs.t())  # (B, N)
        else:
            cand_embs = self.item_embedding(candidate_item_ids)  # (B, N, D)
            scores = (user_repr.unsqueeze(1) * cand_embs).sum(dim=-1)  # (B, N)
        return scores
