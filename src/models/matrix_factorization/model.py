import torch
from torch import nn


class MatrixFactorization(nn.Module):
    """Mô hình phân rã ma trận cho dữ liệu người dùng - bài báo."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        use_bias: bool = True,
    ) -> None:
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.use_bias = use_bias

        # Vector biểu diễn của người dùng.
        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim,
        )

        # Vector biểu diễn của bài báo.
        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim,
        )

        if self.use_bias:
            # Độ lệch riêng của từng người dùng.
            self.user_bias = nn.Embedding(
                num_users,
                1,
            )

            # Độ lệch riêng của từng bài báo.
            self.item_bias = nn.Embedding(
                num_items,
                1,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Khởi tạo tham số ban đầu của mô hình."""
        nn.init.normal_(
            self.user_embedding.weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.normal_(
            self.item_embedding.weight,
            mean=0.0,
            std=0.01,
        )

        if self.use_bias:
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Tính điểm phù hợp giữa người dùng và bài báo."""
        user_vector = self.user_embedding(user_ids)
        item_vector = self.item_embedding(item_ids)

        # Tích vô hướng giữa vector người dùng và bài báo.
        score = (user_vector * item_vector).sum(dim=1)

        if self.use_bias:
            user_bias = self.user_bias(user_ids).squeeze(1)
            item_bias = self.item_bias(item_ids).squeeze(1)
            score = score + user_bias + item_bias

        return score
