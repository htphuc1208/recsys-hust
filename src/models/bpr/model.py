from src.models.matrix_factorization.model import MatrixFactorization


class BPR(MatrixFactorization):
    """Mô hình BPR-MF dùng embedding người dùng và bài báo.

    Kế thừa kiến trúc MatrixFactorization nhưng mặc định tắt user_bias
    do user_bias tự triệt tiêu trong công thức ranking cặp (pairwise).
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        use_item_bias: bool = True,
    ) -> None:
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            use_user_bias=False,
            use_item_bias=use_item_bias,
        )
