"""Matrix Factorization model package."""

from src.models.matrix_factorization.model import MatrixFactorization
from src.models.matrix_factorization.recommender import MatrixFactorizationRecommender

__all__ = ["MatrixFactorization", "MatrixFactorizationRecommender"]
