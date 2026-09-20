from typing import Any

from src.models.base import BaseRecommender
from src.models.baselines.itemknn import ItemKNNRecommender
from src.models.baselines.popularity import PopularityRecommender
from src.models.baselines.random import RandomRecommender
from src.models.bpr import BPRRecommender
from src.models.matrix_factorization import MatrixFactorizationRecommender

MODEL_REGISTRY: dict[str, type[BaseRecommender]] = {
    "popularity": PopularityRecommender,
    "mostpopular": PopularityRecommender,
    "random": RandomRecommender,
    "itemknn": ItemKNNRecommender,
    "mf": MatrixFactorizationRecommender,
    "matrix_factorization": MatrixFactorizationRecommender,
    "bpr": BPRRecommender,
}


def register_model(name: str):
    """Decorator to register a new model class."""
    def decorator(cls: type[BaseRecommender]):
        MODEL_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_model(name: str, **kwargs: Any) -> BaseRecommender:
    """Factory function to instantiate models from config."""
    name_lower = name.lower()
    if name_lower not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' not found in registry. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name_lower](**kwargs)
