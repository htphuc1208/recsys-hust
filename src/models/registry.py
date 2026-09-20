from typing import Any, Dict, Type
from src.models.base import BaseRecommender
from src.models.baselines.popularity import PopularityRecommender

MODEL_REGISTRY: Dict[str, Type[BaseRecommender]] = {
    "popularity": PopularityRecommender,
}


def register_model(name: str):
    """Decorator to register a new model class."""
    def decorator(cls: Type[BaseRecommender]):
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
