from .base import DecisionDatasetAdapter
from .openai_moderation_dataset import OpenAIModerationDataset

DATASET_ADAPTERS: dict[str, type[DecisionDatasetAdapter]] = {
    OpenAIModerationDataset.name: OpenAIModerationDataset,
}

__all__ = [
    "OpenAIModerationDataset",
]
