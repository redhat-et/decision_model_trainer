import logging
from abc import ABC, abstractmethod

import pandas as pd
from datasets import Dataset

logger = logging.getLogger(__name__)


class DecisionDatasetAdapter(ABC):
    name: str
    """Name of the dataset. Recommend using HuggingFace ID if applicable."""

    def __init__(self, *, split: str | None = None):
        logger.info(f"Initializing {self.name} adapter")
        self.dataset: Dataset = None
        self.load_data(split=split)
        if self.dataset is None:
            raise ValueError(f"Failed to load data for {self.name}")
        logger.info(f"Data loaded: {self.dataset.shape}")

    @abstractmethod
    def load_data(self, *, split: str | None = None) -> pd.DataFrame:
        """Load data into a pandas DataFrame."""
