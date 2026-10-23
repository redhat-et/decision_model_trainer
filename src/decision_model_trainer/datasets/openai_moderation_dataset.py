import json

import pandas as pd
from datasets import Dataset, load_dataset

from decision_model_trainer.datasets.base import DecisionDatasetAdapter


class OpenAIModerationDataset(DecisionDatasetAdapter):
    name = "walledai/openai-moderation-dataset"

    def format_row(self, row: dict):
        question_label_map = {
            "sexual": "S",
            "hate": "H",
            "violence": "V",
            "harassment": "HR",
            "self-harm": "SH",
            "sexual/minors": "S3",
            "hate/threatening": "H2",
            "violence/graphic": "V2",
        }

        state = {"prompt": row["prompt"]}

        questions = {
            "sexual": {
                "type": "noul",
                "instructions": "Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).",
            },
            "hate": {
                "type": "noul",
                "instructions": "Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.",
            },
            "violence": {
                "type": "noul",
                "instructions": "Content that promotes or glorifies violence or celebrates the suffering or humiliation of others.",
            },
            "harassment": {
                "type": "noul",
                "instructions": "Content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur.",
            },
            "self-harm": {
                "type": "noul",
                "instructions": "Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.",
            },
            "sexual/minors": {
                "type": "noul",
                "instructions": "Sexual content that includes an individual who is under 18 years old.",
            },
            "hate/threatening": {
                "type": "noul",
                "instructions": "Hateful content that also includes violence or serious harm towards the targeted group.",
            },
            "violence/graphic": {
                "type": "noul",
                "instructions": "Violent content that depicts death, violence, or serious physical injury in extreme graphic detail.",
            },
        }

        gold = {
            question: {
                "confidence": 0.5 if pd.isna(row[label]) else 1.0,
                "label": "true" if row[label] == 1 else "false",
                "noul": 0.5
                if pd.isna(row[label])
                else (1.0 if row[label] == 1 else 0.0),
                "probabilities": {
                    "false": 0.5
                    if pd.isna(row[label])
                    else (0.0 if row[label] == 1 else 1.0),
                    "true": 0.5
                    if pd.isna(row[label])
                    else (1.0 if row[label] == 1 else 0.0),
                },
                "type": "noul",
            }
            for question, label in question_label_map.items()
        }

        formatted_row = {
            "state": json.dumps(state),
            "questions": json.dumps(questions),
            "gold": json.dumps(gold),
        }

        return formatted_row

    def load_data(self, *, name: str | None = None, split: str | None = None):
        """Load dataset into a pandas DataFrame."""
        load_kwargs = {"split": split or "train"}
        if name is not None:
            load_kwargs["name"] = name
        self.raw_dataset = load_dataset(self.name, **load_kwargs)
        formatted_rows = [self.format_row(row) for row in self.raw_dataset]
        self.dataset = Dataset.from_list(formatted_rows)
