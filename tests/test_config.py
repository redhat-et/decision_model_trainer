from pathlib import Path

import pytest
from pydantic import ValidationError

from decision_model_trainer.config import TrainingConfig


def valid_config() -> dict:
    return {
        "model": {},
        "datasets": [{"dataset": "org/dataset", "name": "default"}],
    }


def test_config_resolves_defaults():
    config = TrainingConfig.model_validate(valid_config())

    assert config.output == Path("outputs")
    assert config.device == "auto"
    assert config.seed == 3407
    assert config.datasets[0].eval_fraction == 0.2
    assert config.model.name == "unsloth/Qwen3.5-4B"
    assert config.model.max_seq_length == 2048
    assert config.training.max_steps == 60
    assert config.training.learning_rate == 2e-4


def test_config_requires_model_and_datasets():
    for field in ("model", "datasets"):
        config = valid_config()
        config.pop(field)

        with pytest.raises(ValidationError):
            TrainingConfig.model_validate(config)


def test_config_requires_dataset_reference():
    config = valid_config()
    config["datasets"][0].pop("dataset")

    with pytest.raises(ValidationError, match="dataset"):
        TrainingConfig.model_validate(config)


def test_config_allows_arbitrary_training_arguments():
    config = valid_config()
    config["training"] = {
        "num_train_epochs": 2,
        "optim": "adamw_torch",
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }

    resolved = TrainingConfig.model_validate(config)

    assert resolved.training.num_train_epochs == 2
    assert resolved.training.optim == "adamw_torch"
    assert resolved.training.gradient_checkpointing_kwargs == {"use_reentrant": False}


def test_config_serializes_to_json():
    config = TrainingConfig.model_validate(valid_config())

    serialized = config.model_dump_json()

    assert '"dataset":"org/dataset"' in serialized
    assert '"output":"outputs"' in serialized


def test_config_supports_multiple_weighted_datasets_and_named_splits():
    config = TrainingConfig.model_validate(
        {
            "model": {},
            "datasets": [
                {"dataset": "org/one", "name": "first", "weight": 2},
                {
                    "dataset": "local/data",
                    "train_split": "training",
                    "eval_split": "validation",
                    "test_split": "testing",
                },
            ],
        }
    )

    assert config.datasets[0].weight == 2
    assert config.datasets[1].train_split == "training"
    assert config.datasets[1].eval_split == "validation"
