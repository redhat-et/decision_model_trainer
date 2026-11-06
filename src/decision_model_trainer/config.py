from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    name: str = "unsloth/Qwen3.5-4B"
    max_seq_length: int = 2048
    load_in_4bit: bool = False
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0
    gradient_checkpointing: str | bool = "unsloth"


class TrainingArgs(BaseModel):
    model_config = ConfigDict(extra="allow")

    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    max_steps: int = 60
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    eval_strategy: str = "epoch"
    logging_steps: int = 10
    disable_tqdm: bool = False
    report_to: str | list[str] = "none"
    save_strategy: str = "epoch"


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(min_length=1)
    name: str | None = None
    train_split: str = "train"
    eval_split: str | None = None
    test_split: str | None = None
    eval_fraction: float = 0.2
    test_fraction: float = 0.0
    weight: float | None = None

    @model_validator(mode="after")
    def validate_dataset_options(self) -> DatasetConfig:
        if self.eval_fraction < 0 or self.test_fraction < 0:
            raise ValueError("Dataset split fractions cannot be negative.")
        if self.eval_fraction + self.test_fraction >= 1:
            raise ValueError("Dataset split fractions must sum to less than 1.")
        if self.weight is not None and self.weight <= 0:
            raise ValueError("Dataset weight must be greater than zero.")
        return self


class TrainingConfig(BaseModel):
    """Validated training configuration with all application defaults resolved."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    datasets: list[DatasetConfig]
    output: Path = Path("outputs")
    merged_output: Path | None = None
    device: str = "auto"
    seed: int = 3407
    merged_save_method: str = "merged_16bit"
    resume_from_checkpoint: str | None = None
    training: TrainingArgs = Field(default_factory=TrainingArgs)

    @model_validator(mode="after")
    def validate_datasets(self) -> TrainingConfig:
        if not self.datasets:
            raise ValueError("At least one dataset is required.")
        return self

    def training_values(self, output_path: Path) -> dict[str, Any]:
        values = self.training.model_dump()
        values["seed"] = self.seed
        values["output_dir"] = str(output_path)
        return values
