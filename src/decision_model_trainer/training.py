from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_model_trainer.config import ModelConfig, TrainingArgs, TrainingConfig
from decision_model_trainer.datasets import DATASET_ADAPTERS

Progress = Callable[[str], None]


@dataclass(frozen=True)
class TrainingResult:
    output_path: Path
    train_items: int
    eval_items: int
    skipped_items: int
    total_items: int


def load_yaml(path: Path) -> TrainingConfig:
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("Install the yaml dependency with `uv sync` first.") from exc
    if not path.is_file():
        raise ValueError(f"Config file does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise TypeError("The YAML root must be a mapping.")
    return TrainingConfig.model_validate(config)


def resolve_config(config: TrainingConfig | dict[str, Any]) -> TrainingConfig:
    return config if isinstance(config, TrainingConfig) else TrainingConfig.model_validate(config)


def _resolve_helper_config(config: TrainingConfig | dict[str, Any]) -> TrainingConfig:
    """Keep standalone helpers compatible with their intentionally partial inputs."""
    if isinstance(config, TrainingConfig):
        return config
    return TrainingConfig.model_construct(
        model=ModelConfig.model_validate(config.get("model", {})),
        dataset=config.get("dataset"),
        dataset_name=config.get("dataset_name"),
        dataset_config=config.get("dataset_config"),
        dataset_split=config.get("dataset_split", "train"),
        test_split=config.get("test_split"),
        output=Path(config.get("output", "outputs")),
        merged_output=config.get("merged_output"),
        device=config.get("device", "auto"),
        seed=config.get("seed", 3407),
        holdout_fraction=config.get("holdout_fraction", 0.2),
        merged_save_method=config.get("merged_save_method", "merged_16bit"),
        resume_from_checkpoint=config.get("resume_from_checkpoint"),
        training=TrainingArgs.model_validate(config.get("training", {})),
    )


def resolve_device(config: TrainingConfig | dict[str, Any]) -> str:
    config = _resolve_helper_config(config)
    requested = config.device
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError as exc:
        raise ValueError(
            "PyTorch is not installed. Install the `cuda` extra or set up the NVIDIA environment."
        ) from exc

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_dataset(config: TrainingConfig | dict[str, Any], split: str | None = None):
    from datasets import load_dataset

    config = _resolve_helper_config(config)
    dataset_ref = config.dataset or config.dataset_name
    split = config.dataset_split if split is None else split

    if config.dataset and config.dataset_name:
        raise ValueError("Set only one of `dataset` or `dataset_name`.")
    dataset_config = config.dataset_config
    if not dataset_ref:
        raise ValueError("Config must include `dataset` or `dataset_name`.")

    # Fetch from adapters (if present)
    if dataset_ref in DATASET_ADAPTERS:
        dataset_adapter = DATASET_ADAPTERS[dataset_ref]
        adapter_kwargs = {"split": split}
        if dataset_config is not None:
            adapter_kwargs["name"] = dataset_config
        return dataset_adapter(**adapter_kwargs).dataset

    dataset_path = Path(dataset_ref)
    if dataset_path.exists():
        return load_dataset(path=str(dataset_path), split=split)

    load_kwargs: dict[str, Any] = {"path": str(dataset_ref), "split": split}
    if dataset_config is not None:
        load_kwargs["name"] = dataset_config
    return load_dataset(**load_kwargs)


def report_accuracy(
    FastDecisionModel,
    model,
    tokenizer,
    datasets: dict[str, list],
    stage: str,
    progress: Progress | None = None,
) -> dict[str, float]:
    accuracies: dict[str, float] = {}
    for split, items in datasets.items():
        metrics = FastDecisionModel.evaluate(model, tokenizer, items)
        accuracy = float(metrics["accuracy"])
        ece = float(metrics["ece"])
        accuracies[split] = accuracy
        if progress:
            progress(f"{stage} {split} accuracy: {accuracy:.4f}, ece: {ece:.4f}")
    return accuracies


def load_model(config: TrainingConfig | dict[str, Any]):
    try:
        from unsloth import FastDecisionModel
    except ImportError as exc:
        raise ValueError(
            "Unsloth is not installed. Follow the NVIDIA installation instructions in README.md."
        ) from exc
    model_config = _resolve_helper_config(config).model
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=model_config.name,
        max_seq_length=model_config.max_seq_length,
        load_in_4bit=model_config.load_in_4bit,
    )
    return FastDecisionModel, model, tokenizer


def build_training_args(config: TrainingConfig | dict[str, Any], output_path: Path):
    from transformers import TrainingArguments

    config = _resolve_helper_config(config)
    values = config.training_values(output_path)
    try:
        from unsloth import is_bfloat16_supported

        supports_bf16 = is_bfloat16_supported()
    except ImportError:
        supports_bf16 = False
    values.setdefault("bf16", supports_bf16)
    values.setdefault("fp16", not supports_bf16)
    return TrainingArguments(**values)


def save_model(
    model, tokenizer, output_path: Path, config: TrainingConfig | dict[str, Any]
) -> None:
    config = _resolve_helper_config(config)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_path)
    else:
        model.save_pretrained_merged(
            output_path, tokenizer, save_method=config.merged_save_method
        )


def train_from_config(
    config: TrainingConfig | dict[str, Any], progress: Progress | None = None
) -> TrainingResult:
    """Run dataset preparation, fine-tuning, calibration, and model saving."""
    config = resolve_config(config)
    device = resolve_device(config)
    if progress:
        progress(f"Using device: {device}")

    FastDecisionModel, model, tokenizer = load_model(config)
    dataset = load_dataset(config)
    items, report = FastDecisionModel.build_dataset(dataset, tokenizer, model)
    train_items, eval_items = FastDecisionModel.split_holdout(
        items,
        seed=config.seed,
        fraction=config.holdout_fraction,
    )
    test_items = None
    test_split = config.test_split
    if test_split:
        test_dataset = load_dataset(config, split=test_split)
        test_items, test_report = FastDecisionModel.build_dataset(
            test_dataset, tokenizer, model
        )
        if progress:
            progress(
                f"Prepared {len(test_items)} test items "
                f"(skipped {test_report['skipped']} of {test_report['total']})"
            )

    datasets = {"train": train_items, "eval": eval_items}
    if test_items is not None:
        datasets["test"] = test_items

    if progress:
        progress(
            f"Prepared {len(train_items)} training and {len(eval_items)} validation items"
        )
        progress(f"Skipped {report['skipped']} of {report['total']} decisions")

    model_config = config.model
    output_path = config.output
    output_path.mkdir(parents=True, exist_ok=True)
    from unsloth import DecisionTrainer

    report_accuracy(
        FastDecisionModel, model, tokenizer, datasets, "before_lora", progress
    )

    model = FastDecisionModel.get_peft_model(
        model,
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        use_gradient_checkpointing=model_config.gradient_checkpointing,
        random_state=config.seed,
    )

    trainer = DecisionTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_items,
        eval_dataset=eval_items,
        args=build_training_args(config, output_path),
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    report_accuracy(
        FastDecisionModel, model, tokenizer, datasets, "after_lora", progress
    )
    save_model(model, tokenizer, output_path, config)
    tokenizer.save_pretrained(output_path)
    FastDecisionModel.calibrate(model, tokenizer, eval_items)
    report_accuracy(
        FastDecisionModel, model, tokenizer, datasets, "after_calibration", progress
    )
    save_model(model, tokenizer, output_path, config)
    merged_output = config.merged_output
    if merged_output:
        merged_output_path = Path(merged_output)
        merged_output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(
            merged_output_path,
            tokenizer,
            save_method=config.merged_save_method,
        )
        if progress:
            progress(f"Saved merged model to {merged_output_path}")
    if progress:
        progress(f"Saved model to {output_path}")
    return TrainingResult(
        output_path=output_path,
        train_items=len(train_items),
        eval_items=len(eval_items),
        skipped_items=report["skipped"],
        total_items=report["total"],
    )
