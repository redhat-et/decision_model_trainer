import random
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from time import perf_counter
from typing import Any

from decision_model_trainer.config import (
    DatasetConfig,
    ModelConfig,
    TrainingArgs,
    TrainingConfig,
)
from decision_model_trainer.datasets import DATASET_ADAPTERS

Progress = Callable[[str], None]


@dataclass(frozen=True)
class TrainingResult:
    output_path: Path
    train_items: int
    eval_items: int
    skipped_items: int
    total_items: int
    metrics: dict[str, float]


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
        datasets=[DatasetConfig.model_validate(dataset) for dataset in config.get("datasets", [])],
        output=Path(config.get("output", "outputs")),
        merged_output=config.get("merged_output"),
        device=config.get("device", "auto"),
        seed=config.get("seed", 3407),
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


def load_dataset(
    config: TrainingConfig | DatasetConfig | dict[str, Any], split: str | None = None
):
    from datasets import load_dataset

    if isinstance(config, TrainingConfig):
        if len(config.datasets) != 1:
            raise ValueError("load_dataset requires a single dataset configuration.")
        dataset_config = config.datasets[0]
    elif isinstance(config, DatasetConfig):
        dataset_config = config
    else:
        dataset_config = DatasetConfig.model_validate(config)
    dataset_ref = dataset_config.dataset or dataset_config.dataset_name
    split = dataset_config.train_split if split is None else split

    if not dataset_ref:
        raise ValueError("Config must include `dataset` or `dataset_name`.")

    # Fetch from adapters (if present)
    if dataset_ref in DATASET_ADAPTERS:
        dataset_adapter = DATASET_ADAPTERS[dataset_ref]
        adapter_kwargs = {"split": split}
        if dataset_config.name is not None:
            adapter_kwargs["name"] = dataset_config.name
        return dataset_adapter(**adapter_kwargs).dataset

    dataset_path = Path(dataset_ref)
    if dataset_path.exists():
        return load_dataset(path=str(dataset_path), split=split)

    load_kwargs: dict[str, Any] = {"path": str(dataset_ref), "split": split}
    if dataset_config.name is not None:
        load_kwargs["name"] = dataset_config.name
    return load_dataset(**load_kwargs)


def report_accuracy(
    FastDecisionModel,
    model,
    tokenizer,
    datasets: dict[str, list],
    stage: str,
    progress: Progress | None = None,
) -> dict[str, dict[str, float]]:
    reported: dict[str, dict[str, float]] = {}
    for split, items in datasets.items():
        metrics = FastDecisionModel.evaluate(model, tokenizer, items)
        numeric_metrics = {
            name: float(value)
            for name, value in metrics.items()
            if isinstance(value, Real) and not isinstance(value, bool)
        }
        reported[split] = numeric_metrics
        if progress:
            progress(
                f"{stage} {split} accuracy: {numeric_metrics['accuracy']:.4f}, "
                f"ece: {numeric_metrics['ece']:.4f}"
            )
    return reported


def _split_items(FastDecisionModel, items, dataset_config: DatasetConfig, seed: int):
    """Create missing eval/test splits while preserving explicitly named splits."""
    missing_eval = dataset_config.eval_split is None
    missing_test = dataset_config.test_split is None and dataset_config.test_fraction > 0
    eval_items = None
    test_items = None
    missing_fraction = (
        (dataset_config.eval_fraction if missing_eval else 0)
        + (dataset_config.test_fraction if missing_test else 0)
    )
    if missing_fraction:
        train_items, held_out = FastDecisionModel.split_holdout(
            items, seed=seed, fraction=missing_fraction
        )
        if missing_eval and missing_test:
            eval_ratio = dataset_config.eval_fraction / missing_fraction
            eval_items, test_items = FastDecisionModel.split_holdout(
                held_out, seed=seed, fraction=1 - eval_ratio
            )
        elif missing_eval:
            eval_items = held_out
        else:
            test_items = held_out
    else:
        train_items = items
    return train_items, eval_items, test_items


def _mix_items(item_sets: list[list], weights: list[float | None], seed: int) -> list:
    if len(item_sets) <= 1:
        return item_sets[0] if item_sets else []
    active = [(items, weight) for items, weight in zip(item_sets, weights) if items]
    if not active:
        return []
    values = [items for items, _ in active]
    probabilities = [weight if weight is not None else 1.0 for _, weight in active]
    rng = random.Random(seed)
    positions = [0] * len(values)
    mixed = []
    for _ in range(sum(len(items) for items in values)):
        index = rng.choices(range(len(values)), weights=probabilities, k=1)[0]
        mixed.append(values[index][positions[index] % len(values[index])])
        positions[index] += 1
    return mixed


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
    started_at = perf_counter()
    config = resolve_config(config)
    device = resolve_device(config)
    if progress:
        progress(f"Using device: {device}")

    FastDecisionModel, model, tokenizer = load_model(config)
    train_sets = []
    eval_sets = []
    test_sets = []
    train_reports = []
    for index, dataset_config in enumerate(config.datasets):
        train_dataset = load_dataset(dataset_config, split=dataset_config.train_split)
        items, report = FastDecisionModel.build_dataset(train_dataset, tokenizer, model)
        train_items_for_source, eval_items_for_source, test_items_for_source = _split_items(
            FastDecisionModel, items, dataset_config, config.seed + index
        )
        if dataset_config.eval_split:
            eval_dataset = load_dataset(dataset_config, split=dataset_config.eval_split)
            eval_items_for_source, eval_report = FastDecisionModel.build_dataset(
                eval_dataset, tokenizer, model
            )
            report["skipped"] += eval_report["skipped"]
            report["total"] += eval_report["total"]
        if dataset_config.test_split:
            test_dataset = load_dataset(dataset_config, split=dataset_config.test_split)
            test_items_for_source, test_report = FastDecisionModel.build_dataset(
                test_dataset, tokenizer, model
            )
            report["skipped"] += test_report["skipped"]
            report["total"] += test_report["total"]
        train_sets.append(train_items_for_source)
        if eval_items_for_source is not None:
            eval_sets.append(eval_items_for_source)
        if test_items_for_source is not None:
            test_sets.append(test_items_for_source)
        train_reports.append(report)

    weights = [dataset.weight for dataset in config.datasets]
    train_items = _mix_items(train_sets, weights, config.seed)
    eval_items = [item for items in eval_sets for item in items]
    test_items = [item for items in test_sets for item in items] or None
    datasets = {"train": train_items, "eval": eval_items}
    if test_items is not None:
        datasets["test"] = test_items
    report = {
        "skipped": sum(item["skipped"] for item in train_reports),
        "total": sum(item["total"] for item in train_reports),
    }

    if progress:
        progress(
            f"Prepared {len(train_items)} training and {len(eval_items)} validation items"
        )
        progress(f"Skipped {report['skipped']} of {report['total']} decisions")

    model_config = config.model
    output_path = config.output
    output_path.mkdir(parents=True, exist_ok=True)
    from unsloth import DecisionTrainer

    metrics: dict[str, float] = {
        "dataset_train_items": float(len(train_items)),
        "dataset_eval_items": float(len(eval_items)),
        "dataset_skipped_items": float(report["skipped"]),
        "dataset_total_items": float(report["total"]),
    }
    if test_items is not None:
        metrics["dataset_test_items"] = float(len(test_items))

    before_lora_metrics = report_accuracy(
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
    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    if train_result is not None and hasattr(train_result, "metrics"):
        metrics.update(
            {
                f"training_{name}": float(value)
                for name, value in train_result.metrics.items()
                if isinstance(value, Real) and not isinstance(value, bool)
            }
        )
    after_lora_metrics = report_accuracy(
        FastDecisionModel, model, tokenizer, datasets, "after_lora", progress
    )
    save_model(model, tokenizer, output_path, config)
    tokenizer.save_pretrained(output_path)
    FastDecisionModel.calibrate(model, tokenizer, eval_items)
    after_calibration_metrics = report_accuracy(
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
    for stage, stage_metrics in (
        ("before_lora", before_lora_metrics),
        ("after_lora", after_lora_metrics),
        ("after_calibration", after_calibration_metrics),
    ):
        for split, split_metrics in stage_metrics.items():
            metrics.update(
                {
                    f"{stage}_{split}_{name}": value
                    for name, value in split_metrics.items()
                }
            )
    metrics["training_total_seconds"] = perf_counter() - started_at
    return TrainingResult(
        output_path=output_path,
        train_items=len(train_items),
        eval_items=len(eval_items),
        skipped_items=report["skipped"],
        total_items=report["total"],
        metrics=metrics,
    )
