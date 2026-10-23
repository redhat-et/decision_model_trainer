from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_model_trainer.datasets import DATASET_ADAPTERS

Progress = Callable[[str], None]


@dataclass(frozen=True)
class TrainingResult:
    output_path: Path
    train_items: int
    eval_items: int
    skipped_items: int
    total_items: int


def load_yaml(path: Path) -> dict[str, Any]:
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
    return config


def resolve_device(config: dict[str, Any]) -> str:
    requested = config.get("device", "auto")
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


def load_dataset(config: dict[str, Any], split: str | None = None):
    from datasets import load_dataset

    dataset_ref = config.get("dataset") or config.get("dataset_name")
    split = config.get("dataset_split", "train") if split is None else split

    # Fetch from adapters (if present)
    if dataset_ref in DATASET_ADAPTERS:
        dataset_adapter = DATASET_ADAPTERS[dataset_ref]
        return dataset_adapter(split=split).dataset

    if config.get("dataset") and config.get("dataset_name"):
        raise ValueError("Set only one of `dataset` or `dataset_name`.")
    dataset_config = config.get("dataset_config")
    if not dataset_ref:
        raise ValueError("Config must include `dataset` or `dataset_name`.")

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


def load_model(config: dict[str, Any]):
    try:
        from unsloth import FastDecisionModel
    except ImportError as exc:
        raise ValueError(
            "Unsloth is not installed. Follow the NVIDIA installation instructions in README.md."
        ) from exc
    model_config = config.get("model", {})
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=model_config.get("name", "unsloth/Qwen3.5-4B"),
        max_seq_length=model_config.get("max_seq_length", 2048),
        load_in_4bit=model_config.get("load_in_4bit", True),
    )
    return FastDecisionModel, model, tokenizer


def build_training_args(config: dict[str, Any], output_path: Path):
    from transformers import TrainingArguments

    values = {
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 10,
        "max_steps": 60,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "eval_strategy": "epoch",
        "logging_steps": 10,
        "disable_tqdm": False,
        "report_to": "none",
        "seed": config.get("seed", 3407),
        "output_dir": str(output_path),
        "save_strategy": "epoch",
    }
    values.update(config.get("training", {}))
    values["output_dir"] = str(output_path)
    try:
        from unsloth import is_bfloat16_supported

        supports_bf16 = is_bfloat16_supported()
    except ImportError:
        supports_bf16 = False
    values.setdefault("bf16", supports_bf16)
    values.setdefault("fp16", not supports_bf16)
    return TrainingArguments(**values)


def train_from_config(
    config: dict[str, Any], progress: Progress | None = None
) -> TrainingResult:
    """Run dataset preparation, fine-tuning, calibration, and model saving."""
    device = resolve_device(config)
    if progress:
        progress(f"Using device: {device}")

    FastDecisionModel, model, tokenizer = load_model(config)
    dataset = load_dataset(config)
    items, report = FastDecisionModel.build_dataset(dataset, tokenizer, model)
    train_items, eval_items = FastDecisionModel.split_holdout(
        items,
        seed=config.get("seed", 3407),
        fraction=config.get("holdout_fraction", 0.2),
    )
    test_items = None
    test_split = config.get("test_split")
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

    model_config = config.get("model", {})
    output_path = Path(config.get("output", "outputs"))
    output_path.mkdir(parents=True, exist_ok=True)
    from unsloth import DecisionTrainer

    report_accuracy(
        FastDecisionModel, model, tokenizer, datasets, "before_lora", progress
    )

    model = FastDecisionModel.get_peft_model(
        model,
        r=model_config.get("lora_r", 16),
        lora_alpha=model_config.get("lora_alpha", 16),
        lora_dropout=model_config.get("lora_dropout", 0),
        use_gradient_checkpointing=model_config.get(
            "gradient_checkpointing", "unsloth"
        ),
        random_state=config.get("seed", 3407),
    )

    trainer = DecisionTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_items,
        eval_dataset=eval_items,
        args=build_training_args(config, output_path),
    )
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    report_accuracy(
        FastDecisionModel, model, tokenizer, datasets, "after_lora", progress
    )
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    FastDecisionModel.calibrate(model, tokenizer, eval_items)
    report_accuracy(
        FastDecisionModel, model, tokenizer, datasets, "after_calibration", progress
    )
    model.save_pretrained(output_path)
    merged_output = config.get("merged_output")
    if merged_output:
        merged_output_path = Path(merged_output)
        merged_output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(
            merged_output_path,
            tokenizer,
            save_method=config.get("merged_save_method", "merged_16bit"),
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
