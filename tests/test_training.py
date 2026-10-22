from types import SimpleNamespace

import pytest

from decision_model_trainer import training


def test_load_yaml_reads_mapping(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset: example\nseed: 42\n", encoding="utf-8")

    assert training.load_yaml(config_path) == {"dataset": "example", "seed": 42}


def test_load_yaml_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        training.load_yaml(tmp_path / "missing.yaml")


def test_load_yaml_rejects_non_mapping(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- item\n", encoding="utf-8")

    with pytest.raises(TypeError, match="must be a mapping"):
        training.load_yaml(config_path)


def test_resolve_device_honors_explicit_value():
    assert training.resolve_device({"device": "cpu"}) == "cpu"


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_resolve_device_auto_detects_backend(monkeypatch, cuda, mps, expected):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    assert training.resolve_device({}) == expected


def test_load_dataset_uses_local_path(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()

    class DatasetModule:
        @staticmethod
        def load_dataset(**kwargs):
            return kwargs

    monkeypatch.setitem(__import__("sys").modules, "datasets", DatasetModule)

    result = training.load_dataset(
        {"dataset": str(dataset_path), "dataset_split": "test"}
    )

    assert result == {"path": str(dataset_path), "split": "test"}


def test_load_dataset_can_override_configured_split(monkeypatch):
    calls = []

    class DatasetModule:
        @staticmethod
        def load_dataset(**kwargs):
            calls.append(kwargs)
            return kwargs

    monkeypatch.setitem(__import__("sys").modules, "datasets", DatasetModule)

    result = training.load_dataset(
        {"dataset": "org/data", "dataset_split": "train"}, split="test"
    )

    assert result == {"path": "org/data", "split": "test"}
    assert calls == [{"path": "org/data", "split": "test"}]


def test_report_accuracy_reports_each_dataset():
    calls = []

    class FastDecisionModel:
        @staticmethod
        def evaluate(model, tokenizer, items):
            calls.append((model, tokenizer, items))
            return {"accuracy": len(items) / 10, "ece": len(items) / 100}

    messages = []
    result = training.report_accuracy(
        FastDecisionModel,
        "model",
        "tokenizer",
        {"train": [1, 2], "eval": [3], "test": [4, 5, 6]},
        "after_lora",
        messages.append,
    )

    assert result == {"train": 0.2, "eval": 0.1, "test": 0.3}
    assert calls == [
        ("model", "tokenizer", [1, 2]),
        ("model", "tokenizer", [3]),
        ("model", "tokenizer", [4, 5, 6]),
    ]
    assert messages == [
        "after_lora train accuracy: 0.2000, ece: 0.0200",
        "after_lora eval accuracy: 0.1000, ece: 0.0100",
        "after_lora test accuracy: 0.3000, ece: 0.0300",
    ]


def test_load_dataset_rejects_conflicting_references(monkeypatch):
    class DatasetModule:
        load_dataset = staticmethod(lambda **kwargs: kwargs)

    monkeypatch.setitem(__import__("sys").modules, "datasets", DatasetModule)

    with pytest.raises(ValueError, match="only one"):
        training.load_dataset({"dataset": "one", "dataset_name": "two"})


def test_load_model_passes_model_configuration(monkeypatch):
    calls = {}

    class FastDecisionModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.update(kwargs)
            return "model", "tokenizer"

    monkeypatch.setitem(
        __import__("sys").modules,
        "unsloth",
        SimpleNamespace(FastDecisionModel=FastDecisionModel),
    )

    result = training.load_model(
        {"model": {"name": "test/model", "max_seq_length": 512, "load_in_4bit": False}}
    )

    assert result == (FastDecisionModel, "model", "tokenizer")
    assert calls == {
        "model_name": "test/model",
        "max_seq_length": 512,
        "load_in_4bit": False,
    }
