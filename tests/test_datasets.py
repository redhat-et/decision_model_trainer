import json

from decision_model_trainer import training
from decision_model_trainer.datasets import (
    OpenAIModerationDataset,
    openai_moderation_dataset,
)


def test_openai_moderation_adapter_formats_rows(monkeypatch):
    raw_rows = [
        {
            "prompt": "A harmful prompt",
            "S": 1,
            "H": 0,
            "V": None,
            "HR": 0,
            "SH": 0,
            "S3": 0,
            "H2": 0,
            "V2": 1,
        }
    ]
    calls = []

    def fake_load_dataset(name, *, split):
        calls.append((name, split))
        return raw_rows

    monkeypatch.setattr(openai_moderation_dataset, "load_dataset", fake_load_dataset)

    dataset = OpenAIModerationDataset(split="validation").dataset

    assert calls == [(OpenAIModerationDataset.name, "validation")]
    assert dataset.column_names == ["state", "questions", "gold"]

    row = dataset[0]
    assert json.loads(row["state"]) == {"prompt": "A harmful prompt"}

    questions = json.loads(row["questions"])
    assert questions["sexual"]["type"] == "noul"
    assert questions["violence/graphic"]["instructions"]

    gold = json.loads(row["gold"])
    assert gold["sexual"] == {
        "confidence": 1.0,
        "label": "true",
        "noul": 1.0,
        "probabilities": {"false": 0.0, "true": 1.0},
        "type": "noul",
    }
    assert gold["violence"] == {
        "confidence": 0.5,
        "label": "false",
        "noul": 0.5,
        "probabilities": {"false": 0.5, "true": 0.5},
        "type": "noul",
    }


def test_openai_moderation_adapter_defaults_to_train(monkeypatch):
    calls = []

    def fake_load_dataset(name, *, split):
        calls.append((name, split))
        return []

    monkeypatch.setattr(openai_moderation_dataset, "load_dataset", fake_load_dataset)

    OpenAIModerationDataset()

    assert calls == [(OpenAIModerationDataset.name, "train")]


def test_openai_moderation_adapter_passes_subset(monkeypatch):
    calls = []

    def fake_load_dataset(path, *, split, **kwargs):
        calls.append((path, kwargs.get("name"), split))
        return []

    monkeypatch.setattr(openai_moderation_dataset, "load_dataset", fake_load_dataset)

    OpenAIModerationDataset(name="custom-subset", split="validation")

    assert calls == [(OpenAIModerationDataset.name, "custom-subset", "validation")]


def test_load_dataset_dispatches_to_registered_adapter(monkeypatch):
    calls = []
    expected_dataset = object()

    class Adapter:
        def __init__(self, *, split):
            calls.append(split)
            self.dataset = expected_dataset

    monkeypatch.setitem(training.DATASET_ADAPTERS, "custom/adapter", Adapter)

    result = training.load_dataset({"dataset": "custom/adapter"}, split="test")

    assert result is expected_dataset
    assert calls == ["test"]


def test_load_dataset_passes_configured_split_to_adapter(monkeypatch):
    calls = []

    class Adapter:
        def __init__(self, *, split):
            calls.append(split)
            self.dataset = "dataset"

    monkeypatch.setitem(training.DATASET_ADAPTERS, "custom/adapter", Adapter)

    result = training.load_dataset(
        {"dataset": "custom/adapter", "dataset_split": "validation"}
    )

    assert result == "dataset"
    assert calls == ["validation"]
