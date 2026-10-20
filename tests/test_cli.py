from types import SimpleNamespace

import pytest

from decision_model_trainer import cli


def test_load_training_config_converts_validation_error_to_bad_parameter(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        cli, "load_yaml", lambda path: (_ for _ in ()).throw(ValueError("bad"))
    )

    with pytest.raises(cli.typer.BadParameter, match="bad"):
        cli._load_training_config(tmp_path / "config.yaml")


def test_load_saved_model_loads_model_and_switches_to_inference(monkeypatch, tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    calls = {}

    class FastDecisionModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["load"] = (args, kwargs)
            return "model", "tokenizer"

        @staticmethod
        def for_inference(model):
            calls["inference"] = model

    monkeypatch.setitem(
        __import__("sys").modules,
        "unsloth",
        SimpleNamespace(FastDecisionModel=FastDecisionModel),
    )

    assert cli._load_saved_model(model_path) == (
        FastDecisionModel,
        "model",
        "tokenizer",
    )
    assert calls == {
        "load": ((str(model_path),), {"max_seq_length": 2048, "load_in_4bit": False}),
        "inference": "model",
    }


def test_load_saved_model_rejects_missing_path(tmp_path):
    with pytest.raises(cli.typer.BadParameter, match="does not exist"):
        cli._load_saved_model(tmp_path / "missing")
