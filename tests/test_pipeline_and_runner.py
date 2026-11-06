import json
from unittest.mock import Mock

import pytest

from decision_model_trainer import kfp_runner, pipeline


def test_parse_node_selectors():
    assert pipeline._parse_node_selectors(["gpu=large", "zone=west"])
    assert pipeline._parse_node_selectors(["gpu=large", "zone=west"]) == {
        "gpu": "large",
        "zone": "west",
    }


@pytest.mark.parametrize("value", [["gpu"], ["=large"], ["gpu="]])
def test_parse_node_selectors_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="Invalid node selector"):
        pipeline._parse_node_selectors(value)


def test_parse_tolerations_requires_all_fields():
    value = "key=nvidia.com/gpu,operator=Equal,value=true,effect=NoSchedule"

    assert pipeline._parse_tolerations([value]) == [
        {
            "key": "nvidia.com/gpu",
            "operator": "Equal",
            "value": "true",
            "effect": "NoSchedule",
        }
    ]

    with pytest.raises(ValueError, match="missing fields"):
        pipeline._parse_tolerations(["key=gpu,operator=Equal"])


def test_compile_pipeline_creates_parent_and_passes_options(tmp_path, monkeypatch):
    compiler_instance = Mock()
    compiler_class = Mock(return_value=compiler_instance)
    monkeypatch.setattr(pipeline.compiler, "Compiler", compiler_class)
    output = tmp_path / "nested" / "pipeline.yaml"

    pipeline.compile_pipeline(output, node_selectors={"pool": "gpu"}, tolerations=[])

    assert output.parent.is_dir()
    compiler_instance.compile.assert_called_once()
    assert compiler_instance.compile.call_args.kwargs["package_path"] == str(output)


def test_load_json_config_requires_object():
    assert kfp_runner.load_json_config('{"seed": 42}') == {"seed": 42}

    with pytest.raises(ValueError, match="Invalid JSON"):
        kfp_runner.load_json_config("not-json")
    with pytest.raises(TypeError, match="must be an object"):
        kfp_runner.load_json_config(json.dumps(["not", "an", "object"]))


def test_write_resolved_config_writes_defaults_and_kfp_paths(tmp_path):
    config = {
        "model": {},
        "datasets": [{"dataset_name": "org/dataset", "name": "default"}],
        "output": "/tmp/model",
        "merged_output": "/tmp/merged-model",
    }
    output = tmp_path / "nested" / "resolved-config.json"

    kfp_runner.write_resolved_config(config, str(output))

    resolved = json.loads(output.read_text(encoding="utf-8"))
    assert resolved["seed"] == 3407
    assert resolved["training"]["max_steps"] == 60
    assert resolved["output"] == "/tmp/model"
    assert resolved["merged_output"] == "/tmp/merged-model"


def test_train_component_declares_metrics_output():
    outputs = pipeline.train_component.component_spec.outputs

    assert outputs["metrics"].type == "system.Metrics@0.0.1"


def test_write_metrics_writes_numeric_metrics(tmp_path):
    output = tmp_path / "nested" / "metrics.json"

    kfp_runner.write_metrics(
        {"after_calibration_test_accuracy": 0.91, "training_total_seconds": 12.5},
        str(output),
    )

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "after_calibration_test_accuracy": 0.91,
        "training_total_seconds": 12.5,
    }
