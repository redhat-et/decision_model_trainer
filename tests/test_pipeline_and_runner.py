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
