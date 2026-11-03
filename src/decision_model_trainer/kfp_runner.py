from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .training import resolve_config, train_from_config


def load_json_config(value: str) -> dict[str, Any]:
    try:
        config = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON configuration: {exc.msg}.") from exc
    if not isinstance(config, dict):
        raise TypeError("The JSON configuration root must be an object.")
    return config


def write_resolved_config(config: dict[str, Any], output_path: str) -> None:
    resolved = resolve_config(config)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(resolved.model_dump_json(indent=2) + "\n", encoding="utf-8")


def write_metrics(metrics: dict[str, float], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run decision-model training in a KFP container."
    )
    parser.add_argument(
        "--config-json", required=True, help="Training configuration as a JSON object."
    )
    parser.add_argument(
        "model_path",
        help="KFP-provided path for the model artifact.",
    )
    parser.add_argument(
        "merged_model_path",
        help="KFP-provided path for the merged model artifact.",
    )
    parser.add_argument(
        "resolved_config_path",
        help="KFP-provided path for the resolved configuration artifact.",
    )
    parser.add_argument(
        "metrics_path",
        help="KFP-provided path for the metrics artifact.",
    )
    args = parser.parse_args()
    try:
        config = load_json_config(args.config_json)
    except ValueError as exc:
        parser.error(str(exc))
    config["output"] = args.model_path
    config["merged_output"] = args.merged_model_path
    try:
        write_resolved_config(config, args.resolved_config_path)
        resolved_config = resolve_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    result = train_from_config(resolved_config, progress=print)
    write_metrics(result.metrics, args.metrics_path)


if __name__ == "__main__":
    main()
