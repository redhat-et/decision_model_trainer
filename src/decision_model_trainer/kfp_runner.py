from __future__ import annotations

import argparse
import json
from typing import Any

from .training import train_from_config


def load_json_config(value: str) -> dict[str, Any]:
    try:
        config = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON configuration: {exc.msg}.") from exc
    if not isinstance(config, dict):
        raise TypeError("The JSON configuration root must be an object.")
    return config


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
    args = parser.parse_args()
    try:
        config = load_json_config(args.config_json)
    except ValueError as exc:
        parser.error(str(exc))
    config["output"] = args.model_path
    config["merged_output"] = args.merged_model_path
    train_from_config(config, progress=print)


if __name__ == "__main__":
    main()
