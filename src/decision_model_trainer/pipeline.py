import argparse
import os
from pathlib import Path

from kfp import compiler, dsl, kubernetes

DEFAULT_IMAGE = os.environ.get(
    "DECISION_MODEL_TRAINER_IMAGE",
    "quay.io/rh_ee_taagarwa/decision-model-trainer:latest",
)


@dsl.container_component
def train_component(
    config_json: str,
    model: dsl.Output[dsl.Model],
    merged_model: dsl.Output[dsl.Model],
):
    return dsl.ContainerSpec(
        image=DEFAULT_IMAGE,
        command=["python", "-m", "decision_model_trainer.kfp_runner"],
        args=["--config-json", config_json, model.path, merged_model.path],
    )


DEFAULT_NODE_SELECTORS = {"gpu-pool-size": "large"}
DEFAULT_TOLERATIONS = [
    {
        "key": "nvidia.com/gpu",
        "operator": "Equal",
        "value": "true",
        "effect": "NoSchedule",
    }
]


def _make_training_pipeline(
    node_selectors: dict[str, str], tolerations: list[dict[str, str]]
):
    @dsl.pipeline(name="decision-model-training")
    def training_pipeline(
        config_json: str,
    ):
        task = train_component(config_json=config_json)
        kubernetes.set_image_pull_policy(task, "Always")
        task.set_accelerator_type("nvidia.com/gpu")
        task.set_accelerator_limit(1)
        for label_key, label_value in node_selectors.items():
            kubernetes.add_node_selector(task, label_key, label_value)
        for toleration in tolerations:
            kubernetes.add_toleration(task, **toleration)
        task.set_caching_options(False)

    return training_pipeline


training_pipeline = _make_training_pipeline(DEFAULT_NODE_SELECTORS, DEFAULT_TOLERATIONS)


def _parse_node_selectors(values: list[str] | None) -> dict[str, str]:
    selectors: dict[str, str] = {}
    for value in values or []:
        key, separator, selector_value = value.partition("=")
        if not separator or not key or not selector_value:
            raise ValueError(f"Invalid node selector {value!r}; use KEY=VALUE.")
        selectors[key] = selector_value
    return selectors


def _parse_tolerations(values: list[str] | None) -> list[dict[str, str]]:
    tolerations: list[dict[str, str]] = []
    for value in values or []:
        parsed: dict[str, str] = {}
        for item in value.split(","):
            key, separator, item_value = item.partition("=")
            if not separator or not key or not item_value:
                raise ValueError(
                    f"Invalid toleration field {item!r}; use comma-separated KEY=VALUE fields."
                )
            parsed[key] = item_value
        required = {"key", "operator", "value", "effect"}
        missing = required - parsed.keys()
        if missing:
            raise ValueError(
                f"Toleration is missing fields: {', '.join(sorted(missing))}."
            )
        tolerations.append({key: parsed[key] for key in required})
    return tolerations


def compile_pipeline(
    output: Path,
    *,
    node_selectors: dict[str, str] | None = None,
    tolerations: list[dict[str, str]] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(
        pipeline_func=_make_training_pipeline(
            DEFAULT_NODE_SELECTORS if node_selectors is None else node_selectors,
            DEFAULT_TOLERATIONS if tolerations is None else tolerations,
        ),
        package_path=str(output),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile the decision-model KFP pipeline."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/decision-model-training.yaml"),
        help="Output YAML path.",
    )
    parser.add_argument(
        "--node-selector",
        action="append",
        metavar="KEY=VALUE",
        help="Kubernetes node selector; repeat for multiple selectors. Defaults to gpu-pool-size=large.",
    )
    parser.add_argument(
        "--toleration",
        action="append",
        metavar="KEY=VALUE,...",
        help=(
            "Kubernetes toleration as comma-separated fields, for example "
            "key=nvidia.com/gpu,operator=Equal,value=true,effect=NoSchedule. "
            "Repeat for multiple tolerations."
        ),
    )
    args = parser.parse_args()
    try:
        node_selectors = (
            _parse_node_selectors(args.node_selector)
            if args.node_selector is not None
            else None
        )
        tolerations = (
            _parse_tolerations(args.toleration) if args.toleration is not None else None
        )
    except ValueError as exc:
        parser.error(str(exc))
    compile_pipeline(
        args.output, node_selectors=node_selectors, tolerations=tolerations
    )
    print(f"Compiled pipeline to {args.output}")


if __name__ == "__main__":
    main()
