from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from .training import load_yaml, train_from_config

app = typer.Typer(help="Train and use Unsloth decision models.")


@app.command()
def train(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, readable=True)],
) -> None:
    """Train a decision model using a YAML configuration file."""
    try:
        train_from_config(_load_training_config(config), progress=typer.echo)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _load_training_config(path: Path) -> dict[str, Any]:
    try:
        return load_yaml(path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _load_saved_model(model_path: Path):
    if not model_path.exists():
        raise typer.BadParameter(f"Model path does not exist: {model_path}")
    try:
        from unsloth import FastDecisionModel
    except ImportError as exc:
        raise typer.BadParameter(
            "Install Unsloth before loading a decision model."
        ) from exc
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(model_path),
        max_seq_length=2048,
        load_in_4bit=False,
    )
    FastDecisionModel.for_inference(model)
    return FastDecisionModel, model, tokenizer


@app.command()
def infer(
    model: Annotated[Path, typer.Option("--model", exists=True, file_okay=False)],
    state: Annotated[str, typer.Option("--state", help="Text to classify.")],
    questions: Annotated[
        Path,
        typer.Option(
            "--questions", exists=True, readable=True, help="JSON question object."
        ),
    ],
) -> None:
    """Run one decision locally and print JSON results."""
    FastDecisionModel, loaded_model, tokenizer = _load_saved_model(model)
    with questions.open(encoding="utf-8") as file:
        question_data = json.load(file)
    result = FastDecisionModel.predict(loaded_model, tokenizer, state, question_data)
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command("push")
def push(
    model: Annotated[Path, typer.Option("--model", exists=True, file_okay=False)],
    repo: Annotated[
        str, typer.Option("--repo", help="Hub repository, for example org/model.")
    ],
    token: Annotated[
        str | None, typer.Option("--token", envvar="HF_TOKEN", hide_input=True)
    ] = None,
    merge: Annotated[
        bool,
        typer.Option(
            "--merge", help="Upload a merged 16-bit model instead of adapters."
        ),
    ] = False,
) -> None:
    """Upload a trained model to the Hugging Face Hub."""
    if not token:
        raise typer.BadParameter("Pass --token or set HF_TOKEN.")
    _, loaded_model, tokenizer = _load_saved_model(model)
    if merge:
        loaded_model.push_to_hub_merged(
            repo, tokenizer, save_method="merged_16bit", token=token
        )
    else:
        loaded_model.push_to_hub(repo, token=token)
        tokenizer.push_to_hub(repo, token=token)
    typer.echo(f"Uploaded model to https://huggingface.co/{repo}")


def main() -> None:
    app()
