# Decision Model Trainer

Decision Model Trainer is a Kubeflow Pipelines (KFP) component for Red Hat OpenShift AI (RHOAI) that fine-tunes language models to make structured decisions from text. The pipeline accepts a JSON training configuration, runs GPU-backed preparation and training in a container, and publishes adapter and optional merged-model artifacts to the pipeline artifact store. A local CLI provides the same training utility for development and debugging, along with inference and Hugging Face publishing commands.

## Features

- Accepts local datasets or Hugging Face datasets and configurable dataset splits.
- Supports typed `choice`, `noul`, and `score` questions with structured gold answers.
- Provides reproducible holdout splitting, skipped-record reporting, evaluation, checkpointing, and decision-head calibration.
- Supports configurable base models, sequence length, 4-bit loading, LoRA settings, gradient checkpointing, and standard Transformers training arguments.
- Automatically selects CUDA, Apple MPS, or CPU, with CUDA-oriented settings for RHOAI GPU workloads.
- Provides `train`, `infer`, and `push` CLI commands for local development, validation, and optional merged 16-bit Hub publishing.
- Includes a Kubeflow Pipelines v2 definition for RHOAI, with GPU scheduling controls, disabled caching, and separate adapter and merged-model artifacts.

## Training Pipeline Overview

The training pipeline finetunes a LORA on an existing LLM for structured decisions. Finetuning follows the process outlined by [Unsloth's Decision Model training Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)-Decision.ipynb):
1. Load the existing, pretrained LLM (e.g. `unsloth/Qwen3.5-4B`)
2. Attach a LORA for finetuning
3. Fetch a [training dataset from Huggingface](https://huggingface.co/datasets?other=structured-decisions) or use a local dataset file and format it for calibrated decisions
4. Split the training data into a train set and a holdout set for calibration
5. Report train/eval accuracy before adding the LoRA (and optional test-split accuracy)
6. Train the LoRA on the train set and report accuracy again
7. Calibrate on the eval set and report final accuracy

### Training Datasets Schema

Training rows are expected to contain:

- `state`: the input text
- `questions`: a JSON object (or object-valued column) containing `choice`, `noul`, or `score` questions
- `gold`: the expected answers

## Installation

The CLI is installable on macOS and Linux. Training can use CUDA, Apple MPS, or CPU according to the configured or automatically detected device. CUDA remains the recommended backend for larger RHOAI runs; use the appropriate PyTorch and Unsloth installation for the host.

```bash
# Any Mac or Linux machine, including Apple Silicon/MPS
uv sync

# NVIDIA machine, after selecting the appropriate PyTorch CUDA index for your system
uv sync --extra cuda
```

Unsloth's official installation guidance should be followed if its CUDA/PyTorch wheel combination differs from the versions in your environment: <https://unsloth.ai/docs/get-started/install>.

## Configuration

Copy `examples/config.yaml` and update the dataset and output paths. See the [configuration reference](docs/configuration.md) for all available options.

```bash
decision-model-trainer train --config examples/config.yaml
```

## RHOAI AI Pipelines

The project also includes a Kubeflow Pipelines v2 definition for Red Hat OpenShift AI.
The pipeline runs the same training utility as the CLI inside the published training image;
it does not include inference or Hub publishing steps.

Install the pipeline compiler extra on a machine with Python 3.10 or newer:

```bash
uv sync --extra pipeline
DECISION_MODEL_TRAINER_IMAGE=quay.io/rh_ee_taagarwa/decision-model-trainer:latest \
  uv run python -m decision_model_trainer.pipeline \
  --output dist/decision-model-training.yaml
```

Scheduling defaults to `gpu-pool-size=large` and the
`nvidia.com/gpu=true:NoSchedule` toleration. Override them at compile time:

```bash
uv run decision-model-trainer-pipeline \
  --node-selector gpu-pool-size=large \
  --node-selector accelerator=nvidia \
  --toleration key=nvidia.com/gpu,operator=Equal,value=true,effect=NoSchedule \
  --output dist/decision-model-training.yaml
```

Upload `dist/decision-model-training.yaml` to the RHOAI Data Science Pipelines
server as described in the RHOAI documentation. Set the `config_json` pipeline
parameter to the training configuration as a JSON object. See the
[configuration reference](docs/configuration.md) for the available fields and
JSON conversion instructions. The component exposes the adapter model and a
merged 16-bit model as separate typed KFP `Model` artifacts; the pipeline
runtime manages both output directories and uploads them to the configured
artifact store when the task succeeds. Dataset paths must be provided by
storage mounted into the pipeline task, such as a shared PVC.
The component requests one `nvidia.com/gpu` accelerator and is
constrained to nodes with `gpu-pool-size=large`. It also tolerates the
`nvidia.com/gpu=true:NoSchedule` taint. Change those values in
`src/decision_model_trainer/pipeline.py` if the RHOAI cluster uses different
node labels or taints.

For an OpenShift job, follow the logs with:

```bash
oc logs -f job/decision-model-train
```

The bar may appear as multiple updates in collected logs because log aggregators
do not always interpret terminal carriage returns; the step and metric messages
remain available through `logging_steps`.

## Hugging Face and inference

Set `HF_TOKEN` or pass `--token`:

```bash
decision-model-trainer push --model outputs/qwen-decision --repo your-user/qwen-decision
decision-model-trainer push --model outputs/qwen-decision --repo your-user/qwen-decision --merge
decision-model-trainer infer --model outputs/qwen-decision --state "Refund invoice 4411" --questions examples/questions.json
```

The `--merge` option uploads a merged 16-bit model and may require substantially more disk memory than adapter upload.

## Utility Scripts

### Downloading object storage data

Use `scripts/download_s3_folder.py` to copy all objects below an S3 prefix to a local directory. Credentials are loaded using the standard boto3 environment, profile, or instance-role mechanisms. The endpoint option works with AWS, MinIO, Ceph, and other S3-compatible services:

```bash
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
  uv run python scripts/download_s3_folder.py \
  s3://my-bucket/training-data/ ./downloads/training-data \
  --endpoint-url https://object.example.com --region us-east-1
```

The source prefix is treated as a folder, so `s3://my-bucket/training-data`
downloads keys beginning with `training-data/`. Existing local files are
overwritten. For a public bucket, add `--no-sign-request`.

## Development

### Releases and container images

The project follows [Semantic Versioning](https://semver.org/). The version in
`pyproject.toml` is the source of truth. Use a patch release for fixes, a minor
release for backwards-compatible features, and a major release for breaking
changes.

Bump and tag the version from the project root, then push the release commit:

```bash
# Choose exactly one bump type: patch, minor, or major
uv version --bump patch
uv lock
VERSION=$(uv version --short)
git add pyproject.toml uv.lock
git commit -m "Release v${VERSION}"
git tag -a "v${VERSION}" -m "Release v${VERSION}"
git push origin main --follow-tags
```

Pushing a `vX.Y.Z` tag runs the GitHub Actions release workflow. It publishes
the image to the repository's public GHCR package with these tags:

```text
ghcr.io/OWNER/REPOSITORY:VERSION
ghcr.io/OWNER/REPOSITORY:latest
```

The version tag is immutable by convention and should be used for production
deployments. The `latest` tag is a convenience alias.
The first publication may require setting the package visibility to Public in
the repository's GitHub Packages settings.

## Known Issues

- [ ] MLflow logging is bugged in RHOAI. Jobs will not properly log metrics to MLflow when running in RHOAI
