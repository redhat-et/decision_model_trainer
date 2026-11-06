# Configuration Reference

The same configuration fields are used by the CLI YAML file and the KFP
`config_json` parameter. Start with `examples/config.yaml` and change the
dataset, model, and output settings for your run.

## Top-Level Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `datasets` | list | required | Dataset sources to mix for training. |
| `output` | path | `outputs` | Directory for the adapter/LoRA model. |
| `merged_output` | path | none | Optional directory for the merged model. KFP replaces this with its managed artifact directory. |
| `merged_save_method` | string | `merged_16bit` | Unsloth merge format, used when `merged_output` is set. |
| `device` | string | `auto` | `cuda`, `mps`, `cpu`, or `auto`. Auto selects CUDA, then Apple MPS, then CPU. |
| `seed` | integer | `3407` | Random seed used for model setup and the holdout split. |
| `resume_from_checkpoint` | string or null | none | Checkpoint passed to the trainer for resuming. |

## Dataset Options

`datasets` is a non-empty list. Each entry describes one source dataset.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | string | required | Local dataset directory, or a Hugging Face dataset identifier when it is not a local path. Local paths take precedence. |
| `name` | string or null | none | Hugging Face dataset subset name. |
| `train_split` | string | `train` | Named split used for training and, when needed, fractional holdouts. |
| `eval_split` | string or null | none | Named evaluation split. If omitted, `eval_fraction` is taken from `train_split`. |
| `test_split` | string or null | none | Named test split. If omitted, `test_fraction` is taken from `train_split`. |
| `eval_fraction` | number | `0.2` | Fraction of the training split reserved for evaluation when `eval_split` is not set. |
| `test_fraction` | number | `0.0` | Fraction of the training split reserved for testing when `test_split` is not set. |
| `weight` | positive number or null | none | Relative share of this source in the training mix. If all weights are omitted, sources are mixed equally. |

Fractions must be non-negative and their sum must be less than `1`. Named
evaluation and test splits are loaded independently and are not included in
fractional holdout calculations. Dataset adapters are applied to every named
or fractional split when an adapter is registered for the dataset source.
Local datasets must be provided as directories containing supported dataset
files; individual local files are not supported.

### Dataset Examples

Use named splits when the source already provides evaluation and test data:

```yaml
datasets:
  - dataset: org/support-decisions
    name: default
    train_split: train
    eval_split: validation
    test_split: test
```

Use fractions to split a source's training data when named splits are not
available:

```yaml
datasets:
  - dataset: data/first.jsonl
    eval_fraction: 0.1
    test_fraction: 0.1
```

Combine sources with a weighted training mix. The weights are relative; they
do not need to add up to `1`:

```yaml
datasets:
  - dataset: org/primary-decisions
    weight: 3
  - dataset: org/supplemental-decisions
    weight: 1
```

## Model Options

Place model settings under `model`:

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | string | `unsloth/Qwen3.5-4B` | Base model identifier or local model path. |
| `max_seq_length` | integer | `2048` | Maximum sequence length. |
| `load_in_4bit` | boolean | `false` | Load the base model in 4-bit mode. Only applicable for Unsloth-provided models. |
| `lora_r` | integer | `16` | LoRA rank. |
| `lora_alpha` | integer | `16` | LoRA scaling factor. |
| `lora_dropout` | number | `0` | LoRA dropout. |
| `gradient_checkpointing` | string or boolean | `unsloth` | Gradient checkpointing mode passed to Unsloth. |

## Training Options

Place Hugging Face `TrainingArguments` under `training`. The following defaults
are set by the application and can be overridden:

| Option | Default |
| --- | --- |
| `per_device_train_batch_size` | `8` |
| `gradient_accumulation_steps` | `4` |
| `warmup_steps` | `10` |
| `max_steps` | `60` |
| `learning_rate` | `0.0002` |
| `lr_scheduler_type` | `cosine` |
| `weight_decay` | `0.01` |
| `eval_strategy` | `epoch` |
| `logging_steps` | `10` |
| `disable_tqdm` | `false` |
| `report_to` | `none` |
| `save_strategy` | `epoch` |

Other valid `transformers.TrainingArguments` fields can also be supplied under
`training`. The application always sets `output_dir` to the configured `output`
directory. `bf16` and `fp16` default according to hardware support unless
explicitly configured.

## Outputs

For a CLI run, set both paths when you want both model forms:

```yaml
output: outputs/qwen-decision
merged_output: outputs/qwen-decision-merged
```

The `output` directory contains the adapter model and tokenizer. The
`merged_output` directory contains the merged 16-bit model and tokenizer. The
merged export requires additional disk space and may require more GPU memory.

For KFP, do not set output paths to PVC locations. The pipeline injects separate
artifact directories for `model` and `merged_model`, and uploads them to the
configured artifact store after the task succeeds.

## KFP JSON

The KFP parameter is a JSON object with the same fields as this YAML file. For
example, convert the example YAML with:

```bash
yq -o=json -I=0 examples/config.yaml
```

Pass the resulting JSON as the pipeline's `config_json` parameter. Dataset paths
must be available inside the pipeline task, such as through a mounted PVC.
