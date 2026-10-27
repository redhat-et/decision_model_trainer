# Configuration Reference

The same configuration fields are used by the CLI YAML file and the KFP
`config_json` parameter. Start with `examples/config.yaml` and change the
dataset, model, and output settings for your run.

## Top-Level Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | string | none | Local dataset path, or a dataset identifier if it is not a local path. |
| `dataset_name` | string | none | Hugging Face dataset identifier. Set either `dataset` or `dataset_name`, not both. |
| `dataset_config` | string | none | Hugging Face dataset configuration passed as `name`. |
| `dataset_split` | string | `train` | Dataset split to load. |
| `test_split` | string | none | Optional named split from the same dataset source to use for test accuracy. |
| `output` | path | `outputs` | Directory for the adapter/LoRA model. |
| `merged_output` | path | none | Optional directory for the merged model. KFP replaces this with its managed artifact directory. |
| `merged_save_method` | string | `merged_16bit` | Unsloth merge format, used when `merged_output` is set. |
| `device` | string | `auto` | `cuda`, `mps`, `cpu`, or `auto`. Auto selects CUDA, then Apple MPS, then CPU. |
| `seed` | integer | `3407` | Random seed used for model setup and the holdout split. |
| `holdout_fraction` | number | `0.2` | Fraction of prepared items reserved for evaluation. |
| `resume_from_checkpoint` | string or null | none | Checkpoint passed to the trainer for resuming. |

`dataset` may refer to a local JSON, JSONL, or Parquet file/dataset directory.
If the path does not exist locally, it is passed to Hugging Face Datasets as
the dataset name. `dataset` and `dataset_name` are mutually exclusive. When
`test_split` is set, that split is loaded from the same local path or Hub
dataset. Without it, only train and eval accuracy are reported.

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
