# Training Metrics

During training, the CLI and pipeline print one line for each dataset split at
several stages. A typical line looks like this:

```text
after_calibration eval accuracy: 0.8400, ece: 0.0340
```

## Stages

- `before_lora`: evaluation before the LoRA adapters have been trained. This is a baseline for the newly attached decision head.
- `after_lora`: evaluation after fine-tuning, before probability calibration.
- `after_calibration`: evaluation after temperature calibration has been fitted on the holdout evaluation split.

Calibration does not retrain the model; it adjusts the probabilities produced
by the decision head.

## Dataset Splits

- `train`: items used for LoRA training. These metrics help diagnose underfitting or overfitting, but are not an unbiased estimate of generalization.
- `eval`: rows held out from LoRA training. This split is used to fit the calibration temperatures, so its post-calibration ECE can be optimistic.
- `test`: an optional independent dataset split configured with `test_split`. When available, this is the best split for estimating performance on unseen data and checking whether calibration generalizes.

## Accuracy

`accuracy` is the fraction of individual decisions answered correctly. A row can contain several questions, so this is question-level accuracy rather than the fraction of complete rows for which every question was correct.

The value ranges from `0.0` to `1.0`; for example, `0.84` means 84% of the decisions in that split were correct.

## Expected Calibration Error

`ece` means **Expected Calibration Error**. It measures whether predicted confidence agrees with observed correctness:

```text
ECE = weighted average of |average confidence - accuracy|
```

The calculation groups decisions by their maximum predicted probability. If predictions with roughly 90% confidence are correct about 90% of the time, that confidence range is well calibrated.

ECE ranges from `0.0` upward, with lower being better. An ECE of `0.03` means that confidence and accuracy differ by roughly three percentage points on average across the confidence groups. ECE does not measure accuracy itself: the model can be accurate but overconfident, or less accurate but well calibrated.

## Reading Results

1. Compare `after_lora` accuracy with `before_lora` to check whether fine-tuning improved decision accuracy.
2. Compare `after_calibration` ECE with the earlier ECE to check whether the confidence values became better aligned with correctness.
3. Prefer `test` results over `train` or `eval` results when judging expected performance on new data.
4. Treat a low ECE on a small split cautiously. A small number of decisions or narrow confidence ranges can make ECE noisy.

Temperature calibration usually changes probabilities and ECE, but not the selected answer, because the highest logit generally remains the highest logit after scaling. Therefore, accuracy may stay the same while ECE improves.
