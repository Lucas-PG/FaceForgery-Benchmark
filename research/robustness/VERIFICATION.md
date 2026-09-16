# Software verification ledger

Verification is version-specific. This file distinguishes executable implementation from empirical research results.

## Recorded checks

At commit `1665f8b84c0c5e8e4143c60a8cae6e9c788ff279`, the isolated CPU workflow completed **46 synthetic tests in 6.02 seconds**, including a small one-epoch synthetic train → save → load → evaluate path. Python compilation and CLI help also passed.

Run: https://github.com/Lucas-PG/FaceForgery-Benchmark/actions/runs/35162138135

Subsequent compatibility/artifact changes require their own passing run; the final PR description records that run separately. A previous passing count is not an assertion that later edits were already tested.

## What these checks cover

Explicit label conventions, exact prediction population alignment, manifest/file integrity, grouped bootstrap behavior, calibration-only-on-validation guards, one-class AUC handling, video label consistency, per-generator real-reference requirements, augmentation reproducibility, finite pilot forwards/backwards, matched auxiliary parameter counts, teacher gradient isolation, safe path handling, missing-image failure and checkpoint lifecycle behavior.

Additional regression tests cover historical versus new model initialization semantics, checkpoint-bound score orientation, exported-artifact analysis, legacy Fourier-input parity and stricter dataset failure behavior. Their observed pass/fail status is reported by the final CI run, not assumed here.

## Not established by CI

No full MFFI/Celeb-DF/DF40 inference, GPU training matrix, modern detector reproduction, trained-checkpoint accuracy, formal robustness, faithful real-image attribution, demographic effect, causal mechanism, runtime on the institution's GPU or publication acceptance is claimed. Pretrained initialization downloads are disabled in CPU checks; smoke training uses small generated tensors and images only.

The preserved six-page paper and presentation were already compiled in earlier work. This revision preserves their bytes rather than claiming to remeasure their scientific results. The import record identifies that earlier history.
