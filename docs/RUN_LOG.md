# Run log

## Phase 8 prediction regeneration

| Model | Date | Reason | Previous metric | Final metric |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 2026-10-04 | Regenerated its saved validation/test prediction archive so Phase 8 could evaluate every model from `results/predictions/`. | Test accuracy 0.680000; macro F1 0.469907 | Test accuracy 0.680000; macro F1 0.469907 |
| TF-IDF + LinearSVC | 2026-10-04 | Regenerated its saved validation/test prediction archive so Phase 8 could evaluate every model from `results/predictions/`. | Test accuracy 0.737143; macro F1 0.635154 | Test accuracy 0.737143; macro F1 0.635154 |
| LDA + Logistic Regression | 2026-10-04 | Regenerated its saved validation/test prediction archive so Phase 8 could evaluate every model from `results/predictions/`. | Test accuracy 0.546429; macro F1 0.321414 | Test accuracy 0.546429; macro F1 0.321414 |
| Doc2Vec + Logistic Regression | 2026-10-04 | Regenerated its saved validation/test prediction archive because Phase 8 required saved predictions for every model. Doc2Vec training is not bit-exact across runs. | Test accuracy 0.472857; macro F1 0.322425 | Test accuracy 0.474286; macro F1 0.321791 |

The final values in this table are copied from the final `results/metrics.csv`.
The earlier Doc2Vec values came from the prior Phase 6 run. Doc2Vec used one
worker, but gensim/BLAS execution and inference can still make repeated runs
non-bit-exact; the final saved prediction archive is the Phase 8 provenance.
No test tuning was performed during regeneration.

## Logistic-loss model update

On 2026-10-04, the original saga model was retained as
`tfidf_lr_saga_underconverged` and a new `SGDClassifier(loss="log_loss")` was
tuned on validation over `alpha={1e-6,1e-5,1e-4}`. The selected alpha and
convergence evidence are in `results/converged_lr_selection.json`.

| Model | Reason | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---|---:|---:|---:|
| `tfidf_lr_saga_underconverged` | Retained historical result for comparison | 0.579994 | 0.680000 | 0.469907 |
| `tfidf_logistic_regression` | Replace the badly under-converged final LR baseline | 0.712154 | 0.732143 | 0.628559 |

The new training completed in under the 20-minute time box and reported
convergence (`n_iter_ < max_iter`). The test split was predicted once for the
selected new model.
