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
