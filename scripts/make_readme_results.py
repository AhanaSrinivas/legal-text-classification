"""Regenerate the number-bearing README sections from results/.

Each section lives between
    <!-- BEGIN GENERATED: name -->  and  <!-- END GENERATED: name -->
in README.md. Everything else in the README is hand-written prose.
Run: python scripts/make_readme_results.py [--check]
--check exits non-zero if README.md is out of date instead of rewriting it.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.report_facts import facts, render, results_rows


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def results_block():
    lines = [
        "Official LexGLUE test split (used once per model) and validation split. "
        "Macro F1 is the primary metric.",
        "",
        "| Model | Test macro F1 | 95% CI (macro F1) | Test accuracy | Test weighted F1 "
        "| Val macro F1 | Val accuracy | Val weighted F1 |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in results_rows():
        lines.append(
            f"| {row['name']} | {row['test_f1']} | {row['ci']} | {row['test_acc']} | "
            f"{row['test_wf1']} | {row['val_f1']} | {row['val_acc']} | {row['val_wf1']} |"
        )
    lines += [
        "",
        render(
            "CIs are percentile bootstrap intervals over {{ci_resamples}} resamples of the test set "
            "(seed {{ci_seed}}). They are unpaired and capture test-set sampling only, not "
            "training randomness. The LinearSVC and Legal-BERT intervals {{svc_bert_ci_overlap}}."
        ),
        "",
        render(
            "**Legal-BERT run-to-run variation.** Three full runs with identical settings gave test "
            "macro F1 {{bert_first_run_f1}} (first run; its files were overwritten, value recorded "
            "in `docs/RUN_LOG.md`), {{bert_reported_f1_4}} (the reported run: "
            "`reports/colab_run_log.ipynb`, predictions in `results/transformer_predictions.npz`) "
            "and {{bert_run2_f1_4}} (rerun: `results/transformer_run2_info.json`). We report the "
            "{{bert_reported_f1_4}} run and treat the range {{bert_run_min_4}}–{{bert_run_max_4}} "
            "as run-to-run variation; no best-run selection was made. The demo weights come from "
            "the rerun."
        ),
        "",
        render(
            "**Per-class highlights (test).** Legal-BERT is strong on Criminal Procedure "
            "(F1 {{bert_criminal_procedure_f1}}), Federal Taxation ({{bert_federal_taxation_f1}}) "
            "and First Amendment ({{bert_first_amendment_f1}}), but scores F1 = 0 on "
            "{{bert_zero_f1}} and recalls only {{bert_federalism_rpct}}% of Federalism and "
            "{{bert_interstate_relations_rpct}}% of Interstate Relations cases. LinearSVC's weakest "
            "classes are Miscellaneous (F1 {{svc_miscellaneous_f1}}, test support "
            "{{test_support_miscellaneous}}) and Federalism ({{svc_federalism_f1}}). Most frequent "
            "confusion (count in brackets): {{ea_svc_top_pair}} for LinearSVC and "
            "{{ea_bert_top_pair}} for Legal-BERT."
        ),
        "",
        "Figures: `results/figures/model_comparison.png`, `results/figures/confusion_matrix_normalized_*.png`; "
        "per-class scores with support: `results/per_class_f1.csv`; error analysis: `results/error_analysis.md`.",
    ]
    return "\n".join(lines)


def dataset_block():
    return render("""\
- **Source:** LexGLUE benchmark, `{{dataset_name}}`, config `{{dataset_config}}` on the Hugging Face Hub, loaded with `datasets` {{datasets_version}}. Labels are SCDB issue areas (Supreme Court Database, Spaeth et al.).
- **Size and split (official):** {{n_total}} opinions: train {{n_train}}, validation {{n_val}}, test {{n_test}}. LexGLUE splits SCOTUS chronologically: train {{train_years}}, validation {{val_years}}, test {{test_years}} (Chalkidis et al., 2022, Section 3).
- **Columns:** `text` (full opinion) and `label` ({{n_classes}} classes). SCDB defines one more issue area (Private Action) that has no cases in this corpus.
- **Quality checks:** {{null_texts}} empty texts; {{internal_dups}} exact duplicates within splits and {{cross_split_dups}} across splits (SHA-256 of each text).
- **Imbalance:** largest training class {{largest_class}} ({{largest_count}} documents), smallest {{smallest_class}} ({{smallest_count}}), ratio {{imbalance_ratio}}:1. Rarest test classes: {{rare_desc}}.
- **Length:** median {{words_median_train}} words (train), {{words_median_val}} (validation), {{words_median_test}} (test). On a {{tok_sample}}-document train sample, the Legal-BERT tokenizer gives a median of {{tok_median_exact}} tokens, {{tok_pct_over_512}}% of documents exceed 512 tokens, and the first 512 tokens cover about {{tok_pct_covered}}% of a document on average.

Sources: `results/dataset_info.json`, `results/eda_stats.json`.""")


def models_block():
    return render("""\
| Model | Features | Classifier and selected setting | Selection evidence |
|---|---|---|---|
| Majority baseline | none | always predicts the training majority class ({{majority_class}}) | `results/majority_baseline.json` |
| TF-IDF + LinearSVC | TF-IDF, unigrams + bigrams, {{tfidf_max_features}} features | `LinearSVC` (L2, squared hinge), C = {{svc_c}} chosen from {{svc_c_grid}} | `results/classical_selection.json` |
| TF-IDF + LR (final) | same TF-IDF | `SGDClassifier(loss="log_loss", penalty="l2")`, alpha = {{lr_alpha}} chosen from {{lr_alpha_grid}}; converged in {{lr_n_iter}} of {{lr_max_iter}} epochs | `results/converged_lr_selection.json` |
| TF-IDF + LR (saga) | same TF-IDF | `LogisticRegression(solver="saga")`, max_iter = {{saga_max_iter}}, tol = {{saga_tol}}; **under-converged**, kept only for transparency | `results/classical_selection.json` |
| LDA + LR (paper method) | LDA topic mixture on raw counts ({{lda_count_features}} terms, min_df {{lda_min_df}}, {{lda_max_iter}} batch iterations) | `LogisticRegression` (lbfgs); K = {{lda_k}} topics chosen from {{lda_grid}} | `results/topic_selection.json` |
| Doc2Vec + LR (paper method) | PV-DM, {{d2v_vector_size}} dims, window {{d2v_window}}, {{d2v_epochs}} epochs, first {{d2v_max_tokens}} tokens per document; trained on train only, vectors inferred for val/test | `LogisticRegression` (lbfgs) | `results/topic_selection.json` |
| Legal-BERT | `{{bert_model}}`, first {{bert_max_len}} tokens | fine-tuned {{bert_epochs}} epochs, batch {{bert_batch}} x accumulation {{bert_accum}} = {{bert_eff_batch}}, learning rate {{bert_lr}}, weight decay {{bert_weight_decay}}, linear warm-up over {{bert_warmup_pct}}% of steps, mixed precision, seed {{bert_seed}}, plain cross-entropy (no class weights), best epoch by validation macro F1; {{bert_gpu}}, {{bert_minutes}} minutes | `results/transformer_run_info.json` |

All hyperparameters were chosen on the validation split; final models were fitted on the training split only.""")


def limitations_block():
    return render("""\
- **Truncation.** Legal-BERT reads only the first {{bert_max_len}} tokens, while the median sampled document has about {{tok_median}} tokens; the first {{bert_max_len}} tokens cover about {{tok_pct_covered}}% of a document on average. Its result is a result for truncated input, not for Legal-BERT on full opinions.
- **Reduced paper methods.** LDA ({{lda_count_features}} count features, {{lda_max_iter}} iterations) and Doc2Vec ({{d2v_epochs}} epochs, first {{d2v_max_tokens}} tokens) ran in reduced CPU configurations. The selected K = {{lda_k}} is at the edge of the grid. Their scores are not a fair measure of what these methods can do.
- **One seed.** Each model was trained once (seed {{seed}}). Legal-BERT varied from {{bert_run_min_4}} to {{bert_run_max_4}} test macro F1 across three identical runs.
- **Chronological split.** Test cases ({{test_years}}) come from a later period than training cases ({{train_years}}). Every trained model drops from validation to test (LinearSVC by {{svc_drop}} macro F1); temporal drift is a plausible cause but was not tested.
- **Tiny rare classes.** Test support is {{rare_desc}}, so one prediction can move a class F1 a lot and macro F1 is noisy.
- **Unpaired CIs.** Bootstrap intervals are per model and capture test sampling only; they are not a paired significance test.
- **No class weighting** in any model, and no hyperparameter search beyond the small grids above.
- **Demo artifacts.** The demo LinearSVC artifact uses C = {{demo_svc_c}}, not the reported C = {{svc_c}}; on the test set it agrees with the reported predictions on {{demo_svc_agree}}% of documents (macro F1 {{demo_svc_f1}}). The demo LR agrees on {{demo_lr_agree}}%. The `.joblib` files were saved with scikit-learn {{demo_pickle_sklearn}}, which differs from the pinned version. Source: `results/demo_artifact_check.json`.""")


def summary_block():
    return render(
        "Best model on the test split: **{{best_model}}**, macro F1 {{svc_test_f1}} "
        "(95% bootstrap CI {{svc_ci_lo}}–{{svc_ci_hi}}), accuracy {{svc_test_acc}}. "
        "Fine-tuned Legal-BERT on the first {{bert_max_len}} tokens reaches macro F1 "
        "{{bert_test_f1}} (CI {{bert_ci_lo}}–{{bert_ci_hi}}); the paper's LDA and Doc2Vec "
        "pipelines (reduced CPU configurations) reach {{lda_test_f1}} and {{d2v_test_f1}}."
    )


BLOCKS = {
    "summary": summary_block,
    "dataset": dataset_block,
    "models": models_block,
    "results": results_block,
    "limitations": limitations_block,
}


def update(text):
    for name, builder in BLOCKS.items():
        pattern = re.compile(
            rf"(<!-- BEGIN GENERATED: {name} -->\n)(?:.*?\n)?(<!-- END GENERATED: {name} -->)",
            re.DOTALL,
        )
        if not pattern.search(text):
            raise SystemExit(f"README.md is missing the '{name}' marker block")
        text = pattern.sub(lambda m: m.group(1) + builder() + "\n" + m.group(2), text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = README.read_text(encoding="utf-8")
    new = update(current)
    if args.check:
        if new != current:
            raise SystemExit("README.md generated sections are out of date")
        print("README.md generated sections are up to date")
        return
    README.write_text(new, encoding="utf-8")
    print(f"Updated {len(BLOCKS)} generated sections in README.md")


if __name__ == "__main__":
    main()
