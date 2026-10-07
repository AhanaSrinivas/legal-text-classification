"""Single source for every number quoted in the README, write-up, slides and viva notes.

All values are read from results/ when a document is built. The few settings
that are not saved in results/ are listed in CONFIG together with the file
and the exact literal that proves them; scripts/audit_numbers.py checks that
each literal is still present in that file.

Templates use {{key}} placeholders; render() fails on any unknown key.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Display order: the two TF-IDF models, the transformer, paper methods, references.
MODEL_ORDER = [
    "tfidf_linear_svc", "tfidf_logistic_regression", "transformer_legal_bert",
    "lda_lr", "doc2vec_lr", "tfidf_lr_saga_underconverged", "majority_baseline",
]
SHORT = {
    "tfidf_linear_svc": "svc", "tfidf_logistic_regression": "lr",
    "transformer_legal_bert": "bert", "lda_lr": "lda", "doc2vec_lr": "d2v",
    "tfidf_lr_saga_underconverged": "saga", "majority_baseline": "majority",
}
DISPLAY = {
    "tfidf_linear_svc": "TF-IDF + LinearSVC",
    "tfidf_logistic_regression": "TF-IDF + LR (SGD, log loss)",
    "transformer_legal_bert": "Legal-BERT (first 512 tokens)",
    "lda_lr": "LDA topics + LR",
    "doc2vec_lr": "Doc2Vec + LR",
    "tfidf_lr_saga_underconverged": "TF-IDF + LR (saga, under-converged)",
    "majority_baseline": "Majority class (train)",
}

# Settings that are not stored in results/: key -> (display value, file, literal in file).
CONFIG = {
    "tfidf_max_features": ("10,000", "scripts/train_classical.py", "tfidf_max_features=10_000"),
    "tfidf_min_df": ("2", "src/preprocess.py", "min_df: int = 2"),
    "tfidf_max_df": ("0.98", "src/preprocess.py", "max_df: float = 0.98"),
    "svc_c_grid": ("0.01, 0.1, 1.0", "scripts/train_classical.py", "(0.01, 0.1, 1.0)"),
    "lda_count_features": ("500", "scripts/train_topic.py", "count_max_features=500"),
    "lda_min_df": ("5", "scripts/train_topic.py", "min_df=5"),
    "lda_max_iter": ("2", "scripts/train_topic.py", "max_iter=2"),
    "d2v_min_count": ("2", "scripts/train_topic.py", "min_count=2"),
    "bert_weight_decay": ("0.01", "scripts/04_train_transformer.py",
                          '"--weight_decay", type=float, default=0.01'),
    "bert_warmup_pct": ("10", "scripts/04_train_transformer.py", "int(total_steps * 0.1)"),
    "bert_first_run_f1": ("0.5171", "docs/RUN_LOG.md", "`0.5171` from the"),
    "fixture_chars": ("3,000", "src/data.py", "txt[:3000]"),
    "ci_lower_pct": ("2.5", "scripts/evaluate_all.py", "np.percentile(values, 2.5)"),
    "ci_upper_pct": ("97.5", "scripts/evaluate_all.py", "np.percentile(values, 97.5)"),
    "bert_content_tokens": ("510", "src/demo.py", "usable = limit - 2  # [CLS] and [SEP]"),
}


def _f3(x):
    return f"{x:.3f}"


def _f4(x):
    return f"{x:.4f}"


def _int(x):
    return f"{int(round(x)):,}"


def _slug(name):
    return re.sub(r"[^a-z]+", "_", name.lower()).strip("_")


def _json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _number(x):
    """Integer-looking floats as integers (5000.0 -> 5,000); others as given."""
    return _int(x) if float(x).is_integer() else f"{x:,}"


@lru_cache(maxsize=1)
def metrics():
    return pd.read_csv(RESULTS / "metrics.csv")


def metric(model, split, column):
    rows = metrics()
    row = rows[(rows["model"] == model) & (rows["split"] == split)]
    if len(row) != 1:
        raise KeyError(f"{model}/{split} not in metrics.csv")
    return float(row.iloc[0][column])


def _parse_error_analysis():
    text = (RESULTS / "error_analysis.md").read_text(encoding="utf-8")
    out = {}
    for key, label in (("only_svc", "Only SVC correct"), ("only_bert", "Only Legal-BERT correct"),
                       ("both_wrong", "Both wrong")):
        out[f"ea_{key}"] = int(re.search(rf"- {label}: (\d+)", text).group(1))
    for model, short in (("tfidf_linear_svc", "svc"), ("transformer_legal_bert", "bert")):
        block = text.split(f"### {model}")[1]
        pairs = re.findall(r"- (\d+): ([A-Za-z ]+?) -> ([A-Za-z ]+)\n", block)
        out[f"ea_{short}_pairs"] = [(int(n), a, b) for n, a, b in pairs[:5]]
    quartiles = re.findall(r"\| (\w+) \| (\d) \| (\d+) \| ([0-9.]+) \|", text)
    out["ea_quartiles"] = [(m, int(q), int(n), float(a)) for m, q, n, a in quartiles]
    return out


@lru_cache(maxsize=1)
def facts():
    """Flat dict of display strings, plus a few structured entries (lists/dicts)."""
    f = {}
    info = _json("dataset_info.json")
    eda = _json("eda_stats.json")
    split_sizes = info["splits"]
    f["dataset_name"] = info["dataset_name"]
    f["dataset_config"] = info["config_name"]
    f["datasets_version"] = info["library_versions"]["datasets"]
    f["n_total"] = _int(sum(split_sizes.values()))
    f["n_train"] = _int(split_sizes["train"])
    f["n_val"] = _int(split_sizes["validation"])
    f["n_test"] = _int(split_sizes["test"])
    f["n_classes"] = str(info["num_classes"])
    years = re.findall(r"(\d{4})-(\d{4})", info["split_method"])
    f["train_years"], f["val_years"], f["test_years"] = ("–".join(y) for y in years)
    f["label_names"] = [info["label_mapping_to_scdb_names"][str(i)] for i in range(info["num_classes"])]

    dup = eda["duplicate_report"]
    f["cross_split_dups"] = str(dup["cross_split_duplicates"]["total_cross_split_leaks"])
    f["internal_dups"] = str(sum(dup["internal_duplicates"].values()))
    f["null_texts"] = str(sum(eda["schema_report"][s]["null_text_count"] for s in split_sizes))

    train_counts = eda["class_distributions"]["train"]
    test_counts = eda["class_distributions"]["test"]
    largest = max(train_counts, key=train_counts.get)
    smallest = min(train_counts, key=train_counts.get)
    f["largest_class"], f["largest_count"] = largest, _int(train_counts[largest])
    f["smallest_class"], f["smallest_count"] = smallest, _int(train_counts[smallest])
    f["imbalance_ratio"] = f"{train_counts[largest] / train_counts[smallest]:.1f}"
    for name, count in test_counts.items():
        f[f"test_support_{_slug(name)}"] = _int(count)
    for name, count in train_counts.items():
        f[f"train_count_{_slug(name)}"] = _int(count)
    rare = sorted(test_counts, key=test_counts.get)[:3]
    f["rare_classes"] = rare
    f["rare_supports"] = ", ".join(_int(test_counts[c]) for c in rare)
    f["rare_desc"] = ", ".join(f"{c} ({_int(test_counts[c])})" for c in rare)

    words = eda["length_statistics_words_by_split"]
    for split, key in (("train", "train"), ("validation", "val"), ("test", "test")):
        f[f"words_median_{key}"] = _number(words[split]["word_count_median"])
        f[f"words_mean_{key}"] = _int(words[split]["word_count_mean"])
    tokens = eda["token_statistics_legal_bert"]
    f["tok_sample"] = _int(tokens["sample_size"])
    f["tok_median"] = _int(int(tokens["token_count_median"]))  # 5470.5 -> 5,470
    f["tok_median_exact"] = _number(tokens["token_count_median"])
    f["tok_pct_over_512"] = f"{tokens['pct_docs_exceeding_512_tokens']:.0f}"
    f["tok_pct_covered"] = f"{tokens['mean_pct_covered_by_first_512_tokens']:.0f}"
    f["tokenizer"] = tokens["tokenizer"]

    for model in MODEL_ORDER:
        s = SHORT[model]
        for split, sp in (("test", "test"), ("validation", "val")):
            f[f"{s}_{sp}_f1"] = _f3(metric(model, split, "macro_f1"))
            f[f"{s}_{sp}_f1_4"] = _f4(metric(model, split, "macro_f1"))
            f[f"{s}_{sp}_acc"] = _f3(metric(model, split, "accuracy"))
            f[f"{s}_{sp}_wf1"] = _f3(metric(model, split, "weighted_f1"))
            f[f"{s}_{sp}_mp"] = _f3(metric(model, split, "macro_precision"))
            f[f"{s}_{sp}_mr"] = _f3(metric(model, split, "macro_recall"))
        f[f"{s}_drop"] = _f3(metric(model, "validation", "macro_f1") - metric(model, "test", "macro_f1"))

    ci = _json("bootstrap_ci.json")
    for model in MODEL_ORDER:
        f[f"{SHORT[model]}_ci_lo"] = _f3(ci[model]["lower_95"])
        f[f"{SHORT[model]}_ci_hi"] = _f3(ci[model]["upper_95"])
    f["ci_resamples"] = _int(ci["tfidf_linear_svc"]["resamples"])
    f["ci_seed"] = str(ci["tfidf_linear_svc"]["seed"])
    def overlap(a, b):
        return ci[a]["lower_95"] <= ci[b]["upper_95"] and ci[b]["lower_95"] <= ci[a]["upper_95"]

    f["svc_bert_ci_overlap"] = (
        "overlap" if overlap("tfidf_linear_svc", "transformer_legal_bert") else "do not overlap"
    )
    f["svc_lr_ci_overlap"] = (
        "overlap" if overlap("tfidf_linear_svc", "tfidf_logistic_regression") else "do not overlap"
    )

    test_rank = sorted(MODEL_ORDER, key=lambda m: -metric(m, "test", "macro_f1"))
    f["best_model"] = DISPLAY[test_rank[0]]
    f["test_rank"] = test_rank

    majority = _json("majority_baseline.json")
    f["majority_class"] = majority["train_majority_class_name"]
    f["test_majority_class"] = majority["test_majority_class_name"]
    f["test_majority_acc"] = _f3(majority["test_empirical_majority_accuracy"])
    for name, count in eda["class_distributions"]["validation"].items():
        f[f"val_count_{_slug(name)}"] = _int(count)

    classical = _json("classical_selection.json")
    f["svc_c"] = str(classical["tfidf_linear_svc"]["params"][0])
    saga = classical["tfidf_lr_saga_underconverged"]["configuration"]
    f["saga_max_iter"] = str(saga["max_iter"])
    f["saga_tol"] = str(saga["tol"])
    f["saga_c_grid"] = ", ".join(str(c) for c in saga["c_grid"])
    f["saga_c"] = str(classical["tfidf_lr_saga_underconverged"]["params"][0])

    lr = _json("converged_lr_selection.json")
    f["lr_alpha"] = f"{lr['selected_alpha']:.0e}".replace("e-0", "e-")
    f["lr_alpha_grid"] = ", ".join(f"{a:.0e}".replace("e-0", "e-") for a in lr["alpha_grid"])
    f["lr_n_iter"] = str(lr["candidates"][str(lr["selected_alpha"])]["n_iter"])
    f["lr_max_iter"] = str(lr["max_iter"])
    f["lr_tol"] = f"{lr['tol']:.0e}".replace("e-0", "e-")
    f["lr_minutes"] = f"{lr['elapsed_seconds'] / 60:.1f}"
    f["seed"] = str(lr["random_state"])

    topic = _json("topic_selection.json")
    f["lda_k"] = str(topic["selected_topics"])
    f["lda_grid"] = ", ".join(str(k) for k in topic["topics_considered"])
    f["lda_k_at_edge"] = topic["selected_topics"] == max(topic["topics_considered"])
    for k, values in topic["validation"].items():
        f[f"lda_val_f1_k{k}"] = _f3(values["macro_f1"])
    d2v = topic["doc2vec"]
    f["d2v_vector_size"] = str(d2v["vector_size"])
    f["d2v_window"] = str(d2v["window"])
    f["d2v_epochs"] = str(d2v["epochs"])
    f["d2v_max_tokens"] = str(d2v["max_tokens_per_document"])

    run = _json("transformer_run_info.json")
    run2 = _json("transformer_run2_info.json")
    f["bert_model"] = run["model_name"]
    f["bert_max_len"] = str(run["max_length"])
    f["bert_epochs"] = str(run["epochs"])
    f["bert_batch"] = str(run["batch_size"])
    f["bert_accum"] = str(run["gradient_accumulation_steps"])
    f["bert_eff_batch"] = str(run["effective_batch_size"])
    f["bert_lr"] = f"{run['learning_rate']:.0e}".replace("e-0", "e-")
    f["bert_seed"] = str(run["seed"])
    f["bert_gpu"] = run["hardware"]["gpu_name"]
    f["bert_minutes"] = f"{run['elapsed_seconds'] / 60:.1f}"
    f["bert_run2_minutes"] = f"{run2['elapsed_seconds'] / 60:.1f}"
    f["bert_reported_f1_4"] = _f4(run["test_macro_f1"])
    f["bert_run2_f1_4"] = _f4(run2["test_macro_f1"])
    f["bert_run2_acc"] = _f3(run2["test_accuracy"])
    runs = [float(CONFIG["bert_first_run_f1"][0]), run["test_macro_f1"], run2["test_macro_f1"]]
    f["bert_run_min_4"], f["bert_run_max_4"] = _f4(min(runs)), _f4(max(runs))

    per_class = pd.read_csv(RESULTS / "per_class_f1.csv")
    for model in ("tfidf_linear_svc", "tfidf_logistic_regression", "transformer_legal_bert",
                  "lda_lr", "doc2vec_lr"):
        rows = per_class[per_class["model"] == model]
        for _, row in rows.iterrows():
            key = f"{SHORT[model]}_{_slug(row['class'])}"
            f[f"{key}_f1"] = _f3(row["f1"])
            f[f"{key}_p"] = _f3(row["precision"])
            f[f"{key}_r"] = _f3(row["recall"])
            f[f"{key}_rpct"] = f"{100 * row['recall']:.0f}"
        f[f"{SHORT[model]}_zero_f1"] = list(rows[rows["f1"] == 0]["class"])
    f["per_class"] = per_class

    bench = _json("preprocessing_benchmark.json")
    f["bench_features"] = _int(bench["max_features"])
    f["bench_seconds"] = f"{bench['fit_wall_time_seconds']:.0f}"
    f["bench_gb"] = f"{bench['peak_process_rss_gb']:.2f}"

    ablation = pd.read_csv(RESULTS / "numeric_token_ablation_validation.csv")
    f["abl_drop_f1"] = _f3(float(ablation[ablation["drop_numeric_tokens"]]["macro_f1"].iloc[0]))
    f["abl_keep_f1"] = _f3(float(ablation[~ablation["drop_numeric_tokens"]]["macro_f1"].iloc[0]))

    ea = _parse_error_analysis()
    f.update({k: (str(v) if isinstance(v, int) else v) for k, v in ea.items()})
    for model, short in (("tfidf_linear_svc", "svc"), ("transformer_legal_bert", "bert")):
        accuracies = [acc for m, _, _, acc in ea["ea_quartiles"] if m == model]
        f[f"ea_{short}_q_min"], f[f"ea_{short}_q_max"] = _f3(min(accuracies)), _f3(max(accuracies))
        f[f"ea_{short}_q_longest"] = _f3(accuracies[-1])
    n_svc, a_svc, b_svc = ea["ea_svc_pairs"][0]
    f["ea_svc_top_pair"] = f"{a_svc} → {b_svc} ({n_svc})"
    n_bert, a_bert, b_bert = ea["ea_bert_pairs"][0]
    f["ea_bert_top_pair"] = f"{a_bert} → {b_bert} ({n_bert})"
    if (a_svc, b_svc) == (a_bert, b_bert):
        f["ea_top_pair_sentence"] = (
            f"{a_svc} → {b_svc} for both LinearSVC ({n_svc} test cases) "
            f"and Legal-BERT ({n_bert})"
        )
    else:
        f["ea_top_pair_sentence"] = (
            f"{a_svc} → {b_svc} for LinearSVC ({n_svc} test cases) and "
            f"{a_bert} → {b_bert} for Legal-BERT ({n_bert})"
        )

    check_path = RESULTS / "demo_artifact_check.json"
    if check_path.exists():
        check = json.loads(check_path.read_text(encoding="utf-8"))
        svc, lr_demo = check["tfidf_linear_svc"], check["tfidf_logistic_regression"]
        f["demo_svc_c"] = str(svc["hyperparameter"]["C"])
        f["demo_svc_agree"] = f"{100 * svc['test_agreement_with_reported_predictions']:.0f}"
        f["demo_svc_f1"] = _f3(svc["demo_test_macro_f1"])
        f["demo_lr_agree"] = f"{100 * lr_demo['test_agreement_with_reported_predictions']:.0f}"
        f["demo_lr_f1"] = _f3(lr_demo["demo_test_macro_f1"])
        f["demo_pickle_sklearn"] = svc["pickled_with_sklearn"]

    for key, (value, _, _) in CONFIG.items():
        f[key] = value
    return f


def results_rows():
    """Rows for the results tables, in MODEL_ORDER."""
    f = facts()
    rows = []
    for model in MODEL_ORDER:
        s = SHORT[model]
        rows.append({
            "model": model,
            "name": DISPLAY[model],
            "test_f1": f[f"{s}_test_f1"], "test_acc": f[f"{s}_test_acc"],
            "test_wf1": f[f"{s}_test_wf1"],
            "ci": f"[{f[f'{s}_ci_lo']}, {f[f'{s}_ci_hi']}]",
            "val_f1": f[f"{s}_val_f1"], "val_acc": f[f"{s}_val_acc"],
            "val_wf1": f[f"{s}_val_wf1"],
        })
    return rows


_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


def render(template: str) -> str:
    """Replace {{key}} placeholders with facts; unknown keys raise KeyError."""
    f = facts()

    def substitute(match):
        key = match.group(1)
        if key not in f:
            raise KeyError(f"unknown fact placeholder: {key}")
        value = f[key]
        if isinstance(value, (list, tuple)):
            items = [str(v) for v in value]
            return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
        return str(value)

    return _PLACEHOLDER.sub(substitute, template)


def validation_facts():
    """Display validation evidence separately from the model-number allowlist.

    Logs include deliberate fake audit values. They must never enter facts(),
    which authorizes numbers in the model reports.
    """
    local = _json("validation/local_checks.json")
    fresh = _json("validation/fresh_clone.json")
    context = _json("validation/final_context.json")

    def output(records, needle):
        for entry in records:
            command = entry["command"]
            if needle in (" ".join(command) if isinstance(command, list) else command):
                return entry["output"]
        raise KeyError(f"Missing validation command: {needle}")

    def summary(text):
        return re.findall(r"^\d+ passed[^\n]*", text, re.MULTILINE)[-1]

    v = {
        "validation_pages": str(local["pdf_pages"]),
        "validation_slides": str(local["slides"]),
        "validation_notes": str(local["notes"]),
        "validation_local_tests": summary(local["final_runall"]["output"] if "final_runall" in local
                                          else output(local["commands"], "-m pytest")),
        "validation_fresh_tests": summary(output(fresh["commands"], "-m pytest")),
        "validation_fresh_test_output": output(fresh["commands"], "-m pytest"),
        "validation_shortlog": output(context["commands"], "shortlog"),
        "validation_history": output(context["commands"], "git log"),
        "validation_context_commit": output(context["commands"], "rev-parse").strip(),
        "validation_clone_commit": fresh["source_commit"],
        "validation_clone_path": fresh["clone"],
        "validation_sklearn": local["sklearn_version"],
        "validation_visual_review": local["visual_review"],
        "validation_negative_controls": "\n".join(local["negative_control"]["unmatched"]),
        "validation_diff_stat": output(fresh["commands"], "git diff --stat"),
        "validation_clone_status": fresh["final_status"],
        "validation_clone_runall": output(fresh["commands"], "run_all.py"),
        "validation_evaluation": output(fresh["commands"], "scripts/evaluate_all.py"),
        "validation_png_count": str(sum(name.endswith(".png") for name in fresh["evaluation_changes"])),
    }
    for mode, coverage in local["coverage"].items():
        v[f"validation_{mode}_coverage"] = (
            f"{coverage['matched']}/{coverage['total']} ({coverage['percent']})"
        )
    v["validation_audit_output"] = "\n".join(
        line for line in output(local["commands"], "run_all.py").splitlines()
        if line.startswith("PASS  ") or line.startswith("FAIL  ")
    )
    command_rows = []
    import shlex
    for entry in fresh["commands"]:
        command = entry["command"]
        overrides = entry.get("environment_overrides", {})
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in overrides.items())
        command_rows.append(
            f"| `{entry['cwd']}` | `{prefix + ' ' if prefix else ''}{command}` "
            f"| {entry['exit_code']} |"
        )
    v["validation_commands"] = "\n".join(command_rows)
    v["validation_hash_rows"] = "\n".join(
        f"| `{name}` | `{before}` | {'identical' if before == fresh['after'][name] else 'CHANGED'} |"
        for name, before in fresh["before"].items()
    )
    if not fresh["passed"] or local["artifact_diff"]:
        raise ValueError("Validation evidence records a failed check or local artifact drift")
    for name, comparison in fresh["byte_comparisons"].items():
        if name.endswith(".png") and not comparison["pixels_equal"]:
            raise ValueError(f"Validation template assumes identical PNG pixels: {name}")
        if name.endswith(".npz") and not comparison["member_content_equal"]:
            raise ValueError(f"Validation template assumes identical NPZ members: {name}")
        if name.endswith(".pptx") and comparison["changed_zip_members"] != ["ppt/media/image4.png"]:
            raise ValueError(f"Review changed deck members before updating validation: {name}")
    return v


def render_validation(template: str) -> str:
    """Render the validation document using model facts and captured evidence."""
    values = {**facts(), **validation_facts()}
    return _PLACEHOLDER.sub(lambda match: str(values[match.group(1)]), template)
