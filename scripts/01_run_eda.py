"""
Script 01: Run Exploratory Data Analysis (EDA) on LexGLUE SCOTUS.
Verifies schema, measures real split sizes, audits duplicate leakage,
computes class distributions, evaluates token/word length percentiles,
finds shortest documents, inspects text openings, evaluates majority baseline,
and generates publication-quality figures.
"""

import os
import sys
import json
import random
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer
import datasets

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import (
    load_scotus_dataset,
    validate_schema,
    check_duplicates,
    compute_class_distributions,
    save_stratified_sample_fixtures,
    get_dataset_label_names,
    SCDB_LABEL_NAMES,
    SCDB_LABEL_MAP,
)
from src.utils import set_seed, save_json, detect_hardware


def run_eda():
    set_seed(42)
    print("==================================================================")
    print("PHASE 3: EXPLORATORY DATA ANALYSIS (LEXGLUE SCOTUS - AUDIT FIXES)")
    print("==================================================================")

    # 1. Load dataset
    print("[1/8] Loading coastalcph/lex_glue (scotus)...")
    ds = load_scotus_dataset()
    raw_feature_names = get_dataset_label_names(ds)
    print(f"Dataset loaded. Raw feature names in dataset: {raw_feature_names}")

    # 2. Schema validation
    print("[2/8] Validating dataset schema...")
    schema_report = validate_schema(ds)

    # 3. Duplicate and leakage audit
    print("[3/8] Auditing duplicates and cross-split leakage...")
    dup_report = check_duplicates(ds)

    # 4. Class distributions
    print("[4/8] Computing class distributions...")
    class_dist = compute_class_distributions(ds)

    # 5. Shortest documents analysis (across all splits)
    print("[5/8] Inspecting the 5 shortest documents across all splits...")
    all_docs = []
    for split in ["train", "validation", "test"]:
        for idx, (txt, lbl) in enumerate(zip(ds[split]["text"], ds[split]["label"])):
            w_count = len(txt.split())
            all_docs.append({
                "split": split,
                "split_index": idx,
                "word_count": w_count,
                "char_length": len(txt),
                "label_id": lbl,
                "label_name": SCDB_LABEL_MAP[lbl],
                "text": txt,
            })
    all_docs_sorted = sorted(all_docs, key=lambda x: x["word_count"])
    shortest_5 = all_docs_sorted[:5]

    # 6. Token length measurement with nlpaueb/legal-bert-base-uncased tokenizer
    print("[6/8] Measuring subword token lengths with nlpaueb/legal-bert-base-uncased tokenizer (500 train samples)...")
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    rng = random.Random(42)
    sample_indices = rng.sample(range(len(ds["train"])), 500)
    sample_texts = [ds["train"][i]["text"] for i in sample_indices]

    token_counts = []
    fractions_covered_by_512 = []
    for t in sample_texts:
        # Tokenize without truncation to measure true length
        enc = tokenizer.encode(t, add_special_tokens=True, truncation=False)
        tok_len = len(enc)
        token_counts.append(tok_len)
        frac = min(1.0, 512.0 / tok_len)
        fractions_covered_by_512.append(frac)

    token_pct = np.percentile(token_counts, [0, 25, 50, 75, 90, 95, 99, 100])
    pct_over_512 = float(np.mean([1 if c > 512 else 0 for c in token_counts]) * 100)
    mean_frac_covered = float(np.mean(fractions_covered_by_512) * 100)

    token_stats = {
        "sample_size": len(token_counts),
        "tokenizer": "nlpaueb/legal-bert-base-uncased",
        "token_count_min": int(token_pct[0]),
        "token_count_p25": float(token_pct[1]),
        "token_count_median": float(token_pct[2]),
        "token_count_p75": float(token_pct[3]),
        "token_count_p90": float(token_pct[4]),
        "token_count_p95": float(token_pct[5]),
        "token_count_p99": float(token_pct[6]),
        "token_count_max": int(token_pct[7]),
        "token_count_mean": float(np.mean(token_counts)),
        "token_count_std": float(np.std(token_counts)),
        "pct_docs_exceeding_512_tokens": pct_over_512,
        "mean_pct_covered_by_first_512_tokens": mean_frac_covered,
    }

    # Inspect the first 300 characters of 3 documents
    print("[6b/8] Inspecting first 300 characters of 3 random decisions...")
    opening_snippets = []
    for i in sample_indices[:3]:
        snippet = ds["train"][i]["text"][:300]
        opening_snippets.append({
            "train_index": i,
            "label": SCDB_LABEL_MAP[ds["train"][i]["label"]],
            "snippet": snippet,
        })

    # 7. Compute Majority-Class Baseline on Test Split
    print("[7/8] Computing majority-class baselines on test split...")
    # Training majority class
    train_labels = ds["train"]["label"]
    val_labels = ds["validation"]["label"]
    test_labels = ds["test"]["label"]

    train_class_counts = pd.Series(train_labels).value_counts()
    train_majority_class = int(train_class_counts.index[0])
    train_majority_name = SCDB_LABEL_MAP[train_majority_class]

    # Predictions where every test sample is predicted as the training majority class
    y_true_test = test_labels
    y_pred_train_majority = [train_majority_class] * len(y_true_test)

    maj_acc = accuracy_score(y_true_test, y_pred_train_majority)
    maj_macro_f1 = f1_score(y_true_test, y_pred_train_majority, average="macro", zero_division=0)
    maj_weighted_f1 = f1_score(y_true_test, y_pred_train_majority, average="weighted", zero_division=0)

    # Test set empirical majority (for comparison)
    test_class_counts = pd.Series(test_labels).value_counts()
    test_majority_class = int(test_class_counts.index[0])
    test_majority_name = SCDB_LABEL_MAP[test_majority_class]
    y_pred_test_majority = [test_majority_class] * len(y_true_test)
    test_maj_acc = accuracy_score(y_true_test, y_pred_test_majority)

    majority_baseline = {
        "train_majority_class_id": train_majority_class,
        "train_majority_class_name": train_majority_name,
        "train_majority_count": int(train_class_counts.iloc[0]),
        "train_majority_pct": float((train_class_counts.iloc[0] / len(train_labels)) * 100),
        "test_accuracy": float(maj_acc),
        "test_macro_f1": float(maj_macro_f1),
        "test_weighted_f1": float(maj_weighted_f1),
        "test_majority_class_id": test_majority_class,
        "test_majority_class_name": test_majority_name,
        "test_majority_count": int(test_class_counts.iloc[0]),
        "test_empirical_majority_accuracy": float(test_maj_acc),
    }
    save_json(majority_baseline, "results/majority_baseline.json")

    # 8. Save Stratified Test Samples (Seeded 42)
    print("[8/8] Saving seeded class-stratified test samples to data/sample_cases.json...")
    stratified_samples = save_stratified_sample_fixtures(ds, output_path="data/sample_cases.json", seed=42)

    # Compute word length stats across all splits
    length_stats = {}
    all_word_counts = {}
    for split in ["train", "validation", "test"]:
        words = [len(t.split()) for t in ds[split]["text"]]
        all_word_counts[split] = words
        pct = np.percentile(words, [0, 25, 50, 75, 90, 95, 99, 100])
        length_stats[split] = {
            "total_documents": len(words),
            "word_count_min": int(pct[0]),
            "word_count_p25": float(pct[1]),
            "word_count_median": float(pct[2]),
            "word_count_p75": float(pct[3]),
            "word_count_p90": float(pct[4]),
            "word_count_p95": float(pct[5]),
            "word_count_p99": float(pct[6]),
            "word_count_max": int(pct[7]),
            "word_count_mean": float(np.mean(words)),
            "word_count_std": float(np.std(words)),
        }

    # Persist updated dataset_info.json and eda_stats.json
    dataset_info = {
        "dataset_name": "coastalcph/lex_glue",
        "config_name": "scotus",
        "split_method": "Chronological (Train: 1946-1982, Validation: 1982-1991, Test: 1991-2016 per LexGLUE Chalkidis et al. 2022)",
        "load_timestamp": datetime.now().isoformat(),
        "splits": {s: len(ds[s]) for s in ds.keys()},
        "columns": ds["train"].column_names,
        "num_classes": len(SCDB_LABEL_NAMES),
        "raw_label_feature_names": raw_feature_names,
        "label_mapping_to_scdb_names": SCDB_LABEL_MAP,
        "note_on_14th_class": "SCDB defines 14 issue areas. Class 14 ('Private Action') has 0 examples in modern SCDB (1946-2016), resulting in 13 active classes (labels 1-13, indices 0-12).",
        "library_versions": {
            "datasets": datasets.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "hardware_at_load": detect_hardware(),
    }
    save_json(dataset_info, "results/dataset_info.json")

    eda_stats = {
        "schema_report": schema_report,
        "duplicate_report": dup_report,
        "class_distributions": class_dist,
        "length_statistics_words_by_split": length_stats,
        "token_statistics_legal_bert": token_stats,
        "opening_snippets": opening_snippets,
        "shortest_5_documents": [
            {k: v for k, v in doc.items() if k != "text"} for doc in shortest_5
        ],
        "majority_class_baseline": majority_baseline,
    }
    save_json(eda_stats, "results/eda_stats.json")

    # Generate Figures
    os.makedirs("results/figures", exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # Figure 1: Class Distribution per split
    df_dist = pd.DataFrame(class_dist)
    df_dist["Class"] = df_dist.index
    df_melted = pd.melt(df_dist, id_vars=["Class"], value_vars=["train", "validation", "test"],
                        var_name="Split", value_name="Count")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x="Count", y="Class", hue="Split", orient="h")
    plt.title("LexGLUE SCOTUS: Class Frequency by Split (SCDB Issue Areas)", fontsize=14, weight="bold")
    plt.xlabel("Number of Decisions", fontsize=12)
    plt.ylabel("SCDB Issue Area", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/figures/class_distribution.png", dpi=300)
    plt.close()

    # Figure 2: TRUE TOKEN-LENGTH HISTOGRAM on Token Axis with 512 Marker
    plt.figure(figsize=(10, 5))
    sns.histplot(token_counts, bins=40, kde=True, color="steelblue", stat="density")
    plt.axvline(512, color="crimson", linestyle="--", linewidth=2.0,
                label=f"512 Tokens (Standard BERT Window) - {pct_over_512:.1f}% Docs Exceed Limit")
    plt.axvline(token_stats["token_count_median"], color="darkgreen", linestyle=":", linewidth=2.0,
                label=f"Median Token Length = {int(token_stats['token_count_median']):,} Tokens")
    plt.title("LexGLUE SCOTUS: Subword Token Length Distribution (nlpaueb/legal-bert-base-uncased)",
              fontsize=13, weight="bold")
    plt.xlabel("Subword Tokens per Opinion (measured on 500 train samples)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("results/figures/token_length_histogram.png", dpi=300)
    # Also save as document_length_histogram.png to keep filename compatibility
    plt.savefig("results/figures/document_length_histogram.png", dpi=300)
    plt.close()

    # Output detailed report to console
    print("\n" + "=" * 75)
    print("VERIFIED FINDINGS: PHASE 3 AUDIT ITEMS (1 - 6)")
    print("=" * 75)
    print("\n[Item 1: Majority-Class Baseline on Test Split]")
    print(f"  Training Majority Class : [{train_majority_class}] {train_majority_name} ({majority_baseline['train_majority_count']}/{len(train_labels)} = {majority_baseline['train_majority_pct']:.2f}%)")
    print(f"  Test Set Accuracy       : {maj_acc:.4f} ({maj_acc*100:.2f}%)")
    print(f"  Test Set Macro F1       : {maj_macro_f1:.4f}")
    print(f"  Test Set Weighted F1    : {maj_weighted_f1:.4f}")
    print(f"  (For context: Test-set empirical mode is [{test_majority_class}] {test_majority_name}, test accuracy if cheated: {test_maj_acc:.4f})")

    print("\n[Item 2 & 3: Label Space, 13 vs 14 Classes, and Chronological Split]")
    print(f"  Raw feature names in dataset : {raw_feature_names} (13 classes)")
    print(f"  LexGLUE Benchmark Reference  : Chalkidis et al. (ACL 2022)")
    print(f"  Why 13 classes, not 14?      : SCDB defines 14 issue areas. Class 14 ('Private Action') has zero cases in the 1946-2016 period.")
    print(f"  Split method                 : Chronological (Train: 1946-1982, Val: 1982-1991, Test: 1991-2016). Differing from paper's random split.")

    print("\n[Item 4: Measured Token Statistics (nlpaueb/legal-bert-base-uncased, N=500)]")
    print(f"  Min tokens                   : {token_stats['token_count_min']}")
    print(f"  25th percentile              : {token_stats['token_count_p25']:.1f}")
    print(f"  Median tokens                : {token_stats['token_count_median']:.1f}")
    print(f"  75th percentile              : {token_stats['token_count_p75']:.1f}")
    print(f"  Max tokens                   : {token_stats['token_count_max']}")
    print(f"  Mean tokens                  : {token_stats['token_count_mean']:.1f} (+/- {token_stats['token_count_std']:.1f})")
    print(f"  % of documents > 512 tokens  : {pct_over_512:.2f}%")
    print(f"  Mean % covered by first 512  : {mean_frac_covered:.2f}%")

    print("\n[First 300 characters of 3 decisions (Opening text analysis)]:")
    for idx, snip in enumerate(opening_snippets):
        print(f"  Document {idx+1} [Train idx={snip['train_index']}, {snip['label']}]:")
        clean_snip = snip['snippet'].replace('\n', ' ')
        print(f"    \"{clean_snip}\"")

    print("\n[Item 5: Five Shortest Documents in the Dataset]")
    for idx, doc in enumerate(shortest_5):
        clean_text = doc['text'].replace('\n', ' ')
        print(f"  Rank {idx+1}: [{doc['split']} idx={doc['split_index']}] - {doc['word_count']} words, {doc['char_length']} chars, Class: '{doc['label_name']}'")
        print(f"    Text: \"{clean_text[:200]}...\"")

    print("\n==========================================================================")


if __name__ == "__main__":
    run_eda()
