"""
Data loading, schema validation, duplicate auditing, and split management.
Dataset: coastalcph/lex_glue (config: 'scotus')
"""

import os
import json
import hashlib
import random
from typing import Dict, Any, List
import datasets

# Official SCDB Issue Area mapping corresponding to LexGLUE SCOTUS labels 1..13 (indices 0..12)
# Source: SCDB Codebook (Spaeth et al.) & LexGLUE benchmark (Chalkidis et al., ACL 2022)
SCDB_ISSUE_AREA_MAP = {
    0: "Criminal Procedure",   # Raw label '1'
    1: "Civil Rights",          # Raw label '2'
    2: "First Amendment",      # Raw label '3'
    3: "Due Process",          # Raw label '4'
    4: "Privacy",              # Raw label '5'
    5: "Attorneys",            # Raw label '6'
    6: "Unions",               # Raw label '7'
    7: "Economic Activity",    # Raw label '8'
    8: "Judicial Power",       # Raw label '9'
    9: "Federalism",           # Raw label '10'
    10: "Interstate Relations", # Raw label '11'
    11: "Federal Taxation",     # Raw label '12'
    12: "Miscellaneous",        # Raw label '13'
}

SCDB_LABEL_NAMES = [SCDB_ISSUE_AREA_MAP[i] for i in range(13)]
SCDB_LABEL_MAP = SCDB_ISSUE_AREA_MAP
SCDB_STR_LABEL_MAP = {str(i + 1): SCDB_ISSUE_AREA_MAP[i] for i in range(13)}


def load_scotus_dataset(cache_dir: str = None) -> datasets.DatasetDict:
    """
    Loads the LexGLUE SCOTUS dataset from coastalcph/lex_glue.
    """
    ds = datasets.load_dataset("coastalcph/lex_glue", "scotus", cache_dir=cache_dir)
    return ds


def get_dataset_label_names(ds: datasets.DatasetDict) -> List[str]:
    """
    Extracts raw class label names directly from the dataset features.
    """
    label_feature = ds["train"].features["label"]
    if hasattr(label_feature, "names"):
        return label_feature.names
    return [str(i + 1) for i in range(13)]


def validate_schema(ds: datasets.DatasetDict) -> Dict[str, Any]:
    """
    Validates required splits, column names, null counts, and label consistency.
    """
    required_splits = ["train", "validation", "test"]
    missing_splits = [s for s in required_splits if s not in ds]
    if missing_splits:
        raise ValueError(f"Missing required splits in dataset: {missing_splits}")

    schema_report = {}
    for split_name in required_splits:
        split = ds[split_name]
        col_names = split.column_names
        if "text" not in col_names or "label" not in col_names:
            raise ValueError(f"Split {split_name} missing required columns 'text' or 'label'. Found: {col_names}")

        texts = split["text"]
        labels = split["label"]

        null_texts = sum(1 for t in texts if t is None or len(str(t).strip()) == 0)
        null_labels = sum(1 for l in labels if l is None)

        schema_report[split_name] = {
            "num_rows": len(split),
            "columns": col_names,
            "null_text_count": null_texts,
            "null_label_count": null_labels,
            "label_feature": str(split.features["label"]),
        }

    return schema_report


def check_duplicates(ds: datasets.DatasetDict) -> Dict[str, Any]:
    """
    Detects exact string duplicates within splits and across split pairs using SHA-256 hashes.
    """
    hashes_per_split: Dict[str, List[str]] = {}
    internal_duplicates: Dict[str, int] = {}
    cross_split_duplicates: Dict[str, int] = {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }

    for split_name in ["train", "validation", "test"]:
        split_hashes = []
        unique_in_split = set()
        dup_count = 0
        for text in ds[split_name]["text"]:
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            split_hashes.append(h)
            if h in unique_in_split:
                dup_count += 1
            else:
                unique_in_split.add(h)
        hashes_per_split[split_name] = split_hashes
        internal_duplicates[split_name] = dup_count

    train_set = set(hashes_per_split["train"])
    val_set = set(hashes_per_split["validation"])
    test_set = set(hashes_per_split["test"])

    cross_split_duplicates["train_validation"] = len(train_set.intersection(val_set))
    cross_split_duplicates["train_test"] = len(train_set.intersection(test_set))
    cross_split_duplicates["validation_test"] = len(val_set.intersection(test_set))
    cross_split_duplicates["total_cross_split_leaks"] = (
        cross_split_duplicates["train_validation"]
        + cross_split_duplicates["train_test"]
        + cross_split_duplicates["validation_test"]
    )

    return {
        "internal_duplicates": internal_duplicates,
        "cross_split_duplicates": cross_split_duplicates,
    }


def compute_class_distributions(ds: datasets.DatasetDict) -> Dict[str, Dict[str, int]]:
    """
    Computes exact frequency of each class in each split.
    """
    dist: Dict[str, Dict[str, int]] = {}
    for split_name in ["train", "validation", "test"]:
        labels = ds[split_name]["label"]
        counts = {name: 0 for name in SCDB_LABEL_NAMES}
        for lbl in labels:
            name = SCDB_LABEL_MAP[lbl]
            counts[name] += 1
        dist[split_name] = counts
    return dist


def save_stratified_sample_fixtures(
    ds: datasets.DatasetDict,
    output_path: str = "data/sample_cases.json",
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Extracts a seeded, class-stratified random sample of test set documents (1 per class).
    Saves concise records to data/sample_cases.json.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rng = random.Random(seed)

    test_split = ds["test"]
    class_to_indices: Dict[int, List[int]] = {i: [] for i in range(13)}
    for idx, item in enumerate(test_split):
        class_to_indices[item["label"]].append(idx)

    stratified_samples = []
    for cls_idx in range(13):
        candidates = class_to_indices[cls_idx]
        if not candidates:
            continue
        chosen_idx = rng.choice(candidates)
        example = test_split[chosen_idx]
        txt = example["text"]
        lbl = example["label"]
        stratified_samples.append({
            "test_index": chosen_idx,
            "label_id": int(lbl),
            "raw_scdb_code": int(lbl) + 1,
            "label_name": SCDB_LABEL_MAP[lbl],
            "char_length": len(txt),
            "word_count": len(txt.split()),
            "snippet": txt[:300] + "...",
            "full_text": txt[:3000],  # bounded sample for offline test fixture
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stratified_samples, f, indent=2)

    return stratified_samples
