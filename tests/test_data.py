import json
from pathlib import Path

from src.data import SCDB_LABEL_NAMES, SCDB_STR_LABEL_MAP


ROOT = Path(__file__).resolve().parents[1]


def test_thirteen_issue_areas_in_label_order():
    assert len(SCDB_LABEL_NAMES) == 13
    assert len(set(SCDB_LABEL_NAMES)) == 13
    assert SCDB_LABEL_NAMES[0] == "Criminal Procedure"
    assert SCDB_LABEL_NAMES[7] == "Economic Activity"
    assert SCDB_STR_LABEL_MAP["13"] == "Miscellaneous"


def test_label_names_match_saved_dataset_info():
    info = json.loads((ROOT / "results" / "dataset_info.json").read_text())
    saved = [info["label_mapping_to_scdb_names"][str(i)] for i in range(info["num_classes"])]
    assert saved == SCDB_LABEL_NAMES


def test_sample_fixture_has_one_case_per_class():
    cases = json.loads((ROOT / "data" / "sample_cases.json").read_text())
    assert sorted(case["label_id"] for case in cases) == list(range(13))
    for case in cases:
        assert case["label_name"] == SCDB_LABEL_NAMES[case["label_id"]]
        assert case["raw_scdb_code"] == case["label_id"] + 1
        assert 0 < len(case["full_text"]) <= 3000
        assert 0 <= case["test_index"] < 1400


def test_saved_eda_reports_no_leakage_or_missing_values():
    eda = json.loads((ROOT / "results" / "eda_stats.json").read_text())
    assert eda["duplicate_report"]["cross_split_duplicates"]["total_cross_split_leaks"] == 0
    for split in ("train", "validation", "test"):
        assert eda["duplicate_report"]["internal_duplicates"][split] == 0
        assert eda["schema_report"][split]["null_text_count"] == 0
