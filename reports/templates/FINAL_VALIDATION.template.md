# Final validation

Local implementation and validation are complete. Faculty review, repository
access and submission remain manual. This is evidence of readiness, not a
claim that the faculty has awarded marks or accepted the submission.

Requirements were read directly from `docs/Guidelines.pdf`. Its write-up heading
says “One-Page”, but the body explicitly requests a two-page summary; the body
and the team handoff govern the page limit used here.

Model values and settings are rendered through `src/report_facts.py`. Validation
counts and command transcripts are rendered by its separate `validation_facts()`
from `results/validation/`; validation logs never authorize model-report numbers.
Rebuild with `.venv/bin/python scripts/make_final_validation.py` and check with
the same command plus `--check`.

| Requirement | Status | Evidence |
|---|---|---|
| Assigned team and problem statement | MANUAL: PENDING | Problem is documented in README and write-up; confirm team assignment against the faculty master list and fill Name/SRN placeholders. |
| Course schedule, mandatory review and submission deadline | MANUAL: PENDING | Follow the dates and faculty review schedule in `docs/Guidelines.pdf`; attend the review and submit the PDF and other required deliverables. |
| Private GitHub repository | MANUAL: PENDING | Public according to the supplied handoff; remote visibility was not rechecked. Ahana must make it private. Local changes remain unpushed. |
| Repository shared with faculty/TAs | MANUAL: PENDING | Owner must grant and verify access. Local validation cannot establish remote permissions. |
| README with setup and run instructions | PASS | Generated blocks are current; every path in the repository tree exists. Setup, artifact download, evaluation, rebuild and Streamlit commands are documented. |
| PDF format, page limit and required content | PASS | `reports/writeup.pdf`: {{validation_pages}} pages; Problem statement, Dataset details, Approach, Brief implementation overview and Conclusions all verified. |
| Slide presentation | READY; REVIEW PENDING | `reports/slides.pptx`: {{validation_slides}} slides and {{validation_notes}} non-empty speaker notes; both team placeholders verified. |
| Deliverable quality and write-up clarity | READY; FACULTY ASSESSMENT PENDING | PDF pages and all slides visually inspected; no clipping or overflow observed. `docs/CITATIONS.md` records reference verification. |
| Code functionality | PASS | Local suite: {{validation_local_tests}}. Includes live downloaded TF-IDF models and Legal-BERT, report checks and runner failure propagation. |
| Mandatory live demonstration | READY; REVIEW PENDING | `app.py`, `tests/test_app.py`, live-model tests and `reports/figures/demo_screenshot.png`; startup without artifacts and sample classification covered. Rehearse and demonstrate to faculty. |
| Methodology, code explanation and individual Q&A | READY; REHEARSAL PENDING | `reports/VIVA_QA.md` equals a fresh render and includes per-module walkthroughs; each member must study their implementation. |
| Repository maintenance | PASS LOCALLY | Separate audit, report-test, runner and fresh-clone commits under Shrihari's identity; author history below. No model retraining or push performed. |
| Individual contribution | RECORDED; OWNERS PENDING | `git shortlog -sne HEAD` and author history below; `reports/CONTRIBUTIONS.md` records the phase split. Agree and fill the blank Owner column and Name/SRN placeholders. |
| AI assistance disclosure | PASS | `reports/CONTRIBUTIONS.md` and README disclose assistance; Codex commits carry an honest co-author trailer. |
| Dataset and chronological split | PASS | `results/dataset_info.json` and `results/eda_stats.json`: {{n_train}} train, {{n_val}} validation, {{n_test}} test, {{n_classes}} classes. Chronological split verified in `docs/CITATIONS.md`. |
| Preprocessing and leakage prevention | PASS | `src/preprocess.py`, preprocessing/leakage tests and saved EDA: {{cross_split_dups}} cross-split duplicates; fitted text transforms use training data only. |
| Models and model-selection identity | PASS WITH DISCLOSED LIMITS | Final LR is SGD log loss; saga remains an under-converged reference. Reported Legal-BERT uses saved reported-run predictions. LDA/Doc2Vec use reduced CPU configurations. |
| Evaluation and uncertainty | PASS | Saved prediction tests recompute metrics and per-class scores. Bootstrap intervals are unpaired and cover test sampling only; one seed and rare-class uncertainty are disclosed. |
| Report tests and rebuild runner | PASS | `tests/test_reports.py`, `tests/test_run_all.py`; `run_all.py --tests` completes all steps. Rebuilding in the original environment leaves report artifacts byte-identical. |
| Fresh-clone dependency installation | PASS OFFLINE; ONLINE UNVERIFIED | Network index lookup failed. Intact cached wheels installed into an isolated venv with pinned requirements; `pip check` passed. Shared package and dataset caches are disclosed below. |
| Fresh-clone tests | PASS | {{validation_fresh_tests}}. Artifact-dependent tests skip because model weights were not cloned. |
| Fresh-clone results reproduction | PASS NUMERICALLY; BYTE DRIFT DISCLOSED | `metrics.csv`, `per_class_f1.csv`, `bootstrap_ci.json` and `error_analysis.md` are byte-identical. {{validation_png_count}} PNGs have identical pixels despite changed bytes; NPZ member contents are identical. Deck drift is limited to an embedded PNG. |
| No fabricated report numbers | PASS WITH AUDIT SCOPE | Number audit passes for README, PDF, slide text/notes, viva and contributions. Decimal coverage tightened from {{validation_loose_coverage}} to {{validation_strict_coverage}}; both negative controls fail. It checks value vocabulary, not semantic attribution or image text. |
| Demo LinearSVC differs from reported model | OPEN: AHANA | Demo C={{demo_svc_c}}, reported C={{svc_c}}; agreement {{demo_svc_agree}}%; demo macro F1 {{demo_svc_f1}} versus reported {{svc_test_f1}}. App/README/viva disclose it. Rebuilding and uploading the demo artifact is Ahana's decision and was not performed. |
| Demo pickle dependency mismatch | OPEN: AHANA | Artifacts were saved with scikit-learn {{demo_pickle_sklearn}}, local pinned version is {{validation_sklearn}}. Warnings reproduced; demo LR agrees with {{demo_lr_agree}}% of reported predictions. |
| Announced artifact-download interface | OPEN: AHANA | The supplied handoff reports that the announced interface was not on GitHub. Current local script lacks `--bert`; downloads need `HF_TOKEN`, with clean exit when absent. README documents current behavior. Remote updates were not fetched. |
| Incorrect configuration summary | RESOLVED IN REPORTS | Model TF-IDF uses {{tfidf_max_features}} features; {{bench_features}} belongs to the preprocessing benchmark. LDA uses {{lda_max_iter}} iterations. Unsourced median word-count claims are omitted. |
| First Legal-BERT run and demo rerun | DISCLOSED | First-run macro F1 {{bert_first_run_f1}} is sourced from `docs/RUN_LOG.md`, with no saved prediction file. Run range {{bert_run_min_4}}–{{bert_run_max_4}} is variation, without best-run selection. Demo weights come from the rerun. |
| Long documents and rare classes | DISCLOSED | Legal-BERT reads {{bert_max_len}} tokens; median sampled length {{tok_median}} tokens. Rare test support: {{rare_desc}}. |
| External paper/benchmark comparisons | DISCLOSED | LexGLUE hierarchical-model scores and contradictory Iyer paper numbers are not quoted. Comparisons are qualitative and limitations explicit. |
| Teammate coordination and final publishing | MANUAL: PENDING | Share the open issues with Ahana, obtain agreement before pushing, then rebase on remote main and validate again. No message, upload, visibility change or push was performed. |

## Validation scope and provenance

The fresh clone tested commit `{{validation_clone_commit}}` in
`{{validation_clone_path}}`. It used a newly created venv and no copied model
weights. pip and Hugging Face caches were shared with this Mac. The ordinary
requirements installation failed due to DNS/network restrictions; the fallback
recovered unchanged wheel archives from pip's HTTP cache, then used `--no-index`
and `--find-links`. This validates an offline installation from available cached
packages, not access to PyPI or a cold dataset download on another machine.

The original checkout's report rebuild was byte-identical. The fresh evaluation
changed PNG encodings and the NPZ container while preserving pixels and member
contents. Rebuilding slides then changed only `ppt/media/image4.png` inside the
deck. These changes remain in the temporary clone; no evaluation artifact was
copied back. The clone is deliberately retained for inspection.

{{validation_visual_review}}

Full exact commands, stdout/stderr, exit codes, cache-wheel hashes and file
comparisons are in [`results/validation/fresh_clone.json`](../results/validation/fresh_clone.json).
Local live-model tests, audit coverage and artifact hashes are in
[`results/validation/local_checks.json`](../results/validation/local_checks.json).
The older `results/test_report.txt` is an earlier run; the captured validation
records above are the evidence for this handoff.

## Exact fresh-clone commands

The outer command was `.venv/bin/python scripts/check_fresh_clone.py`.
The script records commands before reporting their status. The failed online
installation is retained along with the successful offline fallback.

| Working directory | Exact command and environment overrides | Exit code |
|---|---|---:|
{{validation_commands}}

## Test and audit outputs

Fresh-clone pytest:

```text
{{validation_fresh_test_output}}```

Local number audit:

```text
{{validation_audit_output}}
```

Deliberately fabricated negative controls, rejected by the audit:

```text
{{validation_negative_controls}}
```

## Evaluation and byte comparison evidence

```text
{{validation_evaluation}}```

| Required output | SHA-256 before evaluation | After evaluation |
|---|---|---|
{{validation_hash_rows}}

Exact evaluation diff:

```text
{{validation_diff_stat}}```

Fresh-clone rebuild output:

```text
{{validation_clone_runall}}```

Exact clone status after evaluation and document rebuild:

```text
{{validation_clone_status}}```

## Contribution history checkpoint

Recorded at commit `{{validation_context_commit}}`, before the final-validation
documentation commit. Counts are a historical checkpoint, not a live total or
a measure of effort. The final documentation commit extends Shrihari's history.

`git shortlog -sne HEAD`:

```text
{{validation_shortlog}}```

`git log -10 --format='%h %an <%ae> %s'`:

```text
{{validation_history}}```

The team's phase split remains the supplied agreement: Ahana owns the data,
models and evaluation work; Shrihari owns the demo, additional tests and
deliverable/validation work. No ownership was invented. Older teammate status
documents were left unchanged; this file records the completed local work.
