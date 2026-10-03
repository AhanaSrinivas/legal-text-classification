# Dataset Decision and Verification Record

## 1. Selected Dataset
- **Candidate**: `coastalcph/lex_glue` (config: `scotus`)
- **Status**: **VERIFIED AND ADOPTED** (Audited and verified on 2026-10-03)
- **Source**: Hugging Face Datasets Hub (`https://huggingface.co/datasets/coastalcph/lex_glue`)
- **Benchmark Reference**: *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English* (Chalkidis et al., ACL 2022).
- **Underlying Source**: Supreme Court Database (SCDB) decisions curated by Washington University in St. Louis.

---

## 2. Dataset Splitting Methodology & Comparison with Reference Paper
- **LexGLUE SCOTUS Split Method**: **Chronological Partitioning**, as described by the [LexGLUE SCOTUS dataset card](https://huggingface.co/datasets/coastalcph/lex_glue).
  - **Training split**: 5,000 cases decided between **1946 and 1982**.
  - **Validation split (dev)**: 1,400 cases decided between **1982 and 1991**.
  - **Test split**: 1,400 cases decided between **1991 and 2016**.
- **Key Difference from Reference Paper**:
  - The reference paper (*Classification of Legal Text*, Iyer 2020) trained models on a random split of ~8,200 decisions extracted via `textacy` (without publishing seeds or split indices).
  - LexGLUE SCOTUS uses an explicit chronological split to reflect realistic temporal generalization (models trained on past jurisprudence must predict future decisions).
  - This difference is intentional and strictly documented.

---

## 3. Label Space: Why 13 Classes Instead of 14?
- The Supreme Court Database (SCDB) codebook defines 14 issue areas:
  1. Criminal Procedure
  2. Civil Rights
  3. First Amendment
  4. Due Process
  5. Privacy
  6. Attorneys
  7. Unions
  8. Economic Activity
  9. Judicial Power
  10. Federalism
  11. Interstate Relations
  12. Federal Taxation
  13. Miscellaneous
  14. Private Action
- **Zero-Example Issue Area**: The [SCDB codebook](https://scdb.wustl.edu/documentation.php?var=version) defines Class 14 as 'Private Action'. The adopted LexGLUE SCOTUS corpus has no active Class 14 examples, as recorded in `results/dataset_info.json`; the reference paper (Iyer 2020, Table 2) likewise observed `# Docs: 0` for Private Action.
- Consequently, `coastalcph/lex_glue` encodes exactly **13 active classes** with raw string IDs `'1'` through `'13'`, mapped to zero-indexed integers `0` through `12`.

---

## 4. Verified Facts & Empirical Baseline

### A. Splits & Sample Sizes
- **Train split**: 5,000 decisions (64.10%)
- **Validation split**: 1,400 decisions (17.95%)
- **Test split**: 1,400 decisions (17.95%)
- **Total corpus**: 7,800 decisions
- **Columns**: `['text', 'label']`
- **Missing / Null records**: 0 null texts, 0 null labels across all splits.

### B. Class Distribution & Support Across Splits
| Class Index | Raw Label | SCDB Issue Area Name | Train Count | Val Count | Test Count | Total Count | % of Corpus |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 0 | 1 | Criminal Procedure | 1,011 | 360 | 372 | 1,743 | 22.35% |
| 1 | 2 | Civil Rights | 811 | 218 | 222 | 1,251 | 16.04% |
| 2 | 3 | First Amendment | 423 | 108 | 88 | 619 | 7.94% |
| 3 | 4 | Due Process | 193 | 70 | 51 | 314 | 4.03% |
| 4 | 5 | Privacy | 45 | 22 | 28 | 95 | 1.22% |
| 5 | 6 | Attorneys | 35 | 35 | 17 | 87 | 1.12% |
| 6 | 7 | Unions | 255 | 51 | 24 | 330 | 4.23% |
| 7 | 8 | Economic Activity | 1,043 | 226 | 260 | 1,529 | 19.60% |
| 8 | 9 | Judicial Power | 717 | 165 | 200 | 1,082 | 13.87% |
| 9 | 10 | Federalism | 191 | 83 | 83 | 357 | 4.58% |
| 10 | 11 | Interstate Relations| 53 | 14 | 15 | 82 | 1.05% |
| 11 | 12 | Federal Taxation | 220 | 38 | 37 | 295 | 3.78% |
| 12 | 13 | Miscellaneous | 3 | 10 | 3 | 16 | 0.21% |

### C. Majority-Class Baseline on Test Split
- **Training Set Majority Class**: Class 7 (*Economic Activity*), comprising 1,043 of 5,000 training cases (20.86%).
- **Evaluated on Test Split (N=1,400)**:
  - **Accuracy**: **18.57%** (0.1857; 260 correct out of 1,400)
  - **Macro F1**: **0.0241**
  - **Weighted F1**: **0.0582**
- *(For reference: if the test-set empirical mode Criminal Procedure were known beforehand, the accuracy would be 26.57%; however, true machine learning requires predicting using the majority class observed on training data).*

### D. Primary Metric Adoption & Support Warnings
- **Primary Metric**: **Macro-averaged F1 Score**.
- **Critical Support Caution**: Three minority classes have critically small test support:
  - **Miscellaneous**: 3 decisions
  - **Interstate Relations**: 15 decisions
  - **Attorneys**: 17 decisions
- A single classification mistake in these categories causes substantial swings in per-class F1. All reports and presentations must display per-class support alongside macro F1.

### E. Split Leakage & Duplicate Audit
- **Internal duplicates**: Train: 0, Validation: 0, Test: 0.
- **Cross-split duplicates**: Train $\cap$ Validation = 0, Train $\cap$ Test = 0, Validation $\cap$ Test = 0.
- **Verdict**: Zero text leakage across splits.

---

## 5. Measured Token Statistics & Opening Text Observations

### A. Subword Token Statistics (`nlpaueb/legal-bert-base-uncased`, N=500 random train decisions; measured by `scripts/01_run_eda.py`)
- **Minimum**: 84 tokens
- **25th Percentile**: 2,334.0 tokens
- **Median**: **5,470.5 tokens**
- **75th Percentile**: 10,565.2 tokens
- **Maximum**: 42,884 tokens
- **Mean $\pm$ Std**: $7,308.5 \pm 6,537.3$ tokens
- **% of Documents Exceeding 512 Tokens**: **93.00%**
- **Mean Fraction of Document Covered by First 512 Tokens**: **20.48%**

### B. What the Opening 300 Characters Actually Contain
Direct inspection of raw decision openings reveals that the initial text does not typically begin with substantive legal holding or factual narrative. Instead, it begins with formal court metadata:
1. **Reporter Citations & Docket**: E.g., `"348 U.S. 540 75 S.Ct. 509 99 L.Ed. 624 Louis SHOMBERG, Petitioner, v. UNITED STATES of America. No. 48."`
2. **Procedural Dates**: E.g., `"Argued March 1, 1955. Decided April 4, 1955."`
3. **Counsel Appearances**: E.g., `"Mr. Alan Y. Cole, Washington, D.C., for petitioner. Mr. Gray Thoron, Washington, D.C., for respondent."`
4. **Delivery of the Opinion**: E.g., `"Mr. Justice CLARK delivered the opinion of the Court."`
5. **Omission Notes**: Some opinions explicitly note `"[Syllabus from pages 591-593 intentionally omitted]"`.
- **Honest Handling**: Truncating to 512 tokens exposes the Transformer primarily to the caption, procedural dates, counsel lists, and opening paragraphs of the opinion. We will state this clearly in all project reports and presentations.

### C. Shortest Documents in the Corpus
- **Rank 1**: Train idx 4185 (19 words, 110 chars, *Criminal Procedure*): `"433 U.S. 682 97 S.Ct. 2912 53 L.Ed.2d 1054 Thomas Leon HARRISv.State of OKLAHOMA. No. 76-5663. June 29, 1977."`
- **Rank 2**: Validation idx 486 (38 words, 239 chars, *Unions*)
- **Rank 3**: Validation idx 1384 (39 words, 238 chars, *Judicial Power*): Per Curiam dismissal (`"The writ of certiorari is dismissed as improvidently granted."`)
- **Rank 4**: Test idx 1201 (39 words, 244 chars, *Judicial Power*): Per Curiam dismissal
- **Rank 5**: Validation idx 821 (45 words, 274 chars, *Judicial Power*)
- **Decision on Shortest Documents**: These are valid procedural memorandum orders and Per Curiam dispositions within the SCDB. They are preserved intact to strictly adhere to the official LexGLUE benchmark split.
