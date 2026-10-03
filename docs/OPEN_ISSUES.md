# Open Issues and Project Ambiguities

This document records ambiguities, discrepancies, and verified constraints identified during requirement analysis and technical verification.

---

### Issue 1: Write-up Page Limit Contradiction in Official Guidelines
- **Source**: `docs/Guidelines.pdf`, Section 3.B (Page 1 vs Page 2).
- **Discrepancy**: The section heading explicitly reads:
  > **"B. One-Page Write-up"**
  However, the immediate descriptive bullet point states:
  > **"A concise, two-page summary of your project."**
- **Resolution Strategy**: 
  - Produce a publication-grade PDF report formatted cleanly with two columns that is **AT MOST 2 pages** (i.e. strictly $\le 2$ pages).
  - Verify page count programmatically during the build pipeline; any build resulting in $> 2$ pages will trigger a hard failure.
  - Keep the editable document source (`reports/writeup.py` generating PDF via `reportlab` or LaTeX/Word) alongside the generated `reports/writeup.pdf`.

---

### Issue 2: Inverted Column Labels in Reference Paper Table 1
- **Source**: Reference paper (*Classification of Legal Text*, Krithika Iyer, Stanford CS229 Spring 2020), Page 3, Table 1.
- **Discrepancy**:
  - Paper Section *Results and Discussion* states:
    > "The classification accuracy for the 279 issue_code is around 0.133. When only the issue_areas (15 categories) are considered, the classification accuracy jumps to 0.47."
  - Yet Table 1 ("Classification accuracy") displays:
    ```
    Model         15-Labels    279 Labels
    LDA + LR      0.13         0.47
    Doc2Vec + LR  0.48         0.63
    ```
  - The table column headers are evidently swapped/inverted relative to the text and statistical reality (predicting 279 fine-grained classes with LDA is much harder than 15 broad classes).
- **Resolution Strategy**:
  - Record the defect explicitly in `docs/reference_notes.md` and presentation slides.
  - Treat the reference paper's figures strictly as qualitative reference context, never as ground truth targets or benchmarks for our models.
  - All numbers in our project will originate strictly from our own reproducible execution and test evaluation.

---

### Issue 3: Incomplete Reference Experiment (BERT OOM)
- **Source**: Reference paper, Page 5, Section *Transformer based neural nets*.
- **Observation**: The paper's BERT experiment ran out of memory on Google Colaboratory (even on paid high-RAM instances), resulting in zero reported Transformer metrics. The specific checkpoint name is not specified in the paper.
- **Resolution Strategy**:
  - Acknowledge that the reference paper provides no Transformer baseline.
  - Implement a dedicated, working Transformer pipeline using `nlpaueb/legal-bert-base-uncased` (with standard `bert-base-uncased` fallback).
  - Honestly state token length constraints: standard BERT models accept a maximum of 512 tokens, whereas US Supreme Court opinions often span thousands of words. We explicitly document that the baseline Transformer evaluates the first 512 tokens.

---

### Issue 4: Dataset Candidate Verification (LexGLUE SCOTUS)
- **Source**: Candidate dataset proposed for reproduction and extension.
- **Current Status**: **UNVERIFIED, to be confirmed in Phase 3**.
- **Items to Verify**:
  - Exact split sizes (train, validation, test) are unverified until loaded in Phase 3.
  - Number of target classes and label schema alignment with SCDB issue areas are unverified until loaded in Phase 3.
  - Column schema, missing values, duplicates, and token length distribution must be verified programmatically and saved directly to `results/dataset_info.json`.
