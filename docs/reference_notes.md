# Reference Paper Notes & Critical Analysis

## Reference Paper Details
- **Title**: Classification of Legal Text
- **Author**: Krithika Iyer (`ksiyer`)
- **Institution / Course**: Stanford University, CS229: Machine Learning (Spring 2020)
- **Report Link**: `https://cs229.stanford.edu/proj2020spr/report/Iyer.pdf`
- **Local Copy**: `docs/references/Iyer.pdf`

---

## 1. Problem Formulation & Dataset
- **Task**: Multi-class classification of US Supreme Court decisions into expert-annotated legal issue areas and issue codes based on the Supreme Court Database (SCDB at Washington University in St. Louis).
- **Dataset Source in Paper**: Extracted via the `textacy` Python package (~8,200 decisions between 1791 and 2018 terms).
- **Labels in Paper**:
  - `issue_area`: ~15 high-level thematic categories described in paper (e.g., Criminal Procedure, Civil Rights, First Amendment, Due Process, Privacy, Economic Activity, Judicial Power, Federalism, Federal Taxation).
  - `issue_code`: ~279 fine-grained specific legal issues described in paper.

---

## 2. Methodology & Modeling Approaches in Paper
The paper evaluated three distinct technical approaches:
1. **Unsupervised Topic Modeling + Classifier (LDA + Logistic Regression)**:
   - Extracted latent topics across decisions using Latent Dirichlet Allocation (via `gensim`) on a TF-IDF matrix.
   - Used topic mixture distributions as feature vectors for a Logistic Regression classifier.
2. **Document Vector Embeddings (Doc2Vec + Logistic Regression)**:
   - Trained a Paragraph Vector / Doc2Vec model (`gensim`) over legal opinions to produce fixed-size dense representations.
   - Evaluated Doc2Vec vectors using Logistic Regression.
   - Also explored Doc2Vec vectors as a semantic search index (e.g., querying cases similar to *Roe v. Wade*).
3. **Transformer Neural Networks (BERT via Hugging Face)**:
   - Intended to fine-tune a BERT-based transformer model (exact checkpoint not specified in the paper).
   - Result: Model ran out of memory (OOM) on Google Colaboratory (even on paid high-RAM instances) due to document lengths. **No BERT results were reported in the paper.**

---

## 3. Reported Results & Critical Inconsistencies in Paper
### Table 1 Discrepancy
In Section *Results and Discussion*, the paper reports:
> "The classification accuracy for the 279 issue_code is around 0.133. When only the issue_areas (15 categories) are considered, the classification accuracy jumps to 0.47."

However, Table 1 in the paper is printed as:
| Model | 15-Labels | 279 Labels |
| :--- | :--- | :--- |
| LDA + LR | 0.13 | 0.47 |
| Doc2Vec + LR | 0.48 | 0.63 |

**Critical Flaw**:
The column headers in Table 1 are inverted relative to the paper's own prose. Empirically and statistically, predicting 15 broad classes is substantially easier than 279 fine-grained classes. The text indicates LDA achieved 47% on 15 classes and 13.3% on 279 classes, whereas the table displays the opposite. Furthermore, Doc2Vec is listed as achieving 48% on 15-labels and 63% on 279-labels, which similarly indicates inconsistent reporting.

### Table 2 Reported Per-Class Metrics in Paper (Doc2Vec + LR, 15 Categories)
The paper presents per-class Precision, Recall, and F1 for 15 classes (ranging from 0.00 for rare classes like 'None', 'Miscellaneous', 'Private Action' to 0.79 for 'Criminal Procedure' and 'Federal Taxation'). Support counts show extreme class imbalance (e.g., Criminal Procedure has 302 test docs, while Private Action has 0 test docs).

---

## 4. Reproducibility Assessment
| Aspect | Status | Reason / Detail |
| :--- | :--- | :--- |
| **Exact Dataset Split** | Not Reproducible | No fixed random seed, split ratios, or record identifiers provided in the paper. |
| **Hyperparameters** | Partially Reproducible | Number of LDA topics, alpha, beta, Doc2Vec vector dimensions, window size, training epochs, and LR regularization parameters were omitted in the text. |
| **Code Availability** | Not Verified | Google Colab links in Appendix A are personal drive links listed in the text; not verified or tested. |
| **Transformer Benchmark**| Not Reproducible | The paper experienced complete OOM failure and published no metrics. |
| **Overall Reproducibility**| Low | Core conceptual pipeline can be reproduced, but exact figures cannot and should not be matched. |

---

## 5. Our Project Strategy & Planned Extensions
To create a rigorous, university-level, defensible project:

| Component | Paper Method | Our Reproduction & Planned Extension | Rationale |
| :--- | :--- | :--- | :--- |
| **Dataset** | `textacy` scrape (~8,200 docs) | LexGLUE SCOTUS (`coastalcph/lex_glue`) [VERIFIED in `results/dataset_info.json`] | Public benchmark with standardized train/validation/test splits and the duplicate audit recorded in `results/eda_stats.json`. |
| **Baselines** | None (jumped straight to LDA) | TF-IDF + Logistic Regression, TF-IDF + LinearSVC [PLANNED] | Essential classical ML baselines required for sound empirical comparison. |
| **Topic Modeling** | LDA (gensim) + LR | LDA (sklearn/gensim) + LR with topic sweep [PLANNED] | Systematically sweeps number of latent topics ($K \in \{10, 20, 30, 40\}$) tuned on validation set. |
| **Document Vectors** | Doc2Vec (gensim) + LR | Doc2Vec (`gensim.models.Doc2Vec`) + LR [PLANNED] | Rigorous split hygiene: Doc2Vec trained strictly on train split; vectors inferred for val/test. |
| **Transformer** | Failed (OOM, checkpoint not specified) | `nlpaueb/legal-bert-base-uncased` fine-tuning [PLANNED] | Only the 20-sample CPU smoke test is recorded; full GPU training remains planned. |
| **Evaluation** | Accuracy + partial F1 table | Unified evaluation pipeline [PLANNED] | Macro & weighted Precision/Recall/F1, Confusion Matrix, Error Analysis on test split. |
| **Interactive Demo**| None | Streamlit Application [PLANNED] | Real-time prediction, probability breakdown, sample opinion loader, token length warning. |
| **Unit Testing** | None | Pytest test suite [PLANNED] | Tests schema integrity, preprocessing, data leakage prevention, model training, evaluation logic. |
