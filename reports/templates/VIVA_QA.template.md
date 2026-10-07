# Viva preparation: Classification of Legal Text

Answers are about **our** implementation. Every number is generated from
`results/` by `scripts/make_viva_qa.py` (edit `reports/templates/VIVA_QA.template.md`,
not the generated file). When an answer is a hypothesis, it says so.

## 1. The project in one minute

We classify US Supreme Court opinions into {{n_classes}} SCDB issue areas using
LexGLUE SCOTUS ({{n_total}} opinions; official chronological splits of
{{n_train}} / {{n_val}} / {{n_test}}). We compare a majority baseline, TF-IDF with
LinearSVC and with logistic regression, the reference paper's LDA + LR and
Doc2Vec + LR, and Legal-BERT fine-tuned on the first {{bert_max_len}} tokens.
The primary metric is macro F1. Best: TF-IDF + LinearSVC, test macro F1
{{svc_test_f1}} (95% CI {{svc_ci_lo}}–{{svc_ci_hi}}). Legal-BERT: {{bert_test_f1}}
({{bert_ci_lo}}–{{bert_ci_hi}}). LDA: {{lda_test_f1}}; Doc2Vec: {{d2v_test_f1}}.

## 2. Data and splits

**Q: Where does the data come from and what does one example look like?**
A: `{{dataset_name}}`, config `{{dataset_config}}`, from the Hugging Face Hub. Each
row has `text` (the full opinion, starting with the citation line, caption,
docket number, argument and decision dates, often a syllabus, then the opinion)
and `label` (an integer 0–12 that `src/data.py` maps to the SCDB issue-area name).

**Q: How was the data split? Did you stratify?**
A: We use the official LexGLUE splits, which are chronological: train
{{train_years}}, validation {{val_years}}, test {{test_years}} (LexGLUE paper,
Section 3; years also in `results/dataset_info.json`). We did **not** re-split
or stratify, for two reasons: the official split keeps results comparable with
the benchmark, and a time-based split is the realistic setting (train on the
past, predict the future). The cost is that class proportions differ between
splits, e.g. Miscellaneous has {{train_count_miscellaneous}} training,
{{val_count_miscellaneous}} validation and {{test_support_miscellaneous}} test
documents. The only stratified sample is the demo/test fixture
`data/sample_cases.json`: one seeded test case per class. The bootstrap is a
plain (not stratified) resample of the test set.

**Q: What evidence is there of distribution shift?**
A: The training majority class is {{majority_class}}, but the test majority class
is {{test_majority_class}}. Predicting the training majority gives test accuracy
{{majority_test_acc}}; always predicting the test majority would give
{{test_majority_acc}}. Every trained model also drops from validation to test
(LinearSVC by {{svc_drop}} macro F1, Legal-BERT by {{bert_drop}}). This is
consistent with temporal drift, but we did not test that cause.

**Q: How imbalanced is it?**
A: {{largest_class}} has {{largest_count}} training documents and
{{smallest_class}} {{smallest_count}}: a ratio of {{imbalance_ratio}}:1. In the test
split the rarest classes are {{rare_desc}}.

**Q: How long are the documents?**
A: Median words: {{words_median_train}} (train), {{words_median_val}} (validation),
{{words_median_test}} (test). On a {{tok_sample}}-document training sample, the
Legal-BERT tokenizer gives a median of about {{tok_median}} tokens;
{{tok_pct_over_512}}% exceed 512, and the first 512 tokens cover about
{{tok_pct_covered}}% of a document on average (`results/eda_stats.json`).

**Q: Did you check data quality?**
A: Yes, in `src/data.py`: {{null_texts}} empty texts, {{internal_dups}} exact
duplicates inside splits and {{cross_split_dups}} across splits, using SHA-256
hashes of every text.

## 3. Data leakage and how we prevented it

**Q: What is data leakage?** Information from the evaluation data reaching
training or model selection, which makes scores optimistic.

**Q: What did you do about it?**
1. *Duplicates across splits:* none (SHA-256 audit).
2. *Fitting preprocessing on evaluation data:* every vectorizer (vocabulary,
   document frequencies, IDF), LDA model and Doc2Vec model is fitted on the
   training split only; validation and test are only transformed / inferred.
   `fit_on_train` and `transform_split` in `src/preprocess.py` enforce the
   pattern, and `tests/test_leakage.py` checks that a word that appears only in
   validation never enters the vocabulary.
3. *Tuning on test:* all hyperparameters (C, alpha, K, Legal-BERT's best epoch)
   are chosen on validation macro F1; final models are fitted on train only and
   the test split is predicted once per model.
4. *Time:* the chronological split means no future cases are seen in training.
5. *Shortcut features:* standalone numbers (citations, docket numbers, years)
   are removed because they identify the era rather than the topic.

**Q: Is choosing the best Legal-BERT epoch on validation leakage?** No. That is
exactly what the validation split is for. It does make validation scores
optimistic, which is why we report the test split.

**Q: Your demo shows test cases. Is that a problem?** No training or tuning uses
them; the demo only displays predictions.

## 4. Preprocessing and TF-IDF

**Q: What preprocessing did you do?** NFKC Unicode normalisation, lower-casing,
whitespace collapsing, removal of standalone numbers (`\b\d+\b`, so `12th` and
`1st` stay), and scikit-learn's English stop words minus legal words that change
meaning (*not, no, nor, never, shall, may, might, must, against, without,
except, unless, cannot, none, neither, nothing, nobody, nowhere*). No stemming.

**Q: Why keep negations and modals?** "The statute shall apply" and "the statute
shall not apply" differ only in *not*; in law that is the whole point. With
bigrams, phrases such as *not apply* can become features.

**Q: Did removing numbers help?** A validation-only check with the earlier saga
model gave macro F1 {{abl_drop_f1}} with numbers removed and {{abl_keep_f1}} with
them kept (`results/numeric_token_ablation_validation.csv`). That model was
under-converged, so treat it as a rough indication only.

**Q: Explain TF-IDF.** Each document becomes a sparse vector over the vocabulary.
Term frequency counts how often a term occurs in the document; with
`sublinear_tf=True` it becomes 1 + log(tf), so each repeat of a word adds less
than the one before. Inverse document frequency, in scikit-learn's smoothed
form log((1 + n) / (1 + df)) + 1, down-weights terms that occur in many
documents. The product is L2-normalised per document so long and short
opinions are comparable.

**Q: Your settings?** Unigrams and bigrams, `min_df={{tfidf_min_df}}` (drop terms in
fewer than two documents), `max_df={{tfidf_max_df}}` (drop terms in almost every
document), and the {{tfidf_max_features}} most frequent terms. A separate
benchmark of a {{bench_features}}-feature vectorizer took {{bench_seconds}} s and
{{bench_gb}} GB peak memory on the laptop; the models use the smaller vocabulary
to keep training feasible on CPU.

**Q: Why not stemming?** It can merge legally distinct words, and we wanted a
baseline without that change. It is untested either way.

## 5. Tokenization and embeddings

**Q: What tokenization does each model use?** TF-IDF, LDA and Doc2Vec use word
tokens (scikit-learn's default regex, or whitespace split after normalisation for
Doc2Vec). Legal-BERT uses its own uncased WordPiece tokenizer: unknown words are
split into known sub-word pieces (e.g. a rare word becomes a stem plus `##`
pieces). `[CLS]` is added at the start and `[SEP]` at the end, so {{bert_content_tokens}} content
tokens fit in the {{bert_max_len}} limit; shorter inputs are padded and masked
with the attention mask.

**Q: What is an embedding?** A dense, learned vector for a discrete item.
Doc2Vec learns a {{d2v_vector_size}}-dimensional vector for every document. In
BERT, each token's input is the sum of a token embedding, a position embedding
and a segment embedding; the layers then turn these into contextual embeddings,
so the same word gets different vectors in different sentences. TF-IDF vectors
are not embeddings: they are sparse and not learned.

## 6. The models

**Q: Logistic regression: how does it work and how did you train it?**
A: It models class probabilities from a linear score passed through a sigmoid
(binary) or softmax (multinomial), and is trained by minimising log loss (cross
entropy) plus a penalty. Our final LR is `SGDClassifier(loss="log_loss",
penalty="l2")`: the same model trained by stochastic gradient descent, one
binary classifier per class (one-vs-rest), with probabilities normalised across
classes. Alpha, the L2 strength, was {{lr_alpha}}, chosen from
{{lr_alpha_grid}} on validation. It converged after {{lr_n_iter}} of
{{lr_max_iter}} epochs (tol {{lr_tol}}); "converged" means `n_iter_ < max_iter`,
i.e. it stopped because the loss stopped improving, not because it ran out of
epochs (`results/converged_lr_selection.json`).

**Q: Why was the first (saga) LR under-converged?**
A: It was `LogisticRegression(solver="saga")` with `max_iter={{saga_max_iter}}` and
`tol={{saga_tol}}`. Saga is an incremental gradient method that needs many passes
over sparse, high-dimensional TF-IDF data; five passes and a very loose
tolerance stopped it long before the optimum. Its validation macro F1 was
{{saga_val_f1}} against {{lr_val_f1}} for the converged SGD model, and test
{{saga_test_f1}} against {{lr_test_f1}}. We kept it, clearly labelled, so the
history is transparent; it is not a final model.

**Q: LinearSVC and the squared hinge loss?**
A: A linear SVM finds weights that separate classes with a large margin.
scikit-learn's `LinearSVC` (liblinear) trains one-vs-rest and minimises
½‖w‖² + C Σ max(0, 1 − yᵢ w·xᵢ)²: the squared hinge loss penalises points inside
the margin quadratically, which makes the objective differentiable and works
well on sparse text. C (inverse regularisation) = {{svc_c}}, chosen from
{{svc_c_grid}}. It outputs margin scores, not probabilities; we did not
calibrate them.

**Q: Why L2 and not Lasso (L1)?** L2 spreads weight across correlated n-grams
(e.g. synonyms and overlapping bigrams), which suits text; L1 would pick a
sparse subset and could be more interpretable. We did not try L1; it is listed
as future work. There is no feature selection beyond the vocabulary cap.

**Q: Explain LDA.** Latent Dirichlet Allocation is a generative topic model:
each document has a mixture of K topics drawn from a Dirichlet distribution,
each topic is a distribution over words, and each word is generated by picking
a topic from the document's mixture and then a word from that topic. Fitting
inverts this to find the topics and each document's topic proportions. We use
the K-dimensional topic proportions as features for logistic regression.

**Q: Your LDA setup and why counts?** scikit-learn's variational-Bayes LDA on raw
word counts: LDA's likelihood is defined over integer word counts, so TF-IDF
weights do not fit the model (the paper used TF-IDF; we record the
difference). Vocabulary {{lda_count_features}} terms (min_df {{lda_min_df}}),
{{lda_max_iter}} batch iterations, K from {{lda_grid}} by validation macro F1:
{{lda_val_f1_k10}}, {{lda_val_f1_k20}}, {{lda_val_f1_k30}}, {{lda_val_f1_k40}}.
K = {{lda_k}} won, at the edge of the grid, so a larger K might do better.

**Q: Explain Doc2Vec.** Paragraph Vectors (PV-DM) learn a vector per document
that, combined with the vectors of the surrounding words, is trained to predict a word
from its context; the
document vector ends up summarising its content. We trained it on the training
split only ({{d2v_vector_size}} dimensions, window {{d2v_window}}, min_count
{{d2v_min_count}}, {{d2v_epochs}} epochs, first {{d2v_max_tokens}} tokens per
document). For validation and test documents `infer_vector` freezes the word
weights and fits only a new document vector, so no evaluation text changes the
model. PV-DBOW is the other variant; we did not use it.

**Q: Why are LDA and Doc2Vec so much worse?** Partly the method: a K-topic or
{{d2v_vector_size}}-dimensional summary is a much coarser representation than
thousands of TF-IDF weights. Mostly, we think, the reduced CPU configurations
(very few iterations/epochs, small LDA vocabulary, truncated Doc2Vec input).
We cannot separate these two causes with our runs.

**Q: Explain BERT and Legal-BERT.** BERT is a stack of Transformer encoder layers
pretrained on unlabelled text with masked-language modelling. Legal-BERT
(`{{bert_model}}`) is BERT pretrained on legal text (Chalkidis et al., 2020). For
classification we take the final `[CLS]` representation, pass it through the
pooler and a new linear layer with {{n_classes}} outputs, apply softmax, and
fine-tune the whole network with cross-entropy.

**Q: Explain attention.** Each token builds a query, key and value vector.
Attention weights are softmax(QKᵀ / √dₖ), so each token's new representation is
a weighted average of all tokens' values, weighted by how relevant they are.
Multi-head attention does this several times in parallel with different
projections. Every token attends to every other token, so cost grows with the
square of the sequence length, and BERT's position embeddings stop at
{{bert_max_len}}: hence the input limit.

**Q: Your Legal-BERT training setup?** {{bert_epochs}} epochs; batch
{{bert_batch}} with gradient accumulation {{bert_accum}} (effective batch
{{bert_eff_batch}}: gradients from two batches are summed before one optimiser
step, to fit GPU memory); AdamW, learning rate {{bert_lr}}, weight decay
{{bert_weight_decay}}; linear warm-up over the first {{bert_warmup_pct}}% of steps,
then linear decay; mixed precision (fp16 autocast with gradient scaling) on a
Colab {{bert_gpu}}; plain cross-entropy (no class weights); seed {{bert_seed}};
the best epoch by validation macro F1 is kept. Training took {{bert_minutes}} minutes.

**Q: Why three Legal-BERT numbers?** Three full runs with identical settings gave
test macro F1 {{bert_first_run_f1}} (first run; files overwritten, value in
`docs/RUN_LOG.md`), {{bert_reported_f1_4}} (reported run,
`reports/colab_run_log.ipynb`) and {{bert_run2_f1_4}} (rerun,
`results/transformer_run2_info.json`). GPU non-determinism makes runs differ even
with a fixed seed. We report the {{bert_reported_f1_4}} run, did not pick the best,
and quote the range as run-to-run variation. The demo uses the rerun's weights.

## 7. Evaluation

**Q: Precision, recall, F1?** For a class c: precision = TP / (TP + FP), the share
of predicted-c documents that really are c; recall = TP / (TP + FN), the share of
real c documents found; F1 = 2PR / (P + R), their harmonic mean, high only if both
are high. A class that is never predicted gets precision 0 (we set
`zero_division=0`) and therefore F1 0.

**Q: Macro vs weighted F1?** Macro F1 is the plain mean of the {{n_classes}}
per-class F1 scores: every issue area counts equally. Weighted F1 weights each
class by its support, so it is dominated by the big classes and behaves like
accuracy. We use macro F1 because the rare areas matter. Example: Legal-BERT's
accuracy ({{bert_test_acc}}) is close to LinearSVC's ({{svc_test_acc}}), but its
macro F1 is much lower ({{bert_test_f1}} vs {{svc_test_f1}}) because it never
correctly predicts {{bert_zero_f1}}.

**Q: Why include a majority baseline?** It shows what "doing nothing" scores:
always predicting {{majority_class}} gives accuracy {{majority_test_acc}} but macro
F1 {{majority_test_f1}}, which shows why accuracy alone misleads on imbalanced data.

**Q: How do you read the confusion matrix?** Rows are true classes, columns are
predicted classes. Our normalised matrices divide each row by its total, so the
diagonal is per-class recall and off-diagonal cells show where a class's
documents go. In counts, the most frequent confusion is {{ea_top_pair_sentence}}.
In Legal-BERT's row-normalised matrix, most Attorneys cases go to Judicial Power
and most Federalism cases to Economic Activity.

**Q: How did you handle class imbalance?** We measured it with macro F1 and
always report per-class support, but did not rebalance: no class weights, no
resampling. That is a limitation; class-weighted or focal loss is the first
thing we would try, especially for Legal-BERT.

**Q: How did you compute the confidence intervals?** Percentile bootstrap: draw
{{ci_resamples}} resamples of the test set with replacement (seed {{ci_seed}}),
recompute macro F1 on each, and take the {{ci_lower_pct}}th and {{ci_upper_pct}}th percentiles. They
capture test-set sampling only, not training randomness (seeds, runs), and each
model is resampled independently (unpaired).

**Q: Is LinearSVC really better than Legal-BERT? Than LR?** LinearSVC
[{{svc_ci_lo}}, {{svc_ci_hi}}] and Legal-BERT [{{bert_ci_lo}}, {{bert_ci_hi}}]
{{svc_bert_ci_overlap}}, so the gap is not explained by test sampling. LinearSVC and
LR [{{lr_ci_lo}}, {{lr_ci_hi}}] {{svc_lr_ci_overlap}}, so we do not rank them.
Non-overlap of unpaired intervals is strong evidence; overlap does not prove
"no difference". A paired test (paired bootstrap on the same resamples, or
McNemar's test on the cases where exactly one model is right: {{ea_only_svc}} vs
{{ea_only_bert}} for LinearSVC vs Legal-BERT) would be the proper comparison; we
did not run it.

## 8. Results and interpretation

**Q: Why did linear models beat Legal-BERT?** Hypotheses, not tested:
1. *Truncation.* Legal-BERT sees about {{tok_pct_covered}}% of a typical opinion,
   and the opening is largely the citation line, caption, docket and counsel
   names. TF-IDF sees the whole document. (Counterpoint: many openings include a
   syllabus that summarises the case, so the opening is not useless.)
2. *No class weighting* plus very small classes: Legal-BERT never correctly
   predicts {{bert_zero_f1}}, while LinearSVC reaches F1 {{svc_attorneys_f1}} on
   Attorneys. Macro F1 punishes this heavily.
3. *Limited fine-tuning:* {{n_train}} training documents, {{bert_epochs}} epochs,
   one configuration, and visible run-to-run variation.
The length analysis does not show a strong effect: Legal-BERT's accuracy by
length quartile ranges {{ea_bert_q_min}}–{{ea_bert_q_max}} (LinearSVC
{{ea_svc_q_min}}–{{ea_svc_q_max}}), so "longer documents hurt Legal-BERT much more"
is not supported by our numbers.

**Q: Which classes are hardest?** Federalism (Legal-BERT recall
{{bert_federalism_rpct}}%, LinearSVC F1 {{svc_federalism_f1}}), which is mostly
predicted as Economic Activity; Interstate Relations for Legal-BERT (recall
{{bert_interstate_relations_rpct}}%); and the tiny classes. Judicial Power is
confused with Economic Activity, Criminal Procedure and Civil Rights: these
cases are often about procedure in a case whose subject is something else.

**Q: Do the models make the same mistakes?** No: {{ea_only_svc}} test cases are
right only for LinearSVC, {{ea_only_bert}} only for Legal-BERT, {{ea_both_wrong}}
wrong for both. An ensemble might help (untested).

**Q: How do your results compare with the reference paper?** Only
qualitatively. The paper used a different dataset (a `textacy` scrape), a split it
does not fully specify, different labels, and its Table 1 column labels
contradict its own text; it has no BERT result. So we do not compare numbers
(`docs/reference_notes.md`).

## 9. Long documents

**Q: How does each model handle length?** TF-IDF and LDA use the whole document
(bag of words, any length). Doc2Vec uses the first {{d2v_max_tokens}} tokens
(CPU limit). Legal-BERT uses the first {{bert_max_len}} tokens.

**Q: What else could you do?** Head + tail truncation (keep the start and the
end); split into chunks, encode each, and pool the chunk vectors or predictions;
hierarchical models that encode paragraphs and combine them (LexGLUE does this
for SCOTUS); long-input Transformers such as Longformer or BigBird with sparse
attention; or strip the caption and counsel block before truncating.

## 10. Limitations and future work

- Legal-BERT read only the first {{bert_max_len}} tokens (about {{tok_pct_covered}}%).
- LDA and Doc2Vec ran in reduced CPU configurations; K = {{lda_k}} is at the grid edge.
- One seed per model; Legal-BERT ranged {{bert_run_min_4}}–{{bert_run_max_4}}.
- Chronological split: the test period is later; all trained models drop on test.
- Rare classes: test support {{rare_supports}}, so macro F1 is noisy.
- Unpaired bootstrap CIs capture test sampling only.
- No class weighting; small hyperparameter grids.
- Demo artifacts: the demo LinearSVC uses C = {{demo_svc_c}} (reported: {{svc_c}}) and
  agrees with the reported predictions on {{demo_svc_agree}}% of test documents
  (`results/demo_artifact_check.json`); the demo Legal-BERT is from the rerun.

Future work: hierarchical or long-input Transformers; class-weighted/focal loss;
several seeds with paired tests; full-size LDA/Doc2Vec and a wider K grid; L1 or
elastic-net TF-IDF models; removing the header block; an ensemble of LinearSVC
and Legal-BERT.

## 11. The demo

**Q: What does the app show?** `streamlit run app.py`. Classify tab: a sample test
case (first {{fixture_chars}} characters of each fixture opinion) or pasted text;
live predictions with top-5 scores from LinearSVC (margin scores), LR and
Legal-BERT (probabilities); a warning showing how much text Legal-BERT read; and
the stored full-document predictions of every model. Results tab: metrics, CIs,
confusion matrices and per-class scores from `results/`.

**Q: Why can live and stored predictions differ?** (1) The sample text is a
{{fixture_chars}}-character excerpt, while stored predictions used the full
opinion. (2) The demo Legal-BERT weights are from the rerun. (3) The demo
LinearSVC artifact uses C = {{demo_svc_c}} instead of {{svc_c}}; run over the test set
it would score macro F1 {{demo_svc_f1}}. The demo LR matches the reported model's
setting and agrees with its predictions on {{demo_lr_agree}}% of test documents.
(4) The `.joblib` files were saved with scikit-learn {{demo_pickle_sklearn}}.

**Q: What if the artifacts are missing?** The app still starts, marks missing
models, and the Results tab works (`tests/test_app.py` checks this).

## 12. Walk me through the code

**`src/data.py`.** `load_scotus_dataset()` loads LexGLUE SCOTUS with `datasets`.
`SCDB_LABEL_NAMES` maps label ids 0–12 to issue-area names. `validate_schema()`
checks the splits, columns and empty texts; `check_duplicates()` hashes every text
with SHA-256 and counts duplicates within and across splits;
`compute_class_distributions()` counts labels per split;
`save_stratified_sample_fixtures()` writes one seeded test case per class to
`data/sample_cases.json`.

**`src/preprocess.py`.** `normalize_text()` does NFKC, lower case, number removal
and whitespace collapsing. `LEGAL_STOP_WORDS` is scikit-learn's list minus
`PRESERVED_LEGAL_TERMS`. `PreprocessingConfig` holds vectorizer settings;
`build_tfidf_vectorizer()` / `build_count_vectorizer()` create unfitted
vectorizers that call `normalize_text` as their preprocessor. `fit_on_train()`
fits on training texts only; `transform_split()` refuses an unfitted vectorizer.

**`src/evaluate.py`.** `evaluate_predictions()` returns accuracy, macro and
weighted precision/recall/F1, per-class scores with support, and the confusion
matrix, using all {{n_classes}} labels even if some are never predicted.
`save_confusion_matrix()` draws it; `metrics_row()` makes one `metrics.csv` row.

**`src/utils.py`.** `set_seed()` seeds Python, NumPy and PyTorch;
`detect_hardware()` records CPU/GPU/RAM; JSON helpers.

**`src/predict.py`.** Maps Hugging Face artifact paths to local paths and
downloads them with `hf_hub_download`, reading the token from the environment.

**`src/demo.py`.** Everything the app needs, testable without Streamlit: finding
artifacts, loading the TF-IDF models and Legal-BERT (config from the hub, weights
from `models/legal_bert_state.pt`), predicting, the truncation note, the
hyperparameter-mismatch note, and reading stored predictions and metrics from
`results/`.

**`src/report_facts.py`.** Reads every results file once and produces formatted
values (`facts()`), the results rows and `render()` for the double-brace placeholders in templates.
Settings not stored in `results/` are listed in `CONFIG` with the source file and
literal that proves them.

**`scripts/01_run_eda.py`.** Runs the data checks, class distributions, word
lengths, Legal-BERT token lengths on a seeded {{tok_sample}}-document sample, the
majority baseline and the EDA figures; writes `dataset_info.json` and `eda_stats.json`.

**`scripts/benchmark_preprocessing.py`.** Times a full-train TF-IDF fit and samples
peak memory in a background thread; falls back to unigrams if limits are exceeded.

**`scripts/train_classical.py`.** Fits the TF-IDF vectorizer on train, tunes the
saga LR and LinearSVC on validation, predicts test once, saves prediction archives,
confusion matrices, `classical_selection.json` and the numeric-token ablation
(validation only).

**`scripts/train_converged_lr.py`.** Same TF-IDF features; tunes
`SGDClassifier(loss="log_loss")` alpha on validation, records `n_iter_` and
convergence, predicts test once.

**`scripts/train_topic.py`.** Count vectorizer on train; LDA sweep over K with LR
on topic proportions; Doc2Vec on train with inferred validation/test vectors and
LR; saves predictions and `topic_selection.json`.

**`scripts/04_train_transformer.py`.** `ScotusDataset` tokenizes with truncation to
{{bert_max_len}}; the training loop uses AdamW, linear warm-up, gradient
accumulation and mixed precision on CUDA; it evaluates on validation each epoch,
keeps the best checkpoint by macro F1, then predicts validation and test and
writes `transformer_predictions.npz` and `transformer_run_info.json`. It refuses
to overwrite full results without `--overwrite` and has a CPU `--smoke_test`.

**`scripts/evaluate_all.py`.** Loads every prediction archive, adds the majority
baseline, recomputes all metrics, per-class scores, bootstrap CIs, confusion
matrices, the comparison chart and `error_analysis.md`. No model is retrained.

**`scripts/make_demo_artifacts.py`, `download_artifacts.py`,
`upload_artifacts.py`, `check_demo_artifacts.py`.** Build the demo TF-IDF models
on CPU; download / upload them from / to Hugging Face; compare the demo artifacts
with the reported test predictions.

**`app.py`.** The Streamlit UI: sidebar artifact status, Classify tab, Results tab,
About tab; models are cached with `st.cache_resource`.

**`reports/writeup.py`, `scripts/make_slides.py`, `scripts/make_viva_qa.py`,
`scripts/make_readme_results.py`.** Generate the PDF (fails above two pages), the
deck (fails unless 15 slides with notes), this file and the README's generated
sections, all from `results/`.

**`scripts/audit_numbers.py` and `run_all.py`.** The audit extracts every number
from the README, PDF, slides (including notes) and this file and checks it
against the values in `results/` and the documented settings; `run_all.py`
rebuilds all documents, runs the audit and prints PASS/FAIL per step.

**`tests/`.** Preprocessing and leakage tests; hand-calculated metric tests;
results-integrity tests that recompute `metrics.csv` and `per_class_f1.csv` from
the saved predictions; data and fixture checks; small-sample model fits;
demo-helper tests; headless Streamlit smoke tests.

## 13. Team and process

**Q: Who did what?** See `reports/CONTRIBUTIONS.md` and `git shortlog -sne`.

**Q: Did you use AI tools?** Yes, AI coding assistants helped draft code, tests
and documents; commits made with an assistant carry a `Co-Authored-By` trailer.
We reviewed the output, and every number is generated from `results/` and
checked by the audit script, so nothing was typed in by hand.
