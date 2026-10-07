"""Streamlit demo: classify a US Supreme Court opinion into one of 13 SCDB issue areas.

Run from the repository root:
    streamlit run app.py

Live models need the optional artifacts in models/ (see README). Without them
the app still runs and shows the stored results from results/.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src import demo
from src.data import SCDB_LABEL_NAMES


FIGURES = demo.RESULTS_DIR / "figures"

st.set_page_config(page_title="SCOTUS Issue-Area Classifier", layout="wide")


@st.cache_resource(show_spinner="Loading TF-IDF model...")
def get_sklearn(name):
    return demo.load_sklearn_model(name)


@st.cache_resource(show_spinner="Loading Legal-BERT (first run downloads the tokenizer)...")
def get_bert():
    return demo.load_bert()


@st.cache_data
def get_samples():
    return demo.load_sample_cases()


def show_prediction(name, result, extra_note=None):
    st.markdown(f"**{demo.MODEL_LABELS[name]}**")
    st.metric("Predicted issue area", SCDB_LABEL_NAMES[result["label"]])
    table = demo.top_k(result["scores"])
    if result["score_kind"] == "probability":
        st.dataframe(
            table,
            hide_index=True,
            column_config={"Score": st.column_config.ProgressColumn(
                "Probability", min_value=0.0, max_value=1.0, format="%.3f")},
        )
    else:
        st.dataframe(table, hide_index=True,
                     column_config={"Score": st.column_config.NumberColumn("Decision score", format="%.3f")})
        st.caption("LinearSVC gives margin scores, not probabilities. Higher means more likely.")
    if extra_note:
        st.caption(extra_note)


# ---------------------------------------------------------------- sidebar

available = demo.available_models()
with st.sidebar:
    st.header("Model artifacts")
    for name in demo.LIVE_MODELS:
        mark = "available" if name in available else "missing"
        st.write(f"{demo.MODEL_LABELS[name]}: **{mark}**")
    if len(available) < len(demo.LIVE_MODELS):
        st.info(
            "Missing models can be downloaded with "
            "`python scripts/download_artifacts.py`. "
            "Stored results in the other tabs work without them."
        )
    st.divider()
    st.caption(
        "Data: LexGLUE SCOTUS (coastalcph/lex_glue, config scotus), "
        "labels from the Supreme Court Database (SCDB)."
    )

st.title("Supreme Court opinion → issue area")
st.write(
    "Classifies a US Supreme Court opinion into one of 13 SCDB issue areas "
    "(Criminal Procedure, Civil Rights, Economic Activity, ...)."
)

classify_tab, results_tab, about_tab = st.tabs(["Classify", "Results", "About & limitations"])

# ---------------------------------------------------------------- classify

with classify_tab:
    samples = get_samples()
    source = st.radio("Input", ["Sample test case", "Paste your own text"], horizontal=True)

    sample = None
    if source == "Sample test case":
        options = {
            f"Test #{case['test_index']}: {case['label_name']}": case for case in samples
        }
        choice = st.selectbox("Sample (one per class, from the test split)", list(options))
        sample = options[choice]
        st.caption(
            f"True label: **{sample['label_name']}**. The fixture keeps the first "
            f"{len(sample['full_text']):,} characters of a {sample['word_count']:,}-word opinion."
        )
        text = st.text_area("Opinion text", sample["full_text"], height=220,
                            key=f"text_{sample['test_index']}")
    else:
        text = st.text_area("Opinion text", "", height=220, key="text_custom",
                            placeholder="Paste the text of a court opinion here.")

    chosen = st.multiselect(
        "Models",
        options=list(demo.LIVE_MODELS),
        default=[m for m in available if m != "transformer_legal_bert"],
        format_func=lambda m: demo.MODEL_LABELS[m],
        help="Legal-BERT is slower on CPU; select it explicitly.",
    )

    if st.button("Classify", type="primary"):
        words = len(text.split())
        if not text.strip():
            st.warning("Enter some text first.")
        elif not chosen:
            st.warning("Select at least one model.")
        else:
            if words < 100:
                st.warning(
                    f"Only {words} words. The models were trained on full opinions, "
                    "so predictions on short snippets are unreliable."
                )
            columns = st.columns(len(chosen))
            for column, name in zip(columns, chosen):
                with column:
                    if name not in available:
                        st.error(f"{demo.MODEL_LABELS[name]}: artifact not found.")
                        continue
                    if name == "transformer_legal_bert":
                        try:
                            tokenizer, model = get_bert()
                        except Exception as exc:  # network or file problem
                            st.error(f"Legal-BERT could not be loaded: {exc}")
                            continue
                        result = demo.predict_bert(tokenizer, model, text)
                        note = demo.truncation_note(demo.count_bert_tokens(tokenizer, text))
                        demo_note = ("Live weights come from a rerun with identical settings; "
                                     "reported metrics use a different run, so predictions can differ.")
                        show_prediction(name, result, " ".join(filter(None, [note, demo_note])))
                    else:
                        vectorizer, classifier = get_sklearn(name)
                        result = demo.predict_sklearn(vectorizer, classifier, text)
                        if result["vocabulary_hits"] == 0:
                            st.warning("No words matched the TF-IDF vocabulary.")
                        show_prediction(name, result,
                                        demo.reported_setting_note(name, classifier))

    if sample is not None:
        st.subheader("Stored predictions on the full document")
        st.caption(
            "From results/predictions/ (full opinion, not the 3,000-character excerpt). "
            "Legal-BERT here is the reported run."
        )
        st.dataframe(demo.stored_predictions(sample["test_index"]), hide_index=True)

# ---------------------------------------------------------------- results

with results_tab:
    split = st.radio("Split", ["test", "validation"], horizontal=True)
    st.dataframe(
        demo.metrics_table(split), hide_index=True,
        column_config={c: st.column_config.NumberColumn(format="%.3f")
                       for c in ("Macro F1", "Accuracy", "Weighted F1")},
    )
    st.caption(
        "All numbers are read from results/metrics.csv and results/bootstrap_ci.json. "
        "Macro F1 is the primary metric. CIs: 1,000 bootstrap resamples of the test set, "
        "unpaired, so they describe test sampling only."
    )
    left, right = st.columns(2)
    with left:
        st.image(str(FIGURES / "model_comparison.png"), caption="Test macro F1 and accuracy")
    with right:
        cm_model = st.selectbox(
            "Normalized confusion matrix (test)",
            ["tfidf_linear_svc", "tfidf_logistic_regression", "transformer_legal_bert",
             "lda_lr", "doc2vec_lr"],
            format_func=lambda m: demo.MODEL_LABELS[m],
        )
        st.image(str(FIGURES / f"confusion_matrix_normalized_{cm_model}.png"))
    st.subheader("Per-class test scores")
    st.dataframe(
        demo.per_class_table(cm_model), hide_index=True,
        column_config={c: st.column_config.NumberColumn(format="%.3f")
                       for c in ("Precision", "Recall", "F1")},
    )
    st.caption("Rare classes have very small test support, so their F1 is noisy.")

# ---------------------------------------------------------------- about

with about_tab:
    run_info = json.loads((demo.RESULTS_DIR / "transformer_run_info.json").read_text())
    st.markdown(
        f"""
**Task.** Single-label classification of US Supreme Court opinions into 13 SCDB issue areas.

**Models.** TF-IDF (unigrams + bigrams) with LinearSVC or SGD logistic regression;
LDA topics and Doc2Vec vectors with logistic regression (the reference paper's methods);
and `{run_info['model_name']}` fine-tuned on the first {run_info['max_length']} tokens.

**Limitations.**
- Legal-BERT sees only the first {run_info['max_length']} tokens of each opinion; most opinions are far longer.
- LDA and Doc2Vec ran in reduced CPU configurations, so their scores are a lower bound for those methods.
- One seed per model; Legal-BERT varied from run to run.
- The official split is chronological, so the test period is later than the training period.
- Rare classes have tiny test support, which makes macro F1 noisy.
- Bootstrap CIs are unpaired and capture test sampling only.
- The demo Legal-BERT weights come from a rerun, not the run whose predictions are reported.
"""
    )
