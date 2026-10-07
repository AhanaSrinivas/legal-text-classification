"""Build reports/writeup.pdf, the two-page project write-up.

This file is the editable source. Every number comes from results/ through
src/report_facts.py ({{key}} placeholders); nothing is typed by hand.
Run: python reports/writeup.py   (fails if the PDF is longer than two pages)
"""

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pypdf import PdfReader
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

rl_config.invariant = 1  # no timestamps, so rebuilding gives identical bytes

from src.report_facts import DISPLAY, MODEL_ORDER, SHORT, facts, metric, render, results_rows

OUTPUT = ROOT / "reports" / "writeup.pdf"
MAX_PAGES = 2
INK = colors.HexColor("#1f2933")
ACCENT = colors.HexColor("#1d4e89")
RULE = colors.HexColor("#c8d1dc")


# ---------------------------------------------------------------- text

TITLE = "Classification of Legal Text: US Supreme Court Opinions by Issue Area"
SUBTITLE = (
    "UE24CS352A Machine Learning mini-project &nbsp;·&nbsp; [Member 1 Name / SRN], "
    "[Member 2 Name / SRN] &nbsp;·&nbsp; github.com/AhanaSrinivas/legal-text-classification"
)

PROBLEM = """Given the full text of a US Supreme Court opinion, predict its issue area: one of the
{{n_classes}} areas of the Supreme Court Database (SCDB), such as Criminal Procedure, Civil Rights or
Economic Activity. It is single-label, multi-class classification of very long and strongly
imbalanced documents. We reproduce the two methods of the reference project (Iyer, CS229 2020: LDA
topics and Doc2Vec vectors, each with logistic regression), add TF-IDF linear baselines, and
fine-tune Legal-BERT, which the reference project could not run because it ran out of memory."""

DATASET = [
    """<b>Source.</b> LexGLUE SCOTUS (<font face="Mono">{{dataset_name}}</font>, config
    <font face="Mono">{{dataset_config}}</font>; Chalkidis et al., 2022) with SCDB labels.
    {{n_total}} opinions in the official chronological splits: train {{n_train}} ({{train_years}}),
    validation {{n_val}} ({{val_years}}), test {{n_test}} ({{test_years}}). Columns: text, label.""",
    """<b>Checks.</b> No empty texts; no exact duplicates within or across splits (SHA-256 of
    every text).""",
    """<b>Imbalance.</b> {{largest_class}} has {{largest_count}} training documents and
    {{smallest_class}} {{smallest_count}} (ratio {{imbalance_ratio}}:1). Rarest test classes:
    {{rare_desc}}.""",
    """<b>Length.</b> Median {{words_median_test}} words per test opinion. On a {{tok_sample}}-document
    sample the median is {{tok_median_exact}} Legal-BERT tokens, {{tok_pct_over_512}}% of documents
    exceed 512 tokens, and the first 512 tokens cover about {{tok_pct_covered}}% of a document.""",
]

APPROACH = [
    """<b>Protocol.</b> Hyperparameters chosen on validation only; final models fitted on train
    only; test predicted once per model. Primary metric: macro F1, so every issue area counts
    equally; we also report accuracy, weighted F1 and per-class scores with support, plus 95%
    percentile bootstrap CIs ({{ci_resamples}} resamples) for test macro F1.""",
    """<b>Preprocessing.</b> NFKC normalisation, lower case, whitespace collapsed; standalone numbers
    removed (citations, docket numbers and years identify the era rather than the topic); scikit-learn
    English stop words minus words that change legal meaning (<i>not, no, shall, must, unless,
    without</i>, ...); no stemming. Vectorizers, LDA and Doc2Vec are fitted on the training split only.""",
    """<b>Models.</b> (a) majority-class baseline; (b) TF-IDF (unigrams + bigrams,
    {{tfidf_max_features}} features, sublinear tf) + LinearSVC (L2, squared hinge, C = {{svc_c}}
    from {{svc_c_grid}}); (c) the same features + logistic regression trained by SGD on log loss
    (L2, alpha = {{lr_alpha}} from {{lr_alpha_grid}}, converged after {{lr_n_iter}} of
    {{lr_max_iter}} epochs); an earlier saga run (max_iter {{saga_max_iter}}, tol {{saga_tol}}) is
    kept only as an under-converged reference; (d) LDA topic mixtures on raw counts + LR, K =
    {{lda_k}} from {{lda_grid}}; (e) Doc2Vec PV-DM ({{d2v_vector_size}} dimensions) + LR, with
    vectors inferred for validation and test; (f) Legal-BERT ({{bert_model}}) fine-tuned
    on the first {{bert_max_len}} tokens: {{bert_epochs}} epochs, effective batch {{bert_eff_batch}},
    learning rate {{bert_lr}}, weight decay {{bert_weight_decay}}, mixed precision, plain
    cross-entropy, best epoch by validation macro F1.""",
]

IMPLEMENTATION = """Python with scikit-learn, gensim, PyTorch and Hugging Face transformers.
<font face="Mono">src/</font> holds reusable modules (data loading and audits, train-only
preprocessing, shared evaluation); <font face="Mono">scripts/</font> has one script per stage.
Classical models were trained on a CPU laptop and Legal-BERT on a Colab {{bert_gpu}}
({{bert_minutes}} minutes). Every model saves its validation and test predictions, and
<font face="Mono">evaluate_all.py</font> rebuilds all metrics, CIs, confusion matrices and the error
analysis from them, so results reproduce without retraining. A Streamlit app
(<font face="Mono">app.py</font>) gives live predictions from the TF-IDF models and Legal-BERT next
to the stored full-document predictions. The pytest suite checks preprocessing, leakage
prevention and hand-computed metrics, and recomputes <font face="Mono">metrics.csv</font> from the
saved predictions. This write-up is generated from <font face="Mono">results/</font> by
<font face="Mono">reports/writeup.py</font>."""

CONCLUSIONS = [
    """Linear models on the full TF-IDF document are best: LinearSVC macro F1 {{svc_test_f1}}
    (CI {{svc_ci_lo}}–{{svc_ci_hi}}) and SGD logistic regression {{lr_test_f1}}. Their intervals
    {{svc_lr_ci_overlap}}, so we do not rank one above the other.""",
    """Legal-BERT on the first {{bert_max_len}} tokens has close accuracy ({{bert_test_acc}} vs
    {{svc_test_acc}}) but much lower macro F1 ({{bert_test_f1}}); its interval and LinearSVC's
    {{svc_bert_ci_overlap}}, so the gap is not test-sampling noise. It never correctly predicts
    {{bert_zero_f1}} and recalls only {{bert_federalism_rpct}}% of Federalism cases. Hypotheses, not
    tested: truncation to an opening that is mostly caption and citations, no class weighting, and a
    single short run (three identical runs ranged {{bert_run_min_4}}–{{bert_run_max_4}}; we report
    {{bert_reported_f1_4}}).""",
    """The paper methods trail far behind (LDA {{lda_test_f1}}, Doc2Vec {{d2v_test_f1}}), but in
    reduced CPU configurations, so this does not show the methods themselves are weak.""",
    """The most frequent confusion is {{ea_top_pair_sentence}}. The two models are partly
    complementary: {{ea_only_svc}} test cases are right only for LinearSVC and {{ea_only_bert}} only
    for Legal-BERT. Every trained model drops from validation to test (LinearSVC by {{svc_drop}}),
    consistent with, but not proof of, drift across the chronological split.""",
]

LIMITATIONS = [
    "Legal-BERT reads about {{tok_pct_covered}}% of a typical opinion (first {{bert_max_len}} tokens).",
    "LDA ({{lda_count_features}} count features, {{lda_max_iter}} iterations; K = {{lda_k}} at the grid "
    "edge) and Doc2Vec ({{d2v_epochs}} epochs, first {{d2v_max_tokens}} tokens) ran in reduced CPU "
    "configurations.",
    "One seed per model; no class weighting. Rare classes have tiny test support ({{rare_supports}} "
    "documents), so macro F1 is noisy.",
    "Bootstrap CIs are unpaired and capture test sampling only. The demo Legal-BERT weights come "
    "from the rerun, not the reported run.",
]

REFERENCES = [
    "K. Iyer. <i>Classification of Legal Text</i>. Stanford CS229 project report, Spring 2020.",
    "I. Chalkidis, A. Jana, D. Hartung, M. Bommarito, I. Androutsopoulos, D. M. Katz, N. Aletras. "
    "<i>LexGLUE: A Benchmark Dataset for Legal Language Understanding in English</i>. ACL 2022 "
    "(arXiv:2110.00976).",
    "H. J. Spaeth, L. Epstein, J. A. Segal, A. D. Martin, T. J. Ruger, S. C. Benesh. "
    "<i>Supreme Court Database</i>, Version 2020 Release 01. Washington University Law.",
    "I. Chalkidis, M. Fergadiotis, P. Malakasiotis, N. Aletras, I. Androutsopoulos. "
    "<i>LEGAL-BERT: The Muppets straight out of Law School</i>. Findings of EMNLP 2020.",
]


# ---------------------------------------------------------------- layout


def register_fonts():
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("Body", font_dir / "STIXGeneral.ttf"))
    pdfmetrics.registerFont(TTFont("Body-Bold", font_dir / "STIXGeneralBol.ttf"))
    pdfmetrics.registerFont(TTFont("Body-Italic", font_dir / "STIXGeneralItalic.ttf"))
    pdfmetrics.registerFont(TTFont("Body-BoldItalic", font_dir / "STIXGeneralBolIta.ttf"))
    pdfmetrics.registerFont(TTFont("Head", font_dir / "DejaVuSans-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("Mono", font_dir / "DejaVuSansMono.ttf"))
    pdfmetrics.registerFontFamily(
        "Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic"
    )


def styles():
    body = ParagraphStyle("body", fontName="Body", fontSize=10, leading=12.1, textColor=INK,
                          alignment=TA_LEFT, spaceAfter=2.5)
    return {
        "title": ParagraphStyle("title", fontName="Head", fontSize=13.5, leading=16.5, textColor=ACCENT),
        "subtitle": ParagraphStyle("subtitle", parent=body, fontSize=8.6, leading=11,
                                   textColor=colors.HexColor("#52606d")),
        "h": ParagraphStyle("h", fontName="Head", fontSize=10.2, leading=13, textColor=ACCENT,
                            spaceBefore=5, spaceAfter=2.5),
        "body": body,
        "small": ParagraphStyle("small", parent=body, fontSize=8.4, leading=10.2, spaceAfter=1.2),
        "cell": ParagraphStyle("cell", parent=body, fontSize=8.6, leading=10, spaceAfter=0),
        "cellc": ParagraphStyle("cellc", parent=body, fontSize=8.6, leading=10, spaceAfter=0,
                                alignment=1),
    }


def bullets(items, style, gap=1.5):
    return ListFlowable(
        [ListItem(Paragraph(render(item), style), leftIndent=10, value="•") for item in items],
        bulletType="bullet", start="•", leftIndent=10, bulletFontSize=7, spaceBefore=0,
        bulletOffsetY=-1,
    )


def results_table(st):
    header = ["Model", "Test<br/>macro F1", "Test macro F1<br/>95% CI", "Test<br/>accuracy",
              "Test<br/>weighted F1", "Validation<br/>macro F1"]
    data = [[Paragraph(f"<b>{h}</b>", st["cell"] if i == 0 else st["cellc"])
             for i, h in enumerate(header)]]
    for row in results_rows():
        data.append([Paragraph(row["name"], st["cell"]), row["test_f1"], row["ci"],
                     row["test_acc"], row["test_wf1"], row["val_f1"]])
    table = Table(data, colWidths=[5.6 * cm, 2.2 * cm, 2.9 * cm, 2.2 * cm, 2.4 * cm, 2.4 * cm],
                  hAlign="LEFT")
    table.setStyle(TableStyle([
        ("FONT", (0, 1), (-1, -1), "Body", 8.6),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, ACCENT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, ACCENT),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#eef3f9")),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
    ]))
    return table


def _png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", metadata={"Software": None})
    plt.close(fig)
    buffer.seek(0)
    return buffer


def _style_axis(ax):
    ax.grid(axis="x", color="#e4e7eb", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=6.4)


FIG_W, FIG_H = 4.0, 3.05  # inches; both figures share the size


def ci_figure(title="A. Model comparison"):
    """Test macro F1 with bootstrap CIs, drawn from results/ at build time."""
    f = facts()
    models = [m for m in MODEL_ORDER if m != "majority_baseline"][::-1]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=220)
    for y, model in enumerate(models):
        s = SHORT[model]
        ax.plot([float(f[f"{s}_ci_lo"]), float(f[f"{s}_ci_hi"])], [y, y], color="#9aa5b1",
                lw=2.4, solid_capstyle="round")
        ax.plot(metric(model, "test", "macro_f1"), y, "o", color="#1d4e89", ms=4.4)
    labels = [DISPLAY[m].replace(" (first 512 tokens)", "").replace(" (saga, under-converged)",
                                                                     " (saga)") for m in models]
    ax.set_yticks(range(len(models)), labels)
    ax.set_xlabel("Test macro F1 (dot) with 95% bootstrap CI (bar)", fontsize=6.6)
    ax.set_xlim(0.2, 0.75)
    if title:
        ax.set_title(title, fontsize=7.4, loc="left", color="#1f2933")
    _style_axis(ax)
    fig.tight_layout(pad=0.3)
    return _png(fig)


def per_class_figure():
    """Per-class test F1 for LinearSVC and Legal-BERT, with test support."""
    per_class = facts()["per_class"]
    svc = per_class[per_class["model"] == "tfidf_linear_svc"].set_index("class")
    bert = per_class[per_class["model"] == "transformer_legal_bert"].set_index("class")
    classes = list(svc.sort_values("support").index)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=220)
    ys = range(len(classes))
    ax.barh([y + 0.2 for y in ys], [svc.loc[c, "f1"] for c in classes], height=0.38,
            color="#1d4e89", label="TF-IDF + LinearSVC")
    ax.barh([y - 0.2 for y in ys], [bert.loc[c, "f1"] for c in classes], height=0.38,
            color="#e8a33d", label="Legal-BERT")
    ax.set_yticks(list(ys), [f"{c} ({int(svc.loc[c, 'support'])})" for c in classes])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Test F1 per class (test support in brackets)", fontsize=6.6)
    ax.set_title("B. Per-class F1", fontsize=7.4, loc="left", color="#1f2933")
    ax.legend(fontsize=6, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.99), ncol=2,
              handlelength=1.2, columnspacing=0.8, borderaxespad=0)
    _style_axis(ax)
    fig.tight_layout(pad=0.3)
    return _png(fig)


def figures_row():
    width = 8.9 * cm
    images = [Image(make(), width=width, height=width * FIG_H / FIG_W)
              for make in (ci_figure, per_class_figure)]
    return Table([images], colWidths=[9.0 * cm, 9.0 * cm],
                 style=[("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0)])


def build(path=OUTPUT):
    register_fonts()
    st = styles()

    story = [
        Paragraph(TITLE, st["title"]),
        Spacer(1, 2),
        Paragraph(SUBTITLE, st["subtitle"]),
        Spacer(1, 3),
        Table([[""]], colWidths=[18.0 * cm], rowHeights=[1],
              style=[("LINEABOVE", (0, 0), (-1, 0), 0.6, RULE)]),
        Paragraph("1. Problem statement", st["h"]),
        Paragraph(render(PROBLEM), st["body"]),
        Paragraph("2. Dataset details", st["h"]),
        bullets(DATASET, st["body"]),
        Paragraph("3. Approach", st["h"]),
        bullets(APPROACH, st["body"]),
        Paragraph("4. Brief implementation overview", st["h"]),
        Paragraph(render(IMPLEMENTATION), st["body"]),
        KeepTogether([Paragraph("Results", st["h"]), results_table(st)]),
        Spacer(1, 4),
        figures_row(),
        Paragraph(render(
            "Official LexGLUE test split; validation macro F1 is shown for reference only. CIs: "
            "{{ci_resamples}} bootstrap resamples of the test set (seed {{ci_seed}}). The saga row "
            "is the under-converged reference, not a final model. Per-class precision, recall, F1 "
            "and support for every model: <font face=\"Mono\">results/per_class_f1.csv</font>."),
            st["small"]),
        Paragraph("5. Conclusions", st["h"]),
        bullets(CONCLUSIONS, st["body"]),
        Paragraph("Limitations", st["h"]),
        bullets(LIMITATIONS, st["small"]),
        Paragraph("References", st["h"]),
        ListFlowable([ListItem(Paragraph(ref, st["small"]), leftIndent=14, value=str(i))
                      for i, ref in enumerate(REFERENCES, 1)],
                     bulletType="1", bulletFormat="[%s]", leftIndent=14, bulletFontSize=8),
    ]

    doc = SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.3 * cm, bottomMargin=1.2 * cm, title=TITLE,
        author="[Member 1 Name / SRN], [Member 2 Name / SRN]",
        subject="UE24CS352A mini-project write-up",
    )
    doc.build(story)
    pages = len(PdfReader(str(path)).pages)
    if pages > MAX_PAGES:
        raise SystemExit(f"FAIL: {path.name} has {pages} pages (limit {MAX_PAGES})")
    print(f"Wrote {path.relative_to(ROOT)} ({pages} pages)")
    return pages


if __name__ == "__main__":
    build()
