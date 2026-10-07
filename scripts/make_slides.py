"""Build reports/slides.pptx: the 15-slide review deck with speaker notes.

Every number is rendered from results/ through src/report_facts.py
({{key}} placeholders). Figures come from results/figures/, plus two charts
drawn from results/ at build time and the demo screenshot in reports/figures/.
Run: python scripts/make_slides.py
"""

import json
import re
import sys
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reports.writeup import ci_figure, per_class_figure  # noqa: E402
from src.preprocess import normalize_text  # noqa: E402
from src.report_facts import DISPLAY, facts, render, results_rows  # noqa: E402

OUTPUT = ROOT / "reports" / "slides.pptx"
FIGURES = ROOT / "results" / "figures"
DEMO_SCREENSHOT = ROOT / "reports" / "figures" / "demo_screenshot.png"

NAVY = RGBColor(0x1D, 0x4E, 0x89)
AMBER = RGBColor(0xE8, 0xA3, 0x3D)
INK = RGBColor(0x1F, 0x29, 0x33)
GREY = RGBColor(0x52, 0x60, 0x6D)
LIGHT = RGBColor(0xEE, 0xF3, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"
W, H = Inches(13.333), Inches(7.5)
TOTAL = 15


# ---------------------------------------------------------------- helpers


_MARKUP = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)")


def _runs(paragraph, text, size, color=INK, bold=False, font=FONT):
    """Add text to a paragraph; **bold**, *italic* and `code` segments are styled."""
    for part in _MARKUP.split(text):
        if not part:
            continue
        run = paragraph.add_run()
        run.font.size = Pt(size)
        run.font.name = font
        run.font.bold = bold
        run.font.color.rgb = color
        if part.startswith("**") and part.endswith("**"):
            part, run.font.bold = part[2:-2], True
        elif part.startswith("`") and part.endswith("`"):
            part, run.font.color.rgb = part[1:-1], NAVY  # code: same font, accent colour
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            part, run.font.italic = part[1:-1], True
        run.text = part


def _bullet(paragraph, level=0):
    p_pr = paragraph._p.get_or_add_pPr()
    indent = Inches(0.3 + 0.3 * level)
    p_pr.set("marL", str(Emu(indent)))
    p_pr.set("indent", str(-Emu(Inches(0.3))))
    bu_clr = etree.SubElement(p_pr, qn("a:buClr"))
    etree.SubElement(bu_clr, qn("a:srgbClr")).set("val", "E8A33D" if level == 0 else "52606D")
    etree.SubElement(p_pr, qn("a:buFont")).set("typeface", "Arial")
    etree.SubElement(p_pr, qn("a:buChar")).set("char", "•" if level == 0 else "–")


def textbox(slide, x, y, w, h, items, size=18, color=INK, bullets=True, gap=8, anchor=MSO_ANCHOR.TOP,
            align=PP_ALIGN.LEFT, font=FONT):
    """items: strings (rendered); a tuple (text, level) gives a sub-bullet."""
    box = slide.shapes.add_textbox(x, y, w, h)
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = Inches(0.05)
    for i, item in enumerate(items):
        text, level = item if isinstance(item, tuple) else (item, 0)
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.space_after = Pt(gap)
        paragraph.line_spacing = 1.05
        _runs(paragraph, render(text), size - 2 * level, color, font=font)
        if bullets:
            _bullet(paragraph, level)
    return box


def new_slide(prs, number, title, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    textbox(slide, Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.8), [title], size=30,
            color=NAVY, bullets=False, anchor=MSO_ANCHOR.BOTTOM)
    for run in slide.shapes[-1].text_frame.paragraphs[0].runs:
        run.font.bold = True
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.65), Inches(1.2), Inches(1.1), Inches(0.06))
    _fill(bar, AMBER)
    textbox(slide, Inches(0.6), Inches(7.0), Inches(8), Inches(0.35),
            ["Classification of Legal Text · UE24CS352A Machine Learning"], size=10, color=GREY,
            bullets=False)
    textbox(slide, Inches(11.2), Inches(7.0), Inches(1.5), Inches(0.35), [f"{number} / {TOTAL}"],
            size=10, color=GREY, bullets=False, align=PP_ALIGN.RIGHT)
    slide.notes_slide.notes_text_frame.text = render(" ".join(notes.split()))
    return slide


def _fill(shape, color, line=None):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    shape.shadow.inherit = False


def picture(slide, source, x, y, max_w, max_h):
    """Add an image scaled to fit the box, centred in it."""
    if isinstance(source, Path):
        width_px, height_px = Image.open(source).size
    else:
        width_px, height_px = Image.open(source).size
        source.seek(0)
    scale = min(max_w / width_px, max_h / height_px)
    w, h = int(width_px * scale), int(height_px * scale)
    return slide.shapes.add_picture(source if not isinstance(source, Path) else str(source),
                                    x + (max_w - w) // 2, y + (max_h - h) // 2, w, h)


def table(slide, x, y, col_widths, rows, size=14, header_size=None, row_h=0.42, highlight=None):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), x, y, sum(col_widths, Emu(0)),
                                   Inches(row_h * len(rows)))
    tbl = shape.table
    tbl.first_row = True
    for c, width in enumerate(col_widths):
        tbl.columns[c].width = width
    for r, row in enumerate(rows):
        tbl.rows[r].height = Inches(row_h)
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            header = r == 0
            _runs(paragraph, render(str(value)), (header_size or size) if header else size,
                  WHITE if header else INK, bold=header or (highlight is not None and r == highlight))
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if header else (LIGHT if r % 2 == 0 else WHITE)
    return shape


def box(slide, x, y, w, h, lines, fill=LIGHT, color=INK, size=14, line=None, bold_first=True):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shape.adjustments[0] = 0.12
    _fill(shape, fill, line)
    frame = shape.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.margin_left = frame.margin_right = Inches(0.08)
    for i, text in enumerate(lines):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.alignment = PP_ALIGN.CENTER
        _runs(paragraph, render(text), size if i == 0 else size - 2, color, bold=bold_first and i == 0)
    return shape


def arrow(slide, x, y, w=Inches(0.42), h=Inches(0.3)):
    shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, h)
    _fill(shape, AMBER)
    return shape


# ---------------------------------------------------------------- slides


def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    _fill(background, NAVY)
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(3.55), Inches(1.6), Inches(0.08))
    _fill(band, AMBER)
    textbox(slide, Inches(0.8), Inches(1.6), Inches(11.5), Inches(1.2), ["Classification of Legal Text"],
            size=44, color=WHITE, bullets=False, anchor=MSO_ANCHOR.BOTTOM)
    slide.shapes[-1].text_frame.paragraphs[0].runs[0].font.bold = True
    textbox(slide, Inches(0.8), Inches(2.8), Inches(11.5), Inches(0.7),
            ["Predicting the issue area of US Supreme Court opinions"], size=24, color=WHITE,
            bullets=False)
    textbox(slide, Inches(0.8), Inches(3.9), Inches(11.5), Inches(1.2),
            ["[Member 1 Name / SRN]", "[Member 2 Name / SRN]"], size=20, color=WHITE, bullets=False,
            gap=4)
    textbox(slide, Inches(0.8), Inches(5.6), Inches(11.5), Inches(1.0),
            ["UE24CS352A Machine Learning · Mini-project review",
             "github.com/AhanaSrinivas/legal-text-classification"],
            size=14, color=RGBColor(0xC8, 0xD6, 0xE8), bullets=False, gap=2)
    slide.notes_slide.notes_text_frame.text = render(" ".join("""
        Introduce the team and the task. We classify US Supreme Court opinions into the
        {{n_classes}} issue areas of the Supreme Court Database. The project reproduces the two
        methods of a Stanford CS229 report, adds strong linear baselines and a legal-domain
        Transformer, and is fully reproducible: every number on these slides is generated from the
        files in results/. Then a live demo.""".split()))


def slide_problem(prs):
    slide = new_slide(prs, 2, "Problem", """
        The input is the full text of one opinion; the output is exactly one issue area, so this is
        single-label multi-class classification. Two things make it hard: documents are very long
        and the classes are very imbalanced. The reference project by Iyer tried LDA topics and
        Doc2Vec vectors with logistic regression; its BERT attempt ran out of memory, so it has no
        Transformer result. Our goal is to reproduce those methods on a public benchmark with
        fixed splits, add TF-IDF baselines and Legal-BERT, and report the results honestly.""")
    textbox(slide, Inches(0.6), Inches(1.55), Inches(6.4), Inches(5.2), [
        "**Input:** full text of a US Supreme Court opinion",
        "**Output:** one of {{n_classes}} SCDB issue areas",
        "Single-label, multi-class; long documents; strong class imbalance",
        "**Reference:** Iyer, Stanford CS229 (2020): LDA + LR and Doc2Vec + LR; BERT ran out of memory",
        "**Our aim:** reproduce the paper methods on a public benchmark, add strong baselines and "
        "Legal-BERT, report honestly",
    ], size=19, gap=12)
    names = facts()["label_names"]
    for i, name in enumerate(names):
        col, row = i % 2, i // 2
        box(slide, Inches(7.45 + col * 2.75), Inches(1.6 + row * 0.72), Inches(2.6), Inches(0.58),
            [name], fill=LIGHT if i % 3 else RGBColor(0xDC, 0xE7, 0xF3), size=13, bold_first=False)


def slide_motivation(prs):
    slide = new_slide(prs, 3, "Motivation", """
        Every opinion has to be catalogued by subject before lawyers and researchers can find it;
        doing this by hand is slow. Legal opinions are a hard test for NLP: they open with captions,
        citations and counsel names, the reasoning is long, and the vocabulary is specialised.
        A practical question follows: is it better to read the whole document with a simple model or
        only the opening with a strong pretrained model? Because some issue areas are rare, we judge
        models by macro F1, which weights every area equally, instead of accuracy.""")
    textbox(slide, Inches(0.6), Inches(1.55), Inches(12), Inches(5.2), [
        "Case law has to be catalogued by subject before it can be searched; doing it by hand is slow",
        "Legal text is hard for NLP: long, citation-heavy, procedural boilerplate, domain vocabulary",
        "Practical question: **whole document with a simple model** or **only the opening with a "
        "pretrained Transformer**?",
        "Rare issue areas matter as much as common ones, so we use **macro F1** as the primary metric",
        "Reproducibility matters: fixed public splits, saved predictions, generated reports",
    ], size=21, gap=16)


def slide_dataset(prs):
    slide = new_slide(prs, 4, "Dataset: LexGLUE SCOTUS", """
        We use the SCOTUS task of the LexGLUE benchmark from the Hugging Face Hub. It has
        {{n_total}} opinions in official splits that LexGLUE created chronologically: training
        {{train_years}}, validation {{val_years}}, test {{test_years}}. We found no empty texts and
        no exact duplicates within or across splits. The chart shows the imbalance:
        {{largest_class}} has {{largest_count}} training documents and {{smallest_class}} only
        {{smallest_count}}, a ratio of {{imbalance_ratio}} to 1. In the test split the rarest
        classes have {{rare_supports}} documents, so their F1 scores are very noisy.""")
    textbox(slide, Inches(0.6), Inches(1.55), Inches(5.0), Inches(5.2), [
        "`{{dataset_name}}`, config `{{dataset_config}}`; SCDB labels",
        "{{n_total}} opinions, official **chronological** splits",
        ("train {{n_train}} ({{train_years}})", 1),
        ("validation {{n_val}} ({{val_years}})", 1),
        ("test {{n_test}} ({{test_years}})", 1),
        "No empty texts, no duplicates (SHA-256)",
        "Imbalance **{{imbalance_ratio}} : 1** ({{largest_class}} vs {{smallest_class}})",
        "Rarest test classes: {{rare_desc}}",
    ], size=17, gap=8)
    picture(slide, FIGURES / "class_distribution.png", Inches(5.7), Inches(1.5), Inches(7.2), Inches(5.3))


def slide_pipeline(prs):
    slide = new_slide(prs, 5, "Pipeline", """
        The data flows left to right. Text is normalised, and every feature extractor is fitted on
        the training split only. Each feature type feeds its classifier: TF-IDF into LinearSVC and
        logistic regression, raw counts into LDA, tokens into Doc2Vec, and word pieces into
        Legal-BERT. Hyperparameters are chosen on validation; the test split is predicted once per
        model. Every model saves its predictions, and one script, evaluate_all.py, computes all
        metrics, confidence intervals, confusion matrices and the error analysis from those files.
        The README, write-up, slides and demo all read from results/.""")
    top = Inches(1.75)
    box(slide, Inches(0.45), Inches(3.05), Inches(1.95), Inches(1.6),
        ["LexGLUE SCOTUS", "{{n_train}} / {{n_val}} / {{n_test}}", "train / val / test"], size=15)
    arrow(slide, Inches(2.48), Inches(3.7))
    box(slide, Inches(2.98), Inches(3.05), Inches(1.95), Inches(1.6),
        ["Normalise", "numbers out, legal", "stop-word list"], size=15)
    arrow(slide, Inches(5.01), Inches(3.7))
    features = ["TF-IDF (1–2-grams)", "Counts → LDA", "Doc2Vec (PV-DM)", "Word pieces (≤ {{bert_max_len}})"]
    models = ["LinearSVC · LR (SGD)", "Logistic regression", "Logistic regression", "Legal-BERT fine-tune"]
    for i, (feature, model) in enumerate(zip(features, models)):
        y = top + Inches(i * 0.95)
        box(slide, Inches(5.5), y, Inches(2.45), Inches(0.72), [feature], fill=RGBColor(0xDC, 0xE7, 0xF3),
            size=14)
        arrow(slide, Inches(8.02), y + Inches(0.21), w=Inches(0.36), h=Inches(0.3))
        box(slide, Inches(8.45), y, Inches(2.25), Inches(0.72), [model], fill=NAVY, color=WHITE, size=14)
    arrow(slide, Inches(10.77), Inches(3.7))
    box(slide, Inches(11.25), Inches(2.45), Inches(1.75), Inches(2.8),
        ["Saved predictions", "evaluate_all.py", "metrics · CIs", "confusion · errors",
         "README · PDF", "slides · demo"], size=14, fill=RGBColor(0xFB, 0xEB, 0xD0))
    textbox(slide, Inches(0.6), Inches(5.75), Inches(12.2), Inches(0.9), [
        "Fit on **train** only  ·  tune on **validation** only  ·  **test** predicted once per model  ·  "
        "seed {{seed}}",
    ], size=16, bullets=False, align=PP_ALIGN.CENTER, color=GREY)


def slide_preprocessing(prs):
    eda = json.loads((ROOT / "results" / "eda_stats.json").read_text(encoding="utf-8"))
    raw = " ".join(eda["opening_snippets"][0]["snippet"].split())[:118]
    clean = normalize_text(raw)
    slide = new_slide(prs, 6, "Preprocessing", """
        Normalisation is deliberately conservative. We apply Unicode NFKC, lower-case the text and
        collapse whitespace. Standalone numbers are deleted, because reporter citations, docket
        numbers and years identify the period of a case rather than its topic, and our split is
        chronological. Tokens like 12th survive. We use scikit-learn's English stop words but keep
        words that change legal meaning, such as not, shall, must and unless. There is no stemming.
        A validation-only check with the earlier saga model gave macro F1 {{abl_drop_f1}} with
        numbers removed and {{abl_keep_f1}} with them kept; that model was under-converged, so this
        is only a rough indication. Most important: every vectorizer and embedding model is fitted on
        the training split only, and a unit test checks that a validation-only word never enters the
        vocabulary.""")
    textbox(slide, Inches(0.6), Inches(1.55), Inches(6.3), Inches(5.2), [
        "Unicode NFKC, lower case, collapse whitespace",
        "Delete standalone numbers (citations, docket numbers, years identify the era, not the "
        "topic); keep `12th`, `1st`",
        "Stop words: scikit-learn list **minus** meaning-changing words: not, no, never, shall, may, "
        "must, without, unless, …",
        "No stemming or lemmatisation",
        "**Leakage control:** vectorizers, LDA and Doc2Vec fitted on train only (unit-tested)",
    ], size=18, gap=12)
    box(slide, Inches(7.2), Inches(1.7), Inches(5.6), Inches(1.9), ["Raw opening (train example)", raw],
        fill=LIGHT, size=13)
    arrow(slide, Inches(9.75), Inches(3.75), w=Inches(0.5), h=Inches(0.35))
    box(slide, Inches(7.2), Inches(4.25), Inches(5.6), Inches(1.9), ["After normalize_text()", clean],
        fill=RGBColor(0xFB, 0xEB, 0xD0), size=13)


def slide_classical(prs):
    slide = new_slide(prs, 7, "Classical ML: TF-IDF + linear models", """
        Both classical models share one TF-IDF matrix of word unigrams and bigrams, capped at
        {{tfidf_max_features}} features, with sublinear term frequency, fitted on train. LinearSVC
        minimises the squared hinge loss with L2 regularisation; C = {{svc_c}} was best of
        {{svc_c_grid}} on validation. For logistic regression the first attempt used the saga solver
        with only {{saga_max_iter}} iterations and tolerance {{saga_tol}}, so it stopped long before
        convergence: validation macro F1 {{saga_val_f1}}. We kept it only for transparency and
        replaced it with SGDClassifier on log loss, which is logistic regression trained by
        stochastic gradient descent. With alpha {{lr_alpha}} it converged after {{lr_n_iter}} of
        {{lr_max_iter}} epochs and reached {{lr_val_f1}} on validation. No class weights were used.""")
    textbox(slide, Inches(0.6), Inches(1.55), Inches(6.4), Inches(5.2), [
        "TF-IDF: unigrams + bigrams, {{tfidf_max_features}} features, sublinear tf, min_df "
        "{{tfidf_min_df}}, max_df {{tfidf_max_df}}",
        "**LinearSVC:** L2, squared hinge; C = {{svc_c}} chosen from {{svc_c_grid}}",
        "**Logistic regression:** SGD on log loss, L2; alpha = {{lr_alpha}} from {{lr_alpha_grid}}; "
        "converged in {{lr_n_iter}} / {{lr_max_iter}} epochs",
        "First LR (saga, max_iter {{saga_max_iter}}, tol {{saga_tol}}) was **under-converged**: kept "
        "only as a reference",
        "All choices made on validation macro F1",
    ], size=18, gap=12)
    table(slide, Inches(7.35), Inches(2.2), [Inches(2.9), Inches(1.4), Inches(1.4)], [
        ["Model", "Val macro F1", "Test macro F1"],
        ["TF-IDF + LinearSVC", "{{svc_val_f1}}", "{{svc_test_f1}}"],
        ["TF-IDF + LR (SGD)", "{{lr_val_f1}}", "{{lr_test_f1}}"],
        ["TF-IDF + LR (saga, under-conv.)", "{{saga_val_f1}}", "{{saga_test_f1}}"],
    ], size=15, row_h=0.6)


def slide_topic(prs):
    f = facts()
    slide = new_slide(prs, 8, "Paper methods: LDA and Doc2Vec", """
        These are the reference paper's two methods. For LDA we use raw word counts, which suit
        LDA's multinomial model better than the TF-IDF input the paper describes. We swept the
        number of topics over {{lda_grid}} on validation; K = {{lda_k}} was best, at the edge of the
        grid. Doc2Vec is the PV-DM paragraph-vector model with {{d2v_vector_size}} dimensions,
        trained on train only; validation and test vectors are inferred. Both ran in reduced CPU
        configurations: LDA with {{lda_count_features}} count features and {{lda_max_iter}}
        iterations, Doc2Vec with {{d2v_epochs}} epochs on the first {{d2v_max_tokens}} tokens. Their
        test macro F1, {{lda_test_f1}} and {{d2v_test_f1}}, is far below TF-IDF, but we do not claim
        the methods are weak in general.""")
    textbox(slide, Inches(0.6), Inches(1.55), Inches(6.6), Inches(5.2), [
        "**LDA + LR:** topic mixture of each document as features",
        ("raw counts ({{lda_count_features}} terms, min_df {{lda_min_df}}), {{lda_max_iter}} batch iterations", 1),
        ("K chosen on validation from {{lda_grid}} → K = {{lda_k}} (grid edge)", 1),
        "**Doc2Vec + LR:** PV-DM, {{d2v_vector_size}} dims, window {{d2v_window}}, {{d2v_epochs}} epochs",
        ("trained on train only; val/test vectors inferred", 1),
        ("first {{d2v_max_tokens}} tokens of each document", 1),
        "Reduced CPU configurations: results are a **lower bound** for these methods",
    ], size=18, gap=9)
    rows = [["Model / setting", "Val macro F1", "Test macro F1"]]
    for k in f["lda_grid"].split(", "):
        rows.append([f"LDA, K = {k}", f"{{{{lda_val_f1_k{k}}}}}",
                     "{{lda_test_f1}}" if k == f["lda_k"] else "–"])
    rows.append(["Doc2Vec", "{{d2v_val_f1}}", "{{d2v_test_f1}}"])
    table(slide, Inches(7.6), Inches(2.0), [Inches(2.3), Inches(1.5), Inches(1.5)], rows, size=15,
          row_h=0.55, highlight=len(rows) - 2)


def slide_transformer(prs):
    slide = new_slide(prs, 9, "Transformer: Legal-BERT", """
        We fine-tuned {{bert_model}}, a BERT model pretrained on legal text, with a classification
        head. BERT reads at most {{bert_max_len}} tokens, and these opinions are long: on a
        {{tok_sample}}-document sample the median is about {{tok_median}} tokens, {{tok_pct_over_512}}
        percent exceed the limit, and the first {{bert_max_len}} tokens cover about
        {{tok_pct_covered}} percent of a document. Training: {{bert_epochs}} epochs, batch
        {{bert_batch}} with gradient accumulation {{bert_accum}}, learning rate {{bert_lr}}, weight
        decay {{bert_weight_decay}}, mixed precision, plain cross-entropy, best epoch by validation
        macro F1, on a Colab {{bert_gpu}} in {{bert_minutes}} minutes. Three identical runs gave
        test macro F1 between {{bert_run_min_4}} and {{bert_run_max_4}}; we report the
        {{bert_reported_f1_4}} run, and the demo weights come from the rerun.""")
    textbox(slide, Inches(0.6), Inches(1.55), Inches(5.6), Inches(5.2), [
        "`{{bert_model}}` + classification head",
        "Input: **first {{bert_max_len}} tokens** only",
        "{{bert_epochs}} epochs, batch {{bert_batch}} × accumulation {{bert_accum}} = {{bert_eff_batch}}, "
        "lr {{bert_lr}}, weight decay {{bert_weight_decay}}, mixed precision",
        "No class weights; best epoch by validation macro F1",
        "Colab {{bert_gpu}}, {{bert_minutes}} min",
        "Three identical runs: test macro F1 {{bert_run_min_4}}–{{bert_run_max_4}} (reported: "
        "{{bert_reported_f1_4}})",
    ], size=17, gap=10)
    picture(slide, FIGURES / "token_length_histogram.png", Inches(6.3), Inches(1.6), Inches(6.7), Inches(3.6))
    box(slide, Inches(6.6), Inches(5.4), Inches(6.1), Inches(1.25), [
        "{{tok_pct_over_512}}% of documents exceed {{bert_max_len}} tokens",
        "first {{bert_max_len}} tokens ≈ {{tok_pct_covered}}% of a document (median ≈ {{tok_median}} tokens)",
    ], fill=RGBColor(0xFB, 0xEB, 0xD0), size=17)


def slide_results(prs):
    slide = new_slide(prs, 10, "Results (test split)", """
        Here are the final test results; macro F1 is the primary metric and the intervals are
        bootstrap 95 percent intervals over {{ci_resamples}} test resamples. The two TF-IDF models
        are best, LinearSVC {{svc_test_f1}} and logistic regression {{lr_test_f1}}; their intervals
        {{svc_lr_ci_overlap}}, so we do not rank one above the other. Legal-BERT has accuracy close
        to them, {{bert_test_acc}}, but macro F1 only {{bert_test_f1}}: it does well on the large
        classes and badly on the small ones. Its interval and LinearSVC's {{svc_bert_ci_overlap}}.
        The paper methods are near {{lda_test_f1}}. The saga row is the under-converged reference,
        and the majority baseline shows why accuracy alone is misleading.""")
    rows = [["Model", "Macro F1", "95% CI", "Accuracy"]]
    for row in results_rows():
        name = row["name"].replace(" (first 512 tokens)", " (512 tok.)").replace(
            " (saga, under-converged)", " (saga, under-conv.)")
        rows.append([name, row["test_f1"], row["ci"], row["test_acc"]])
    table(slide, Inches(0.6), Inches(1.6), [Inches(3.3), Inches(1.15), Inches(1.75), Inches(1.15)],
          rows, size=14, row_h=0.56, highlight=1)
    picture(slide, ci_figure(title=""), Inches(8.0), Inches(1.6), Inches(5.0), Inches(3.8))
    textbox(slide, Inches(8.1), Inches(5.55), Inches(4.9), Inches(1.2), [
        "TF-IDF models best; Legal-BERT's CI and LinearSVC's {{svc_bert_ci_overlap}}",
        "Macro F1 punishes ignoring rare classes",
    ], size=15, gap=6)


def slide_errors(prs):
    f = facts()
    quart = {}
    for model, _, _, accuracy in f["ea_quartiles"]:
        quart.setdefault(model, []).append(accuracy)
    svc_q, bert_q = quart["tfidf_linear_svc"], quart["transformer_legal_bert"]
    slide = new_slide(prs, 11, "Error analysis", f"""
        This is Legal-BERT's normalised confusion matrix on the test split. The most frequent
        confusion is {{{{ea_top_pair_sentence}}}}. Judicial Power is often confused with Economic Activity,
        Criminal Procedure and Civil Rights. Legal-BERT never correctly predicts {{{{bert_zero_f1}}}},
        and recalls only {{{{bert_federalism_rpct}}}} percent of Federalism and
        {{{{bert_interstate_relations_rpct}}}} percent of Interstate Relations. The two models make
        different mistakes: {{{{ea_only_svc}}}} test cases are right only for LinearSVC,
        {{{{ea_only_bert}}}} only for Legal-BERT, and {{{{ea_both_wrong}}}} are wrong for both.
        Accuracy changes little across document-length quartiles: LinearSVC {min(svc_q):.3f} to
        {max(svc_q):.3f}, Legal-BERT {min(bert_q):.3f} to {max(bert_q):.3f}. These explanations are
        hypotheses from saved predictions, not tested causes.""")
    textbox(slide, Inches(0.6), Inches(1.55), Inches(5.6), Inches(5.2), [
        "Top confusion: {{ea_top_pair_sentence}}",
        "Judicial Power is confused with Economic Activity, Criminal Procedure and Civil Rights",
        "Legal-BERT never correctly predicts {{bert_zero_f1}}; Federalism recall "
        "{{bert_federalism_rpct}}%",
        "Different mistakes: {{ea_only_svc}} cases right only for SVC, {{ea_only_bert}} only for "
        "Legal-BERT",
        f"Little length effect: accuracy by length quartile {min(svc_q):.3f}–{max(svc_q):.3f} (SVC), "
        f"{min(bert_q):.3f}–{max(bert_q):.3f} (Legal-BERT)",
    ], size=17, gap=10)
    picture(slide, FIGURES / "confusion_matrix_normalized_transformer_legal_bert.png",
            Inches(6.3), Inches(1.45), Inches(6.8), Inches(5.45))


def slide_demo(prs):
    slide = new_slide(prs, 12, "Live demo: Streamlit app", """
        Run streamlit run app.py. In the Classify tab, pick a sample test case, one per class, or
        paste any opinion, choose models and press Classify. Each model shows its prediction and its
        top five scores: probabilities for logistic regression and Legal-BERT, margin scores for
        LinearSVC. Legal-BERT also reports how much of the text it could read. Below, the app shows
        the stored predictions each model made on the full opinion. The Results tab shows the same
        metrics as these slides, read from results/. Two honest caveats: the demo Legal-BERT weights
        are from the rerun, and the demo LinearSVC artifact uses C = {{demo_svc_c}} rather than the
        reported {{svc_c}}; the app says both on screen.""")
    textbox(slide, Inches(0.6), Inches(1.55), Inches(4.6), Inches(5.2), [
        "`streamlit run app.py`",
        "Pick a sample test case or paste an opinion",
        "Live predictions from LinearSVC, LR and Legal-BERT with top-5 scores",
        "Truncation warning: how much text Legal-BERT read",
        "Stored full-document predictions for comparison",
        "Results tab: metrics, CIs, confusion matrices from results/",
    ], size=17, gap=10)
    if DEMO_SCREENSHOT.exists():
        picture(slide, DEMO_SCREENSHOT, Inches(5.3), Inches(1.5), Inches(7.7), Inches(5.3))
    else:
        box(slide, Inches(5.3), Inches(1.5), Inches(7.7), Inches(5.3), ["Demo screenshot missing"])


def slide_limitations(prs):
    slide = new_slide(prs, 13, "Limitations", """
        We want to be clear about what these results do and do not show. Legal-BERT read only the
        first {{bert_max_len}} tokens, about {{tok_pct_covered}} percent of a typical opinion, so its
        result is for truncated input. LDA and Doc2Vec ran in reduced CPU configurations. Every model
        was trained once with one seed, and Legal-BERT alone varied between {{bert_run_min_4}} and
        {{bert_run_max_4}}. The split is chronological, so test cases come from a later period;
        every trained model drops from validation to test. The rarest test classes have
        {{rare_supports}} documents, so macro F1 is noisy. Our bootstrap intervals are unpaired and
        only capture test sampling. No model used class weights.""")
    textbox(slide, Inches(0.6), Inches(1.55), Inches(12), Inches(5.2), [
        "Legal-BERT sees only the first {{bert_max_len}} tokens (≈ {{tok_pct_covered}}% of a typical opinion)",
        "LDA and Doc2Vec ran in reduced CPU configurations (K = {{lda_k}} at the grid edge)",
        "One seed per model; Legal-BERT runs ranged {{bert_run_min_4}}–{{bert_run_max_4}}",
        "Chronological split: test is a later period; every trained model drops val → test "
        "(LinearSVC by {{svc_drop}})",
        "Rare classes: test support {{rare_supports}} documents, so macro F1 is noisy",
        "Bootstrap CIs are unpaired and capture test sampling only; no class weighting anywhere",
    ], size=20, gap=14)


def slide_conclusion(prs):
    slide = new_slide(prs, 14, "Conclusion and future work", """
        To conclude: simple linear models that read the whole document were the best, at about
        {{svc_test_f1}} macro F1. Legal-BERT on the opening of each opinion reached similar accuracy
        but much lower macro F1, mainly because it fails on rare classes; we think truncation and the
        lack of class weighting explain much of this, but we have not tested it. The paper's methods
        trail behind in our reduced configurations. Next steps: a hierarchical or long-document
        Transformer over the whole opinion, class-weighted loss and several seeds with paired tests,
        full-size LDA and Doc2Vec, and removing the caption and counsel header before classifying.""")
    textbox(slide, Inches(0.6), Inches(1.55), Inches(6.2), Inches(5.2), [
        "**Whole-document TF-IDF + linear model wins:** LinearSVC {{svc_test_f1}}, LR {{lr_test_f1}} macro F1",
        "**Legal-BERT on {{bert_max_len}} tokens:** close accuracy ({{bert_test_acc}}), lower macro F1 "
        "({{bert_test_f1}}); fails on rare classes",
        "Paper methods {{lda_test_f1}} / {{d2v_test_f1}} in reduced configurations",
        "Everything reproducible from saved predictions; numbers generated, not typed",
    ], size=18, gap=14)
    box(slide, Inches(7.2), Inches(1.7), Inches(5.6), Inches(0.6), ["Future work"], fill=NAVY, color=WHITE,
        size=18)
    textbox(slide, Inches(7.3), Inches(2.5), Inches(5.5), Inches(4.3), [
        "Hierarchical Legal-BERT or Longformer over the whole opinion",
        "Class-weighted / focal loss for rare classes",
        "Several seeds; paired bootstrap or McNemar tests",
        "Full-size LDA and Doc2Vec; wider K grid",
        "Strip caption and counsel header before classifying",
    ], size=17, gap=10)


def slide_references(prs):
    slide = new_slide(prs, 15, "References", """
        These are the four sources we rely on. The chronological split and its years come from the
        LexGLUE paper, section three; the labels come from the Supreme Court Database. The code,
        results and these documents are in the GitHub repository; the fitted demo models are on the
        Hugging Face Hub. Thank you; we are happy to take questions.""")
    textbox(slide, Inches(0.6), Inches(1.55), Inches(12.1), Inches(3.9), [
        "K. Iyer. *Classification of Legal Text.* Stanford CS229 project report, Spring 2020.",
        "I. Chalkidis, A. Jana, D. Hartung, M. Bommarito, I. Androutsopoulos, D. M. Katz, N. Aletras. "
        "LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. ACL 2022 "
        "(arXiv:2110.00976).",
        "H. J. Spaeth, L. Epstein, J. A. Segal, A. D. Martin, T. J. Ruger, S. C. Benesh. Supreme Court "
        "Database, Version 2020 Release 01. Washington University Law.",
        "I. Chalkidis, M. Fergadiotis, P. Malakasiotis, N. Aletras, I. Androutsopoulos. LEGAL-BERT: "
        "The Muppets straight out of Law School. Findings of EMNLP 2020.",
    ], size=16, gap=12)
    textbox(slide, Inches(0.6), Inches(5.5), Inches(12.1), Inches(1.2), [
        "Code and results: github.com/AhanaSrinivas/legal-text-classification",
        "Model artifacts: huggingface.co/AhanaSrinivas/legal-text-classification-artifacts",
    ], size=15, color=GREY, bullets=False, gap=4, font=FONT)


def _fix_zip_timestamps(path):
    """Rewrite the .pptx with fixed entry timestamps so rebuilds give identical bytes."""
    with zipfile.ZipFile(path) as source:
        entries = [(info, source.read(info.filename)) for info in source.infolist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as target:
        for info, data in entries:
            fixed = zipfile.ZipInfo(info.filename, date_time=(2026, 1, 1, 0, 0, 0))
            fixed.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(fixed, data)


def build(path=OUTPUT):
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    for make in (slide_title, slide_problem, slide_motivation, slide_dataset, slide_pipeline,
                 slide_preprocessing, slide_classical, slide_topic, slide_transformer, slide_results,
                 slide_errors, slide_demo, slide_limitations, slide_conclusion, slide_references):
        make(prs)
    count = len(prs.slides)
    if count != TOTAL:
        raise SystemExit(f"FAIL: expected {TOTAL} slides, built {count}")
    missing_notes = [i + 1 for i, s in enumerate(prs.slides)
                     if not s.has_notes_slide or not s.notes_slide.notes_text_frame.text.strip()]
    if missing_notes:
        raise SystemExit(f"FAIL: slides without speaker notes: {missing_notes}")
    prs.core_properties.title = "Classification of Legal Text"
    prs.core_properties.author = "[Member 1 Name / SRN], [Member 2 Name / SRN]"
    prs.save(str(path))
    _fix_zip_timestamps(path)
    print(f"Wrote {path.relative_to(ROOT)} ({count} slides, all with speaker notes)")
    return count


if __name__ == "__main__":
    build()
