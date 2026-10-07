# Citation checks

How each reference used in the README, write-up, slides and viva notes was
verified. Checked on 2026-10-07.

| Reference | Used for | Verified against | Status |
|---|---|---|---|
| Iyer, *Classification of Legal Text*, Stanford CS229 report, Spring 2020 | reference paper, its methods (LDA + LR, Doc2Vec + LR), BERT out-of-memory | `docs/references/Iyer.pdf` (title and author on page 1; BERT notebook listed as "runs out of memory" in Appendix A); critique in `docs/reference_notes.md` | VERIFIED |
| Chalkidis et al., *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English*, ACL 2022 (arXiv:2110.00976) | dataset, chronological split and year ranges | BibTeX on the `coastalcph/lex_glue` dataset card; paper text, Section 3 (arXiv v4) | VERIFIED |
| Spaeth et al., *Supreme Court Database*, Version 2020 Release 01, Washington University Law | source of the issue-area labels | dataset card ("Spaeth et al. (2020)", http://scdb.wustl.edu); full entry in the LexGLUE reference list | VERIFIED |
| Chalkidis et al., *LEGAL-BERT: The Muppets straight out of Law School*, Findings of EMNLP 2020 | the `nlpaueb/legal-bert-base-uncased` model | LexGLUE reference list | VERIFIED |

## Chronological split

The LexGLUE paper, Section 3 (SCOTUS), states that the cases "are
chronologically split into training (5k, 1946–1982), development (1.4k,
1982–1991), test (1.4k, 1991–2016) sets". The same years are stored in
`results/dataset_info.json` (`split_method`). The dataset card itself does not
describe the split method, so the paper is the source.

## Not used

Benchmark scores from the LexGLUE paper are not quoted anywhere: they use a
hierarchical model over the whole document and different reporting, so they
are not comparable with our truncated run. The reference paper's Table 1 is
not used as a target because its column labels contradict its own text
(`docs/reference_notes.md`).
