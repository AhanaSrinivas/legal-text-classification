# Error analysis

This analysis uses the saved validation/test predictions and the test text. Explanations are hypotheses, not established causes.

## Most confused test-class pairs

### tfidf_linear_svc
- 37: Federalism -> Economic Activity
- 34: Civil Rights -> Criminal Procedure
- 27: Judicial Power -> Economic Activity
- 21: Judicial Power -> Criminal Procedure
- 19: Judicial Power -> Civil Rights
Lowest-F1 classes: Miscellaneous (F1=0.0000, support=3); Federalism (F1=0.3364, support=83); Privacy (F1=0.5238, support=28); Due Process (F1=0.5500, support=51); Judicial Power (F1=0.6432, support=200)

### transformer_legal_bert
- 49: Federalism -> Economic Activity
- 35: Judicial Power -> Economic Activity
- 26: Civil Rights -> Criminal Procedure
- 24: Judicial Power -> Criminal Procedure
- 21: Judicial Power -> Civil Rights
Lowest-F1 classes: Attorneys (F1=0.0000, support=17); Miscellaneous (F1=0.0000, support=3); Federalism (F1=0.1250, support=83); Interstate Relations (F1=0.2353, support=15); Due Process (F1=0.4719, support=51)

## Accuracy by test document-length quartile
| Model | Quartile | Documents | Accuracy |
|---|---:|---:|---:|
| tfidf_linear_svc | 1 | 350 | 0.6943 |
| tfidf_linear_svc | 2 | 349 | 0.7507 |
| tfidf_linear_svc | 3 | 351 | 0.7692 |
| tfidf_linear_svc | 4 | 350 | 0.7343 |
| transformer_legal_bert | 1 | 350 | 0.7286 |
| transformer_legal_bert | 2 | 349 | 0.7307 |
| transformer_legal_bert | 3 | 351 | 0.7322 |
| transformer_legal_bert | 4 | 350 | 0.7057 |

## Correctness overlap
- Only SVC correct: 121
- Only Legal-BERT correct: 103
- Both wrong: 265

## Concrete misclassified examples

### Example 1 (test index 0)
- True: Criminal Procedure; SVC: Civil Rights; Legal-BERT: Civil Rights.
- Text (first 300 characters): 502 U.S. 314 112 S.Ct. 719 116 L.Ed.2d 823 IMMIGRATION AND NATURALIZATION SERVICE, Petitionerv.Joseph Patrick DOHERTY. No. 90-925. Argued Oct. 16, 1991. Decided Jan. 15, 1992.  Syllabus Respondent Doherty, a citizen of both Ireland and the United Kingdom, was found guilty in absentia by a Northern I
- Hypothesis: the excerpt may contain overlapping issue-area language or insufficient context for the classifier; this is not verified from the text alone.

### Example 2 (test index 1)
- True: Due Process; SVC: Judicial Power; Legal-BERT: Judicial Power.
- Text (first 300 characters): 502 U.S. 367 112 S.Ct. 748 116 L.Ed.2d 867 Robert C. RUFO, Sheriff of Suffolk County, et  al., Petitioners,v.INMATES OF the SUFFOLK COUNTY JAIL et al.  Thomas C. RAPONE, Commissioner of Correction of  Massachusetts, Petitioner,  v.  INMATES OF the SUFFOLK COUNTY JAIL et al. Nos. 90-954, 90-1004. Arg
- Hypothesis: the excerpt may contain overlapping issue-area language or insufficient context for the classifier; this is not verified from the text alone.

### Example 3 (test index 8)
- True: Economic Activity; SVC: Federal Taxation; Legal-BERT: Economic Activity.
- Text (first 300 characters): 503 U.S. 30 112 S.Ct. 1011 117 L.Ed.2d 181 UNITED STATES, Petitionerv.NORDIC VILLAGE, INC., David O. Simon, Trustee. No. 90-1629. Argued Dec. 9, 1991. Decided Feb. 25, 1992.   Syllabus  After respondent Nordic Village, Inc., filed a petition for relief under Chapter 11 of the Bankruptcy Code, one of
- Hypothesis: the excerpt may contain overlapping issue-area language or insufficient context for the classifier; this is not verified from the text alone.

### Example 4 (test index 17)
- True: First Amendment; SVC: First Amendment; Legal-BERT: Criminal Procedure.
- Text (first 300 characters): 503 U.S. 159 112 S.Ct. 1093 117 L.Ed.2d 309 David DAWSON, Petitioner,v.DELAWARE. No. 90-6704. Argued Nov. 12, 1991. Decided March 9, 1992.   Syllabus  A Delaware jury convicted petitioner Dawson of first-degree murder and other crimes.  At the penalty hearing, the prosecution, inter alia, read a sti
- Hypothesis: the excerpt may contain overlapping issue-area language or insufficient context for the classifier; this is not verified from the text alone.

### Example 5 (test index 22)
- True: Judicial Power; SVC: Economic Activity; Legal-BERT: Economic Activity.
- Text (first 300 characters): 503 U.S. 258 112 S.Ct. 1311 117 L.Ed.2d 532 Robert G. HOLMES, Jr., Petitionerv.SECURITIES INVESTOR PROTECTION CORPORATION et al. No. 90-727. Argued Nov. 13, 1991. Decided March 24, 1992.   Syllabus  Pursuant to its authority under the Securities Investor Protection Act (SIPA), respondent Securities 
- Hypothesis: the excerpt may contain overlapping issue-area language or insufficient context for the classifier; this is not verified from the text alone.

### Example 6 (test index 23)
- True: Civil Rights; SVC: Criminal Procedure; Legal-BERT: Criminal Procedure.
- Text (first 300 characters): 503 U.S. 291 112 S.Ct. 1329 117 L.Ed.2d 559 UNITED STATES, Petitionerv.R.L.C. No. 90-1577. Argued Dec. 10, 1991. Decided March 24, 1992.   Syllabus  Because certain conduct of respondent R.L.C. at age 16 would have constituted the crime of involuntary manslaughter under 18 U.S.C. §§ 1112(a) and 1153
- Hypothesis: the excerpt may contain overlapping issue-area language or insufficient context for the classifier; this is not verified from the text alone.

### Example 7 (test index 24)
- True: Economic Activity; SVC: Unions; Legal-BERT: Economic Activity.
- Text (first 300 characters): 503 U.S. 318 112 S.Ct. 1344 117 L.Ed.2d 581 NATIONWIDE MUTUAL INSURANCE COMPANY, et al., Petitionersv.Robert T. DARDEN. No. 90-1802. Argued Jan. 21, 1992. Decided March 24, 1992.   Syllabus  Contracts between petitioners Nationwide Mutual Insurance Co. et al. and respondent Darden provided, among ot
- Hypothesis: the excerpt may contain overlapping issue-area language or insufficient context for the classifier; this is not verified from the text alone.

### Example 8 (test index 25)
- True: Judicial Power; SVC: Civil Rights; Legal-BERT: Civil Rights.
- Text (first 300 characters): 503 U.S. 347 112 S.Ct. 1360 118 L.Ed.2d 1 Sue SUTER, et al., Petitionersv.ARTIST M. et al. No. 90-1488. Argued Dec. 2, 1991. Decided March 25, 1992.   Syllabus  The Adoption Assistance and Child Welfare Act of 1980 provides that a State will be reimbursed by the Federal Government for certain expens
- Hypothesis: the excerpt may contain overlapping issue-area language or insufficient context for the classifier; this is not verified from the text alone.

### Example 9 (test index 26)
- True: Judicial Power; SVC: Economic Activity; Legal-BERT: Economic Activity.
- Text (first 300 characters): 503 U.S. 407 112 S.Ct. 1394 118 L.Ed.2d 52 NATIONAL RAILROAD PASSENGER CORPORATION, et al.,  Petitioners,v.BOSTON AND MAINE CORPORATION, et al.  INTERSTATE COMMERCE COMMISSION and United  States, Petitioners  v.  BOSTON AND MAINE CORPORATION, et al. Nos. 90-1419, 90-1769. Argued Jan. 13, 1992. Decid
- Hypothesis: the excerpt may contain overlapping issue-area language or insufficient context for the classifier; this is not verified from the text alone.

### Example 10 (test index 29)
- True: Economic Activity; SVC: Federal Taxation; Legal-BERT: Federal Taxation.
- Text (first 300 characters): 503 U.S. 393 112 S.Ct. 1386 118 L.Ed.2d 39 William BARNHILL, Petitionerv.Elliot JOHNSON, Trustee. No. 91-159. Argued Jan. 14, 1992. Decided March 25, 1992.   Syllabus  The debtor's check in payment of a bona fide debt was delivered to petitioner Barnhill in New Mexico on November 18 and honored by t
- Hypothesis: the excerpt may contain overlapping issue-area language or insufficient context for the classifier; this is not verified from the text alone.

## Validation-to-test macro-F1 change
- tfidf_linear_svc: validation 0.723840 to test 0.635154 (change 0.088686).
- transformer_legal_bert: validation 0.593541 to test 0.512374 (change 0.081166).

## Qualitative comparison with the paper
The paper's reported ordering is used only qualitatively. Its dataset construction, split, labels, preprocessing, and reported settings differ from this project; its transformer experiment reported no metric. Therefore these results are not a numerical reproduction of the paper.