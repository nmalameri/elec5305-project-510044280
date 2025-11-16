# Example Inference Results Across All Models

This document presents two real input utterances from the ASVspoof 2019 Logical Access (LA) evaluation set — one spoof and one bona fide.  
Each utterance is evaluated by all four trained systems:

- GMM_LFCC  
- GMM_MFCC  
- CNN_LFCC  
- CNN_MFCC  

For each model, the table shows:
- the model score assigned to the utterance (from `scores_eval.csv`)
- the decision threshold learned from the DEV set (from `metrics.json`)
- the predicted class label
- whether the prediction matches the ground truth

These examples concretely illustrate how the models behave on real ASVspoof inputs.

---

## 🔊 Example 1 — Spoof Utterance  
**utt_id: `LA_E_3667402`**  
Ground truth label: **spoof**

| Model      | Score (EVAL) | DEV Thr    | Predicted | Correct | Explanation |
|------------|--------------|------------|-----------|---------|-------------|
| **GMM_LFCC** | 61.912170    | -1.921286  | spoof     | ✔️ | Score far above threshold → confidently spoof |
| **GMM_MFCC** | 12.622248    | -1.282040  | spoof     | ✔️ | Strong separation → correct spoof detection |
| **CNN_LFCC** | -8.990713    | 1.087710   | bona fide | ❌ | Score below a high threshold → model sensitivity to distribution shift |
| **CNN_MFCC** | -2.507898    | -2.443451  | bona fide | ❌ | Score slightly below threshold → borderline misclassification |

---

## 🔈 Example 2 — Bona Fide Utterance  
**utt_id: `LA_E_6276020`**  
Ground truth label: **bona fide**

| Model      | Score (EVAL) | DEV Thr    | Predicted | Correct | Explanation |
|------------|--------------|------------|-----------|---------|-------------|
| **GMM_LFCC** | -9.372585    | -1.921286  | bona fide | ✔️ | Deep negative score → confidently bona fide |
| **GMM_MFCC** | -2.889600    | -1.282040  | bona fide | ✔️ | Score below threshold → correct decision |
| **CNN_LFCC** | -10.581306   | 1.087710   | bona fide | ✔️ | Correct despite high threshold |
| **CNN_MFCC** | -9.861833    | -2.443451  | bona fide | ✔️ | Consistent bona fide classification |

---

## 📘 Interpretation

- **GMM models (LFCC & MFCC)**  
  - Correctly classify both spoof and bona fide examples.  
  - Show stable behaviour consistent with their strong evaluation performance.

- **CNN models (LFCC & MFCC)**  
  - Correctly classify the bona fide example.  
  - Misclassify the spoof example, illustrating their reduced robustness on EVAL compared to DEV.  

These examples are representative of the broader trends observed in the quantitative analysis.
