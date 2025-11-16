"""
10_demo_examples.py

Purpose:
    To demonstrate how the trained models (GMM_LFCC, GMM_MFCC, CNN_LFCC, CNN_MFCC)
    classify a spoof and a bona fide example from the ASVspoof 2019 LA EVAL set.

Functionality:
    - Loading scores_eval.csv and metrics.json for each model.
    - Automatically select:
        * one spoof example (label_int = 1)
        * one bona fide example (label_int = 0)
      from the evaluation scores.
    - For each example and for each model, print:
        * utt_id
        * ground truth label
        * score
        * DEV EER threshold
        * predicted label (bona fide / spoof)
        * whether the prediction is correct.

Usage:
    1. Ensure you have already run the GMM and CNN scripts (05 and 07).
    2. Edit the MODEL_RUNS dictionary below so that each path points to the
       correct run_tag directory for your experiments.
    3. Run:
        python tools/10_demo_examples.py

Notes:
    - This script does NOT re-run feature extraction or training.
      It only uses the saved scores and thresholds.
"""

import json
import os
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------
MODEL_RUNS = {
    "GMM_LFCC": "results/models/gmm_lfcc/K64_cap100_S6000_F800000_20251116-1640",
    "GMM_MFCC": "results/models/gmm_mfcc/K64_cap100_S6000_F800000_20251116-1649",
    "CNN_LFCC": "results/models/cnn_lfcc/cnn_T400_E15_B32_S6000_BAL_20251116-1823",
    "CNN_MFCC": "results/models/cnn_mfcc/cnn_T400_E15_B32_S6000_BAL_20251116-1659",
}


# -----------------------------
# Helpers
# -----------------------------
def load_metrics(run_dir):
    """
    Loading metrics.json from a run directory and extracting the DEV threshold.
    """
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics.json not found at: {metrics_path}")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    if "dev" not in metrics or "thr" not in metrics["dev"]:
        raise KeyError(
            f"Expected metrics['dev']['thr'] in {metrics_path}, "
            f"found top-level keys: {list(metrics.keys())}"
        )

    thr = float(metrics["dev"]["thr"])
    return metrics, thr


def load_scores_eval(run_dir):
    """Loading scores_eval.csv from the run directory."""
    scores_path = os.path.join(run_dir, "scores_eval.csv")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"scores_eval.csv not found at: {scores_path}")
    df = pd.read_csv(scores_path)
    # expected columns: utt_id,score,label_int
    required_cols = {"utt_id", "score", "label_int"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"scores_eval.csv at {scores_path} is missing one of the required "
            f"columns: {required_cols}, got: {set(df.columns)}"
        )
    return df


def pick_example(df, label_int, prefer_correct=True, thr=None):
    """
    Picks a single example from df with the desired label_int (0 or 1).

    """
    df_label = df[df["label_int"] == label_int]

    if df_label.empty:
        raise ValueError(f"No examples found with label_int = {label_int}")

    if prefer_correct and thr is not None:
        # for label_int = 1 (spoof), correct if score > thr
        # for label_int = 0 (bona fide), correct if score <= thr
        if label_int == 1:
            df_correct = df_label[df_label["score"] > thr]
        else:
            df_correct = df_label[df_label["score"] <= thr]

        if not df_correct.empty:
            return df_correct.sample(1).iloc[0]

    # fall back to any example if no correct ones
    return df_label.sample(1).iloc[0]


def predict_label(score, thr):
    """returns 1 for spoof if score > thr, else 0 for bona fide."""
    return 1 if score > thr else 0


def label_str(label_int):
    """converts 0/1 integer to text label."""
    return "bona fide" if label_int == 0 else "spoof"


# -----------------------------
# Main demonstration logic
# -----------------------------

def main():
    # loading metrics and scores for each model
    model_data = {}
    for name, run_dir in MODEL_RUNS.items():
        print(f"\nLoading model: {name}")
        print(f"  Run directory: {run_dir}")
        metrics, thr = load_metrics(run_dir)
        scores_df = load_scores_eval(run_dir)
        model_data[name] = {
            "run_dir": run_dir,
            "metrics": metrics,
            "thr": thr,
            "scores": scores_df,
        }

    # choosing a reference model to select example utterances
    ref_model_name = "GMM_MFCC"
    if ref_model_name not in model_data:
        raise KeyError(
            f"Reference model {ref_model_name} not found in MODEL_RUNS. "
            f"Available: {list(model_data.keys())}"
        )

    ref = model_data[ref_model_name]
    ref_thr = ref["thr"]
    ref_scores = ref["scores"]

    print("\nSelecting example utterances from reference model:", ref_model_name)

    # picking one spoof and one bona fide example from the reference model
    spoof_row = pick_example(ref_scores, label_int=1, prefer_correct=True, thr=ref_thr)
    bona_row = pick_example(ref_scores, label_int=0, prefer_correct=True, thr=ref_thr)

    spoof_utt = spoof_row["utt_id"]
    bona_utt = bona_row["utt_id"]

    print(f"  Chosen spoof example:     {spoof_utt}")
    print(f"  Chosen bona fide example: {bona_utt}")

    print("\n" + "=" * 70)
    print("EXAMPLE 1: SPOOF UTTERANCE")
    print("=" * 70)
    print(f"utt_id: {spoof_utt}")

    # for each model, show how it scores this spoof example
    for name, data in model_data.items():
        thr = data["thr"]
        df = data["scores"]
        row = df[df["utt_id"] == spoof_utt]
        if row.empty:
            print(f"\n[{name}] No entry for utt_id {spoof_utt} in scores_eval.csv.")
            continue
        row = row.iloc[0]
        score = float(row["score"])
        true_label = int(row["label_int"])
        pred_label = predict_label(score, thr)

        print(f"\n[{name}]")
        print(f"  Threshold (DEV EER): {thr:.6f}")
        print(f"  Score (EVAL):        {score:.6f}")
        print(f"  Ground truth:        {label_str(true_label)} (label_int={true_label})")
        print(f"  Predicted:           {label_str(pred_label)} (pred={pred_label})")
        print(f"  Correct:             {pred_label == true_label}")

    print("\n" + "=" * 70)
    print("EXAMPLE 2: BONA FIDE UTTERANCE")
    print("=" * 70)
    print(f"utt_id: {bona_utt}")

    # for each model, show how it scores this bona fide example
    for name, data in model_data.items():
        thr = data["thr"]
        df = data["scores"]
        row = df[df["utt_id"] == bona_utt]
        if row.empty:
            print(f"\n[{name}] No entry for utt_id {bona_utt} in scores_eval.csv.")
            continue
        row = row.iloc[0]
        score = float(row["score"])
        true_label = int(row["label_int"])
        pred_label = predict_label(score, thr)

        print(f"\n[{name}]")
        print(f"  Threshold (DEV EER): {thr:.6f}")
        print(f"  Score (EVAL):        {score:.6f}")
        print(f"  Ground truth:        {label_str(true_label)} (label_int={true_label})")
        print(f"  Predicted:           {label_str(pred_label)} (pred={pred_label})")
        print(f"  Correct:             {pred_label == true_label}")


if __name__ == "__main__":
    main()