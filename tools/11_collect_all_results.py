"""
11_collect_all_results.py

To collect headline metrics and ROC curves for ALL models and save them
into results/summary/:

  - headline_all_models.csv
  - headline_all_models.md
  - roc_eval_all_models.png

It assumes each model run directory contains:
  - metrics.json
  - scores_eval.csv   (with columns: utt_id,score,label_int)

You MUST edit the RUNS dictionary to point to your actual run dirs.

Usage:
    python tools/11_collect_all_results.py
"""

import csv
import json
import os

from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------
# EDIT THIS: paths to your model run directories
# ---------------------------------------------------

RUNS: Dict[str, str] = {
    # fill with your real paths:
    "GMM_LFCC": "results/models/gmm_lfcc/K64_cap100_S6000_F800000_20251116-1640",
    "GMM_MFCC": "results/models/gmm_mfcc/K64_cap100_S6000_F800000_20251116-1649",
    "CNN_LFCC": "results/models/cnn_lfcc/cnn_T400_E15_B32_S6000_BAL_20251116-1823",
    "CNN_MFCC": "results/models/cnn_mfcc/cnn_T400_E15_B32_S6000_BAL_20251116-1659",
    "RawNetLite": "results/models/rawnetlite/rawnetlite_pretrained_20251117-144910",
    "RawNet2": "results/models/rawnet2/rawnet2_pretrained_20251117-153319",
    "AASIST_LA": "results/models/aasist/aasist_LA_pretrained_20251117-160306",
}

SUMMARY_DIR = "results/summary"


# -----------------
# Data structures
# -----------------

@dataclass
class ModelMetrics:
    name: str
    dev_eer: float
    dev_acc: float
    dev_thr: float
    eval_eer: float
    eval_acc: float
    run_dir: str


# -----------
# Utilities
# -----------

def load_metrics(run_dir: str, name: str) -> ModelMetrics:
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics.json not found for {name} at {metrics_path}")

    with open(metrics_path, "r") as f:
        m = json.load(f)

    dev = m.get("dev", {})
    ev = m.get("eval", {})

    return ModelMetrics(
        name=name,
        dev_eer=float(dev.get("eer", float("nan"))),
        dev_acc=float(dev.get("acc", float("nan"))),
        dev_thr=float(dev.get("thr", 0.0)),
        eval_eer=float(ev.get("eer", float("nan"))),
        eval_acc=float(ev.get("acc", float("nan"))),
        run_dir=run_dir,
    )


def load_scores_eval(run_dir: str):
    scores_path = os.path.join(run_dir, "scores_eval.csv")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"scores_eval.csv not found at {scores_path}")
    scores = []
    labels = []
    with open(scores_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row["score"]))
            labels.append(int(row["label_int"]))
    return np.array(scores, dtype=np.float64), np.array(labels, dtype=np.int32)


def compute_roc(scores: np.ndarray, labels: np.ndarray):
    """
    Computing ROC curve given scores and labels (1=spoof, 0=bonafide).

    Returns:
        fpr, tpr
    """
    # sort by descending score
    idx = np.argsort(-scores)
    s = scores[idx]
    l = labels[idx]

    P = (l == 1).sum()
    N = (l == 0).sum()

    tprs = []
    fprs = []

    tp = 0
    fp = 0
    prev = None

    for i in range(len(s)):
        if prev is None or s[i] != prev:
            tprs.append(tp / P if P > 0 else 0.0)
            fprs.append(fp / N if N > 0 else 0.0)
            prev = s[i]
        if l[i] == 1:
            tp += 1
        else:
            fp += 1

    tprs.append(tp / P if P > 0 else 0.0)
    fprs.append(fp / N if N > 0 else 0.0)

    return np.array(fprs), np.array(tprs)


# --------------------------------
# Summary table & ROC plotting
# --------------------------------

def write_headline_table(metrics_list: List[ModelMetrics]):
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    csv_path = os.path.join(SUMMARY_DIR, "headline_all_models.csv")
    md_path = os.path.join(SUMMARY_DIR, "headline_all_models.md")

    # CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "dev_eer_percent",
            "dev_acc_percent",
            "eval_eer_percent",
            "eval_acc_percent",
            "run_dir",
        ])
        for m in metrics_list:
            writer.writerow([
                m.name,
                f"{m.dev_eer * 100:.4f}",
                f"{m.dev_acc * 100:.2f}",
                f"{m.eval_eer * 100:.4f}",
                f"{m.eval_acc * 100:.2f}",
                m.run_dir,
            ])

    # markdown summary
    with open(md_path, "w") as f:
        f.write("# Headline Results for All Models\n\n")
        f.write("| Model | DEV EER (%) | DEV ACC (%) | EVAL EER (%) | EVAL ACC (%) |\n")
        f.write("|-------|-------------|-------------|--------------|--------------|\n")
        for m in metrics_list:
            f.write(
                f"| {m.name} | "
                f"{m.dev_eer * 100:.2f} | "
                f"{m.dev_acc * 100:.2f} | "
                f"{m.eval_eer * 100:.2f} | "
                f"{m.eval_acc * 100:.2f} |\n"
            )

    print(f"Headline CSV written to: {csv_path}")
    print(f"Headline Markdown written to: {md_path}")


def plot_eval_roc(metrics_list: List[ModelMetrics]):
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    plt.figure(figsize=(7, 7))

    for m in metrics_list:
        scores, labels = load_scores_eval(m.run_dir)
        fpr, tpr = compute_roc(scores, labels)
        plt.plot(fpr, tpr, label=f"{m.name} (EER={m.eval_eer*100:.1f}%)")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("EVAL ROC – All Models")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)

    out_path = os.path.join(SUMMARY_DIR, "roc_eval_all_models.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"EVAL ROC figure written to: {out_path}")


# -------
# Main
# -----

def main():
    if not RUNS:
        raise RuntimeError(
            "Please edit RUNS at the top of this script to include your model run directories."
        )

    metrics_list: List[ModelMetrics] = []
    for name, run_dir in RUNS.items():
        print(f"Loading metrics for {name} from {run_dir}")
        m = load_metrics(run_dir, name)
        metrics_list.append(m)

    # writing headline summary
    write_headline_table(metrics_list)

    # plotting EVAL ROC overlay
    plot_eval_roc(metrics_list)


if __name__ == "__main__":
    main()