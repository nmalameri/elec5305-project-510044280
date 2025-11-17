"""
rawnetlite_per_attack.py

Compute per-attack EER and per-attack operational metrics (ACC/TPR/FPR)
for a RawNetLite run, using:
    - scores_eval.csv        (from rawnetlite_infer.py)
    - metrics.json           (global DEV threshold)
    - eval manifest (utt_id -> attack_id mapping)

Outputs (in run_dir):
    - per_attack_eer_eval.csv
    - per_attack_op_eval_at_dev_threshold.csv

Usage:
    python pretrained_models/rawnetlite/rawnetlite_per_attack.py \
        --run_dir results/models/rawnetlite/rawnetlite_pretrained_XXXX \
        --eval_manifest results/manifests/eval.csv
"""

import argparse
import csv
import json
import os
import numpy as np
from collections import defaultdict


# -------------------------------------------------------
# EER utility (same method used in your system)
# -------------------------------------------------------
def compute_eer(scores, labels):
    """Compute EER given scores and binary labels (1=spoof, 0=bonafide)."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    # Sort scores descending
    idx = np.argsort(-scores)
    scores = scores[idx]
    labels = labels[idx]

    P = np.sum(labels == 1)
    N = np.sum(labels == 0)

    if P == 0 or N == 0:
        return float("nan")

    tprs = []
    fprs = []

    tp = 0
    fp = 0
    prev = None
    for s, y in zip(scores, labels):
        if prev is None or s != prev:
            tprs.append(tp / P)
            fprs.append(fp / N)
            prev = s
        if y == 1:
            tp += 1
        else:
            fp += 1

    tprs.append(tp / P)
    fprs.append(fp / N)

    tprs = np.array(tprs)
    fprs = np.array(fprs)
    fnrs = 1.0 - tprs

    idx = np.argmin(np.abs(fnrs - fprs))
    eer = (fnrs[idx] + fprs[idx]) / 2
    return float(eer)


# -------------------------------------------------------
# LOADERS
# -------------------------------------------------------
def load_scores_eval(path):
    """Load scores_eval.csv -> dict: utt_id -> (score, label_int)."""
    scores = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores[row["utt_id"]] = (float(row["score"]), int(row["label_int"]))
    return scores


def load_utt2attack(manifest_path):
    """Return dict utt_id -> attack_id from eval manifest."""
    utt2attack = {}
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            utt2attack[row["utt_id"]] = row["attack_id"]
    return utt2attack


def load_dev_threshold(metrics_path):
    """Load dev threshold from metrics.json."""
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return float(metrics["dev"]["thr"])


# -------------------------------------------------------
# MAIN LOGIC
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True,
                        help="Folder containing scores_eval.csv + metrics.json")
    parser.add_argument("--eval_manifest", required=True,
                        help="Path to eval manifest CSV with attack_id column")
    args = parser.parse_args()

    scores_path = os.path.join(args.run_dir, "scores_eval.csv")
    metrics_path = os.path.join(args.run_dir, "metrics.json")

    if not os.path.exists(scores_path):
        raise FileNotFoundError(scores_path)
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(metrics_path)
    if not os.path.exists(args.eval_manifest):
        raise FileNotFoundError(args.eval_manifest)

    print("Loading scores and thresholds...")
    scores = load_scores_eval(scores_path)
    utt2attack = load_utt2attack(args.eval_manifest)
    dev_thr = load_dev_threshold(metrics_path)

    # Group scores by attack
    attack_scores = defaultdict(list)
    attack_labels = defaultdict(list)

    for utt_id, (score, label) in scores.items():
        attack = utt2attack.get(utt_id)
        if attack is None:
            continue
        attack_scores[attack].append(score)
        attack_labels[attack].append(label)

    # Prepare outputs
    eer_rows = []
    op_rows = []

    print("Computing per-attack metrics...")

    for attack in sorted(attack_scores.keys()):
        sc = np.array(attack_scores[attack])
        lb = np.array(attack_labels[attack])

        n_total = len(sc)
        n_bf = int((lb == 0).sum())
        n_sp = int((lb == 1).sum())

        # EER
        eer = compute_eer(sc, lb)
        eer_percent = eer * 100 if not np.isnan(eer) else float("nan")

        # Operational metrics @ global DEV threshold
        preds = (sc > dev_thr).astype(int)
        acc = (preds == lb).sum() / n_total if n_total > 0 else float("nan")

        if n_sp > 0:
            tpr = ((preds == 1) & (lb == 1)).sum() / n_sp
        else:
            tpr = float("nan")

        if n_bf > 0:
            fpr = ((preds == 1) & (lb == 0)).sum() / n_bf
        else:
            fpr = float("nan")

        eer_rows.append({
            "attack_id": attack,
            "n_total": n_total,
            "n_bonafide": n_bf,
            "n_spoof": n_sp,
            "eer_percent": eer_percent,
        })

        op_rows.append({
            "attack_id": attack,
            "n_total": n_total,
            "n_bonafide": n_bf,
            "n_spoof": n_sp,
            "acc": acc,
            "tpr": tpr,
            "fpr": fpr,
        })

    # Write outputs
    out_eer = os.path.join(args.run_dir, "per_attack_eer_eval.csv")
    out_op = os.path.join(args.run_dir, "per_attack_op_eval_at_dev_threshold.csv")

    with open(out_eer, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["attack_id", "n_total", "n_bonafide", "n_spoof", "eer_percent"]
        )
        writer.writeheader()
        writer.writerows(eer_rows)

    with open(out_op, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["attack_id", "n_total", "n_bonafide", "n_spoof", "acc", "tpr", "fpr"]
        )
        writer.writeheader()
        writer.writerows(op_rows)

    print(f"Wrote: {out_eer}")
    print(f"Wrote: {out_op}")
    print("Done.")


if __name__ == "__main__":
    main()