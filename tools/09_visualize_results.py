"""
09_visualize_results.py

Purpose:
    To visualise ROC curves across all selected models on DEV and EVAL.

Functionality:
    - Automatically find or load specific run_tag directories
    - Loading ROC CSV samples created in Step 05 and Step 07
    - Generating overlay ROC plots (DEV and EVAL)
    - Saving figures for use in reports and README

Inputs:
    results/models/*/<run_tag>/

Outputs:
    results/summary/roc_dev_overlay.png
    results/summary/roc_eval_overlay.png

Notes:
    To compare model behaviour visually.
"""


from __future__ import annotations
from pathlib import Path
import argparse, json, csv, time
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# Helpers
# --------------------------
def _resolve_model_dir(parent: Path) -> Path | None:
    """
    Resolution order:
    1) parent/latest symlink (if points to a run dir with metrics.json)
    2) newest subdir under parent that has metrics.json
    3) parent itself if it has metrics.json
    """
    if not parent.exists():
        return None
    latest = parent / "latest"
    try:
        if latest.exists():
            tgt = latest.resolve()
            if (tgt / "metrics.json").exists():
                return tgt
    except Exception:
        pass
    candidates = [p for p in parent.iterdir()
                  if p.is_dir() and (p / "metrics.json").exists()]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return parent if (parent / "metrics.json").exists() else None

def _safe_json(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def _safe_op_csv(p: Path) -> dict:
    # expects columns: acc,tpr,fpr,bal_acc,thr_dev,[eval_eer],[dev_eer]
    if not p.exists():
        return {}
    try:
        with p.open() as f:
            r = csv.DictReader(f)
            rows = list(r)
        return rows[0] if rows else {}
    except Exception:
        return {}

def _safe_load_roc(p: Path):
    if not p.exists():
        return None
    try:
        fpr, tpr = [], []
        with p.open() as f:
            r = csv.DictReader(f)
            for row in r:
                fpr.append(float(row["fpr"]))
                tpr.append(float(row["tpr"]))
        if not fpr:
            return None
        return np.array(fpr), np.array(tpr)
    except Exception:
        return None

def _gather_models(args):
    models = []
    # Explicit path > fallback to latest/newest
    mapping = [
        ("GMM-LFCC", args.gmm_lfcc, Path("results/models/gmm_lfcc")),
        ("GMM-MFCC", args.gmm_mfcc, Path("results/models/gmm_mfcc")),
        ("CNN-LFCC", args.cnn_lfcc, Path("results/models/cnn_lfcc")),
        ("CNN-MFCC", args.cnn_mfcc, Path("results/models/cnn_mfcc")),
    ]
    for label, explicit, base in mapping:
        if explicit:
            d = Path(explicit)
            if (d / "metrics.json").exists():  # allow pointing directly at a run dir
                models.append((label, d))
            else:
                # also allowing parent family folder (it will fallback to latest/newest)
                resolved = _resolve_model_dir(d)
                if resolved:
                    models.append((label, resolved))
        else:
            d = _resolve_model_dir(base)
            if d:
                models.append((label, d))
    return models

def _pretty_num(x, pct=False):
    if x is None:
        return "-"
    try:
        if pct:
            return f"{100*float(x):.2f}%"
        return f"{float(x):.4f}"
    except Exception:
        return "-"

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Compare selected ASVspoof runs and draw overlay ROC plots.")
    ap.add_argument("--gmm-lfcc", help="Path to a specific GMM-LFCC run dir (or family folder).")
    ap.add_argument("--gmm-mfcc", help="Path to a specific GMM-MFCC run dir (or family folder).")
    ap.add_argument("--cnn-lfcc", help="Path to a specific CNN-LFCC run dir (or family folder).")
    ap.add_argument("--cnn-mfcc", help="Path to a specific CNN-MFCC run dir (or family folder).")
    ap.add_argument("--out-dir", default="results/models/summary",
                    help="Where to save overlay plots and combined CSV.")
    args = ap.parse_args()

    models = _gather_models(args)
    if not models:
        print("No models found. Train GMM/CNN or point to runs with flags.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M")

    # collecting numbers + draw overlay ROCs
    summary_rows = []
    have_dev_roc = False
    have_eval_roc = False
    plt.figure()
    for label, run_dir in models:
        m = _safe_json(run_dir / "metrics.json")
        feat   = m.get("feature", "?")
        dev_eer = m.get("dev", {}).get("eer", None)
        dev_thr = m.get("dev", {}).get("thr", None)
        ev_eer  = m.get("eval", {}).get("eer", None)
        ev_acc  = m.get("eval", {}).get("acc", None)
        ev_tpr  = m.get("eval", {}).get("tpr", None)
        ev_fpr  = m.get("eval", {}).get("fpr", None)
        run_tag = run_dir.name

        # op point (CSV priority if present)
        op = _safe_op_csv(run_dir / "op_eval_at_dev_threshold.csv")
        if op:
            ev_acc = float(op.get("acc", ev_acc or 0.0))
            ev_tpr = float(op.get("tpr", ev_tpr or 0.0))
            ev_fpr = float(op.get("fpr", ev_fpr or 0.0))
            dev_thr = float(op.get("thr_dev", dev_thr or 0.0))
            # prefers eval_eer in OP if present; otherwise keep metrics.json’s value
            try:
                ev_eer = float(op.get("eval_eer", ev_eer))
            except Exception:
                pass

        # Dev/Eval ROC files (for overlay plotting)
        roc_dev  = _safe_load_roc(run_dir / "roc_dev.csv")
        roc_eval = _safe_load_roc(run_dir / "roc_eval.csv")

        # DEV overlay
        if roc_dev is not None:
            fpr_d, tpr_d = roc_dev
            plt.figure(1)
            plt.plot(fpr_d, tpr_d, label=f"{label} ({feat}) — DEV")
            have_dev_roc = True

        # EVAL overlay
        if roc_eval is not None:
            fpr_e, tpr_e = roc_eval
            plt.figure(2)
            plt.plot(fpr_e, tpr_e, label=f"{label} ({feat}) — EVAL")
            have_eval_roc = True

        summary_rows.append({
            "model": label,
            "feature": feat,
            "run_dir": str(run_dir),
            "dev_eer": dev_eer,
            "dev_thr": dev_thr,
            "eval_eer": ev_eer,
            "eval_acc": ev_acc,
            "eval_tpr": ev_tpr,
            "eval_fpr": ev_fpr,
            "run_tag": run_tag
        })

    # Saving overlay plots
    if have_dev_roc:
        plt.figure(1)
        plt.plot([0,1],[0,1],"--", linewidth=1)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC — DEV (overlay)")
        plt.legend()
        p = out_dir / f"roc_dev_overlay_{ts}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"[plot] {p}")

    if have_eval_roc:
        plt.figure(2)
        plt.plot([0,1],[0,1],"--", linewidth=1)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC — EVAL (overlay)")
        plt.legend()
        p = out_dir / f"roc_eval_overlay_{ts}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"[plot] {p}")

    # printing Markdown table
    print("\n# Results (selected runs)\n")
    print("| Model | Feature | Run Tag | DEV EER | DEV THR | EVAL EER | EVAL ACC | EVAL TPR | EVAL FPR |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in summary_rows:
        print("| {model} | {feature} | {run_tag} | {dev_eer} | {dev_thr} | {eval_eer} | {eval_acc} | {eval_tpr} | {eval_fpr} |".format(
            model=r["model"],
            feature=r["feature"],
            run_tag=r["run_tag"],
            dev_eer=_pretty_num(r["dev_eer"], pct=True),
            dev_thr=_pretty_num(r["dev_thr"], pct=False),
            eval_eer=_pretty_num(r["eval_eer"], pct=True),
            eval_acc=_pretty_num(r["eval_acc"], pct=True),
            eval_tpr=_pretty_num(r["eval_tpr"], pct=True),
            eval_fpr=_pretty_num(r["eval_fpr"], pct=True),
        ))

    # saving combined CSV
    csv_path = out_dir / f"summary_{ts}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model","feature","run_tag","run_dir","dev_eer","dev_thr","eval_eer","eval_acc","eval_tpr","eval_fpr"])
        for r in summary_rows:
            w.writerow([
                r["model"], r["feature"], r["run_tag"], r["run_dir"],
                r["dev_eer"], r["dev_thr"], r["eval_eer"], r["eval_acc"], r["eval_tpr"], r["eval_fpr"]
            ])
    print(f"\n[save] {csv_path}")

if __name__ == "__main__":
    main()
