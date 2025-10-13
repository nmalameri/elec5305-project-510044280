#!/usr/bin/env python3
# tools/06_gmm_plot_and_table.py
# Scan ALL gmm_<feat>/* runs, plot ROCs, print combined summary,
# and write per-attack/eval-at-dev-threshold CSVs for each run.

from pathlib import Path
import csv, json
import numpy as np
import matplotlib.pyplot as plt

def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None

def load_roc_csv(p: Path):
    if not p.exists(): return None
    fpr,tpr = [],[]
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            fpr.append(float(row["fpr"])); tpr.append(float(row["tpr"]))
    return np.array(fpr), np.array(tpr)

def load_scores_csv(p: Path):
    if not p.exists(): return [], np.array([]), np.array([])
    utts, scores, y = [], [], []
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            utts.append(row["utt_id"]); scores.append(float(row["score"])); y.append(int(row["label_int"]))
    return utts, np.array(scores, np.float32), np.array(y, np.int64)

def load_manifest(split: str):
    mpath = Path("results/manifests")/f"{split}.csv"
    idx = {}
    if not mpath.exists(): return idx
    with mpath.open() as f:
        r = csv.DictReader(f)
        for d in r:
            idx[d["utt_id"]] = {"label": d.get("label",""), "attack_id": d.get("attack_id","")}
    return idx

def eer_from_scores(scores, y):
    from sklearn.metrics import roc_curve
    if len(scores) == 0: return None, None
    fpr,tpr,thr = roc_curve(y, scores, pos_label=1)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])

def op_metrics_at_threshold(scores, y, thr):
    if len(scores) == 0: return None
    yhat = (scores >= thr).astype(int)
    tp = int(((y==1) & (yhat==1)).sum())
    tn = int(((y==0) & (yhat==0)).sum())
    fp = int(((y==0) & (yhat==1)).sum())
    fn = int(((y==1) & (yhat==0)).sum())
    P = max(1, int((y==1).sum())); N = max(1, int((y==0).sum()))
    tpr = tp / P; fpr = fp / N; tnr = tn / N; fnr = fn / P
    acc = (tp + tn) / (P + N); bal_acc = 0.5 * (tpr + tnr)
    return {"tpr":tpr,"fpr":fpr,"tnr":tnr,"fnr":fnr,"acc":acc,"bal_acc":bal_acc,
            "tp":tp,"tn":tn,"fp":fp,"fn":fn,"P":P,"N":N}

def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def per_attack_eval(run_dir: Path, dev_thr: float, eval_manifest: dict):
    # scores
    e_utts, e_scores, e_y = load_scores_csv(run_dir/"scores_eval.csv")
    if not e_utts: return
    eval_map = {u: {"score": s, "y": int(yi), "attack": (eval_manifest.get(u, {}).get("attack_id") or "").strip().upper()}
                for u, s, yi in zip(e_utts, e_scores, e_y)}
    eval_bona_scores = np.array([d["score"] for d in eval_map.values() if d["y"] == 0], np.float32)
    eval_bona_y      = np.zeros_like(eval_bona_scores, dtype=np.int64)
    attacks = sorted({d["attack"] for d in eval_map.values() if d["y"] == 1 and d["attack"] not in ("", "-")})

    rows_eer, rows_op = [], []
    for aid in attacks:
        S_pos = np.array([d["score"] for d in eval_map.values() if d["y"] == 1 and d["attack"] == aid], np.float32)
        Y_pos = np.ones_like(S_pos, dtype=np.int64)
        S = np.concatenate([S_pos, eval_bona_scores], axis=0)
        Y = np.concatenate([Y_pos, eval_bona_y], axis=0)
        ee, _ = eer_from_scores(S, Y) if len(S_pos) and len(eval_bona_scores) else (None, None)
        n_pos = int(len(S_pos)); n_neg = int(len(eval_bona_scores))
        ee_pct = "-" if ee is None else f"{100*ee:.2f}"
        rows_eer.append([aid, n_pos + n_neg, n_neg, n_pos, ee_pct])

        if dev_thr is not None and len(S_pos):
            op = op_metrics_at_threshold(S, Y, dev_thr)
            rows_op.append([aid, len(S), int((Y==0).sum()), int((Y==1).sum()),
                            dev_thr, op["tpr"], op["fpr"], op["tnr"], op["fnr"], op["acc"], op["bal_acc"]])

    if rows_eer:
        write_csv(run_dir/"per_attack_eer_eval.csv",
                  ["attack_id","n_total","n_bonafide","n_spoof","eer_percent"], rows_eer)
        print(f"[save] {run_dir/'per_attack_eer_eval.csv'}")
    if dev_thr is not None and rows_op:
        write_csv(run_dir/"per_attack_op_eval_at_dev_threshold.csv",
                  ["attack_id","n_total","n_bonafide","n_spoof","thr_dev_minEER",
                   "TPR","FPR","TNR","FNR","ACC","BAL_ACC"], rows_op)
        print(f"[save] {run_dir/'per_attack_op_eval_at_dev_threshold.csv'}")

def process_run(feat: str, run_dir: Path, eval_manifest: dict, combined_rows: list):
    m = load_json(run_dir/"metrics.json")
    if not m:
        print(f"[skip] {run_dir} missing metrics.json")
        return
    dev = m["dev"]; ev = m["eval"]; run_tag = m.get("run_tag","")

    # ROC plots (ensure PNGs exist)
    for split in ["dev","eval"]:
        roc = load_roc_csv(run_dir/f"roc_{split}.csv")
        if roc is not None:
            fpr,tpr = roc
            plt.figure()
            plt.plot(fpr, tpr, label=f"{feat.upper()} {split.upper()} ({run_tag})")
            plt.plot([0,1],[0,1], linestyle="--")
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC: {feat} ({split}) [{run_tag}]")
            plt.legend()
            figp = run_dir / f"roc_{split}.png"
            figp.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(figp, dpi=140, bbox_inches="tight"); plt.close()
            print(f"[plot] {figp}")

    # Global eval @ DEV threshold
    e_utts, e_scores, e_y = load_scores_csv(run_dir/"scores_eval.csv")
    if len(e_utts) and dev.get("thr", None) is not None:
        op = op_metrics_at_threshold(e_scores, e_y, dev["thr"])
        savep = run_dir/"op_eval_at_dev_threshold.csv"
        write_csv(savep,
                  ["n_eval","n_bonafide","n_spoof","thr_dev_minEER","TPR","FPR","TNR","FNR","ACC","BAL_ACC"],
                  [[len(e_utts), int((e_y==0).sum()), int((e_y==1).sum()),
                    dev["thr"], op["tpr"], op["fpr"], op["tnr"], op["fnr"], op["acc"], op["bal_acc"]]])
        print(f"[save] {savep}")
        print(f"[eval@DEVthr][{feat.upper()}][{run_tag}] ACC={100*op['acc']:.2f}%  "
              f"BAL_ACC={100*op['bal_acc']:.2f}%  TPR={100*op['tpr']:.2f}%  FPR={100*op['fpr']:.2f}%")

    # Per-attack CSVs
    per_attack_eval(run_dir, dev.get("thr", None), eval_manifest)

    # Row for combined console/CSV summary
    def pct(x): return "-" if x is None else f"{100*x:.2f}%"
    combined_rows.append([
        feat.upper(), run_tag, pct(dev["eer"]), pct(dev["acc"]), pct(ev["eer"]), pct(ev["acc"])
    ])

def main():
    eval_manifest = load_manifest("eval")
    combined_rows = []

    for feat in ["lfcc","mfcc"]:
        base = Path("results/models")/f"gmm_{feat}"
        if not base.exists():
            print(f"[skip] {base} missing")
            continue
        for d in sorted(base.iterdir()):
            if d.name == "latest": continue
            if not d.is_dir(): continue
            process_run(feat, d, eval_manifest, combined_rows)

    if combined_rows:
        print("\n# Baseline GMM results (all runs)\n")
        print("| Feature | Run Tag | DEV EER | DEV ACC | EVAL EER | EVAL ACC |")
        print("|---|---|---:|---:|---:|---:|")
        for row in combined_rows:
            print("| " + " | ".join(row) + " |")

        out_csv = Path("results/models")/"gmm_summary_all_runs.csv"
        write_csv(out_csv,
                  ["feature","run_tag","dev_eer_percent","dev_acc_percent","eval_eer_percent","eval_acc_percent"],
                  combined_rows)
        print(f"\n[save] {out_csv}")

if __name__ == "__main__":
    main()
