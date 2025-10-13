#!/usr/bin/env python3
from pathlib import Path
import json, csv, shutil

# === EDIT THESE to the run directories you want in the paper ===
RUNS = {
    "LFCC+GMM":  "results/models/gmm_lfcc/K64_cap100_S6000_F800000_20251013-1226",
    "MFCC+GMM":  "results/models/gmm_mfcc/K64_cap100_S6000_F800000_20251013-1237",
    "LFCC+CNN":  "results/models/cnn_lfcc/cnn_T400_E15_B32_S6000_BAL_20251013-1649",
    "MFCC+CNN":  "results/models/cnn_mfcc/cnn_T400_E15_B32_S6000_BAL_20251013-1734",
}

OUTDIR = Path("results/summary")
FIGDIR = Path("figs")
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

def read_metrics(run_dir: Path):
    p = run_dir/"metrics.json"
    if not p.exists(): return None
    m = json.loads(p.read_text())
    # normalize fields
    dev_eer = m.get("dev", {}).get("eer")
    dev_thr = m.get("dev", {}).get("thr")
    ev      = m.get("eval", {})
    ev_eer  = ev.get("eer")
    ev_acc  = ev.get("acc")
    return {
        "dev_eer": dev_eer, "dev_thr": dev_thr,
        "eval_eer": ev_eer, "eval_acc": ev_acc,
        "run_dir": str(run_dir)
    }

def fmt_pct(x):
    return "-" if x is None else f"{100*float(x):.2f}%"

# --- Headline table from metrics.json ---
headline_rows = []
for name, path in RUNS.items():
    if not path: continue
    m = read_metrics(Path(path))
    if not m: continue
    headline_rows.append([
        name,
        fmt_pct(m["dev_eer"]),
        fmt_pct(m["eval_eer"]),
        fmt_pct(m["eval_acc"]),
        m["run_dir"]
    ])

with (OUTDIR/"headline.md").open("w") as f:
    f.write("| System | Dev EER | Eval EER | Eval ACC | Run Dir |\n")
    f.write("|---|---:|---:|---:|---|\n")
    for r in headline_rows:
        f.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | `{r[4]}` |\n")
print(f"[save] {OUTDIR/'headline.md'}")

with (OUTDIR/"headline.tex").open("w") as f:
    f.write("\\begin{table}[H]\\centering\n")
    f.write("\\caption{Headline performance. DEV threshold (min-EER) applied to EVAL.}\n")
    f.write("\\label{tab:headline}\n")
    f.write("\\begin{tabular}{lccc}\\toprule\n")
    f.write("System & Dev EER (\\%) & Eval EER (\\%) & Eval ACC (\\%) \\\\ \\midrule\n")
    for name, dev_eer, ev_eer, ev_acc, _ in headline_rows:
        def onlynum(p): return "-" if p=="-" else p[:-1]
        f.write(f"{name} & {onlynum(dev_eer)} & {onlynum(ev_eer)} & {onlynum(ev_acc)} \\\\\n")
    f.write("\\bottomrule\\end{tabular}\\end{table}\n")
print(f"[save] {OUTDIR/'headline.tex'}")

# --- Per-attack EER (side-by-side merge) ---
# read each per_attack_eer_eval.csv into dict: attack_id -> eer
attacks = set()
per_attack = {}
for name, path in RUNS.items():
    if not path: continue
    csvp = Path(path)/"per_attack_eer_eval.csv"
    if not csvp.exists(): continue
    d = {}
    with csvp.open() as f:
        r = csv.DictReader(f)
        for row in r:
            aid = row["attack_id"]
            eer = row["eer_percent"]
            attacks.add(aid)
            d[aid] = eer
    per_attack[name] = d

attacks = sorted(attacks)
cols = ["attack_id"] + list(per_attack.keys())
with (OUTDIR/"per_attack_eer_eval.md").open("w") as f:
    f.write("| " + " | ".join(["Attack"] + list(per_attack.keys())) + " |\n")
    f.write("|" + "|".join(["---"]*(1+len(per_attack))) + "|\n")
    for aid in attacks:
        vals = []
        for name in per_attack.keys():
            v = per_attack[name].get(aid)
            vals.append("-" if v is None else f"{float(v):.2f}")
        f.write("| "+aid+" | "+" | ".join(vals)+" |\n")
print(f"[save] {OUTDIR/'per_attack_eer_eval.md'}")

# --- Copy ROCs into figs/ (if you’ve rendered PNGs already) ---
for name, path in RUNS.items():
    if not path: continue
    rd = Path(path)
    for split in ["dev","eval"]:
        for ext in ["png","csv"]:
            src = rd/f"roc_{split}.{ext}"
            if src.exists():
                dst = FIGDIR/f"roc_{split}_{name.replace(' ','_').replace('+','_')}.{ext}"
                try:
                    shutil.copyfile(src, dst)
                    print(f"[copy] {src} -> {dst}")
                except Exception:
                    pass
