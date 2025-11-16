"""
05_gmm.py

Purpose:
    To train and evaluate two diagonal-covariance GMMs (bona fide vs spoof)
    using LFCC or MFCC features.

Functionality:
    - Loading features from Step 04 (04_extract_features.py)
    - Balanced attack sampling and frame caps
    - Training GMMs with EM algorithm
    - Scoring utterances using mean log-likelihood ratio
    - Computing EER on DEV and EVAL
    - Saving metrics, scores, ROC samples, and per-attack tables

Inputs:
    results/features/<feat_type>/*
    Command-line args: feat type, output directory

Outputs:
    results/models/gmm_<feat>/<run_tag>/
        run_config.json
        metrics.json
        scores_{dev,eval}.csv
        roc_{dev,eval}.csv
        per_attack tables

Notes:
    This script produces a fully reproducible GMM baseline.
"""


from pathlib import Path
import csv, json, re, shutil, time, warnings, argparse
import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import joblib
from datetime import datetime
from collections import Counter

# =======================
# Reproducible defaults
# =======================
SEED = 42
np.random.seed(SEED)

DEFAULT_NCOMP = 64          # 128 if you want a fuller baseline; 64 is more M1-friendly which I was using
DEFAULT_CAP_FRAMES = 100    # frames per utterance (train pooling)
MAX_SPOOF_UTTS = 6000       # subsample spoof utterances to bound RAM (6000 is chosen based on experiments and computational capacity)
MAX_SPOOF_FRAMES = 800_000  # global frame cap (spoof) before fitting (computational capacity related as well)

EM_MAX_ITER = 120
EM_TOL = 3e-3
EM_N_INIT = 1
EM_COV_TYPE = "diag"
EM_RETRY_REGS = (1e-4, 1e-3, 1e-2)  # escalate if ill-conditioned
# =======================

LA_UTT_RE = re.compile(r"^(LA_[A-Z]_\d{7})$", re.IGNORECASE)

# ---------- IO / paths ----------
def read_manifest(split):
    m = Path("results/manifests")/f"{split}.csv"
    rows=[]
    with m.open() as f:
        r = csv.DictReader(f)
        for d in r: rows.append(d)
    return rows

def free_bytes(p: Path):
    try: return shutil.disk_usage(p).free
    except: return 0

def find_feature_root():
    # prefers the largest existing location automatically
    cands=[Path("results/features"), Path.home()/ "asv_features"]
    vol=Path("/Volumes")
    if vol.exists():
        for d in vol.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                cands.append(d/"asv_features")
    existing=[c for c in cands if c.exists()]
    return max(existing, key=lambda p: free_bytes(p)) if existing else Path("results/features")

def feature_paths(root: Path, feat: str, split: str):
    d = root/feat/split
    out={}
    if not d.exists(): return out
    for p in d.glob("LA_*_???????.npz"): out[p.stem]=p
    for p in d.glob("LA_*_???????.npy"): out.setdefault(p.stem, p)
    return out

def load_feat(p: Path):
    if p.suffix==".npz":
        Z=np.load(p); return Z["feat"]
    return np.load(p, mmap_mode="r")

# ---------- labels / pooling ----------
def lbl_int(lbl:str): return 0 if lbl.strip().lower()=="bonafide" else 1  # 0=bona,1=spoof

def pool_frames(rows, fpaths, scaler, cap=DEFAULT_CAP_FRAMES, rng=np.random.default_rng(SEED)):
    """Collecting up to 'cap' frames per utterance; return (X, y_framewise)."""
    feats=[]; labs=[]
    for r in tqdm(rows, desc="pool", unit="utt"):
        u=r["utt_id"]
        if u not in fpaths: continue
        X=load_feat(fpaths[u])
        if X.ndim!=2 or X.shape[0]==0: continue
        if X.shape[0]>cap:
            idx=rng.choice(X.shape[0], size=cap, replace=False)
            X=X[idx]
        if scaler is not None: X=scaler.transform(X)
        feats.append(X); labs.append(np.full(X.shape[0], lbl_int(r["label"]), dtype=np.int64))
    if not feats: return np.empty((0,0),np.float32), np.empty((0,),np.int64)
    return np.vstack(feats).astype(np.float32), np.concatenate(labs)

def score_utts(rows, fpaths, scaler, g_b, g_s, feat_tag=""):
    utts=[]; scores=[]; ytrue=[]
    for r in tqdm(rows, desc=f"score{(':'+feat_tag) if feat_tag else ''}", unit="utt"):
        u=r["utt_id"]
        if u not in fpaths: continue
        X=load_feat(fpaths[u])
        if X.ndim!=2 or X.shape[0]==0: continue
        if scaler is not None: X=scaler.transform(X)
        s=float(np.mean(g_s.score_samples(X)-g_b.score_samples(X)))
        utts.append(u); scores.append(s); ytrue.append(lbl_int(r["label"]))
    return utts, np.array(scores, np.float32), np.array(ytrue, np.int64)

def eer(scores, y):
    fpr,tpr,thr=roc_curve(y, scores, pos_label=1)
    fnr=1-tpr
    i=int(np.argmin(np.abs(fpr-fnr)))
    return float((fpr[i]+fnr[i])/2), float(thr[i]), (fpr,tpr,thr)

def write_scores(dirp: Path, split, utts, scores, y):
    dirp.mkdir(parents=True, exist_ok=True)
    with (dirp/f"scores_{split}.csv").open("w", newline="") as f:
        w=csv.writer(f); w.writerow(["utt_id","score","label_int"])
        for u,s,l in zip(utts, scores, y): w.writerow([u, float(s), int(l)])

def write_roc(dirp: Path, split, fpr,tpr,thr):
    dirp.mkdir(parents=True, exist_ok=True)
    with (dirp/f"roc_{split}.csv").open("w", newline="") as f:
        w=csv.writer(f); w.writerow(["fpr","tpr","thr"])
        for a,b,c in zip(fpr,tpr,thr): w.writerow([float(a),float(b),float(c)])

# ---------- sanity / robust EM ----------
def sanitize_for_gmm(X: np.ndarray, name: str):
    """Keeping finite rows and cast to float64 for stable GMM fitting (CPU)."""
    if X.size == 0: return X.astype(np.float64, copy=False)
    mask = np.isfinite(X).all(axis=1)
    removed = int((~mask).sum())
    if removed: print(f"[sanitize] {name}: removed {removed} non-finite rows")
    X = X[mask]
    return X.astype(np.float64, copy=False)

def warn_zero_variance(X: np.ndarray, name: str):
    if X.size == 0: return
    col_std = X.std(axis=0)
    zeros = int((col_std == 0).sum())
    if zeros: print(f"[warn] {name}: {zeros} dims have zero variance; reg_covar will handle them")

def fit_gmm_safely(X, seed, feat_tag, label_tag, n_comp):
    """Fitting a GMM (CPU) with labeled prints + retries on reg_covar/K."""
    for reg_try in EM_RETRY_REGS:
        for k_try in (n_comp, max(2, n_comp // 2)):
            try:
                print(f"[{feat_tag}][{label_tag}] EM start  K={k_try} reg={reg_try} "
                      f"(max_iter={EM_MAX_ITER}, tol={EM_TOL}, n_init={EM_N_INIT})")
                t0 = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    g = GaussianMixture(
                        n_components=k_try, covariance_type=EM_COV_TYPE,
                        max_iter=EM_MAX_ITER, tol=EM_TOL,
                        reg_covar=reg_try,
                        n_init=EM_N_INIT, init_params="kmeans",
                        random_state=seed,
                        verbose=0
                    ).fit(X)
                dur = time.time()-t0
                print(f"[{feat_tag}][{label_tag}] EM done   K={k_try} reg={reg_try} "
                      f"iter={getattr(g,'n_iter_', 'NA')} converged={getattr(g,'converged_', 'NA')} "
                      f"({dur:.1f}s)")
                return g
            except ValueError as e:
                print(f"[{feat_tag}][{label_tag}] retry… K={k_try} reg={reg_try} ({e})")
                continue
    # final fallback
    print(f"[{feat_tag}][{label_tag}] final fallback  K=8 reg=1e-2")
    g = GaussianMixture(
        n_components=8, covariance_type=EM_COV_TYPE,
        max_iter=EM_MAX_ITER, tol=EM_TOL,
        reg_covar=1e-2,
        n_init=EM_N_INIT, init_params="kmeans",
        random_state=seed,
        verbose=0
    ).fit(X)
    print(f"[{feat_tag}][{label_tag}] EM done   K=8 reg=1e-2 "
          f"iter={getattr(g,'n_iter_','NA')} converged={getattr(g,'converged_','NA')}")
    return g

# ---------- run directory mgmt ----------
def make_run_dir(base_dir: Path, feat: str, n_comp: int, cap: int, tag: str|None):
    """Creating a unique run dir under results/models/gmm_<feat>/<run_tag> and update 'latest' symlink."""
    base_dir.mkdir(parents=True, exist_ok=True)
    if not tag or not tag.strip():
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        tag = f"K{n_comp}_cap{cap}_S{MAX_SPOOF_UTTS}_F{MAX_SPOOF_FRAMES}_{ts}"
    run_dir = base_dir / tag
    i = 1
    while run_dir.exists():
        run_dir = base_dir / f"{tag}_r{i}"
        i += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    # 'latest' symlink (best-effort)
    latest = base_dir / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name)  # relative symlink
    except Exception:
        pass
    return run_dir, tag

# ---------- main per-feature ----------
def run_one(feat, root: Path, n_comp, cap, tag=None):
    rng = np.random.default_rng(SEED)
    feat_tag = feat.upper()
    base_model_dir = Path("results/models") / f"gmm_{feat}"
    run_dir, run_tag = make_run_dir(base_model_dir, feat, n_comp, cap, tag)
    t_all = time.time()

    print(f"\n=== [{feat_tag}] baseline GMM ===")
    print(f"feature_root: {root}")
    print(f"model_dir   : {run_dir}  (tag: {run_tag})")
    print(f"settings    : n_components={n_comp}, cap_frames_per_utt={cap}, cov={EM_COV_TYPE}")

    # load manifests
    train = read_manifest("train"); dev = read_manifest("dev"); eval_ = read_manifest("eval")
    n_bona = sum(1 for r in train if r["label"].lower()=="bonafide")
    n_spoof= sum(1 for r in train if r["label"].lower()=="spoof")
    print(f"[data] train: bona={n_bona}  spoof={n_spoof};  dev={len(dev)}  eval={len(eval_)}")

    # map utt -> feature file
    f_tr  = feature_paths(root, feat, "train")
    f_dev = feature_paths(root, feat, "dev")
    f_ev  = feature_paths(root, feat, "eval")
    if not f_tr:
        print(f"[WARN] no features found at {root/feat/'train'}; skipping {feat}")
        return

    # --- scaler (fit on small, balanced sample) ---
    print("[scaler] fitting StandardScaler on small sample …")
    train_b = [r for r in train if r["label"].lower()=="bonafide"]
    train_s = [r for r in train if r["label"].lower()=="spoof"]

    # stratified spoof subsample by attack_id to avoid bias
    if len(train_s) > MAX_SPOOF_UTTS:
        by_attack = {}
        for r in train_s:
            aid = (r.get("attack_id") or "").strip().upper()
            by_attack.setdefault(aid, []).append(r)
        K = max(1, MAX_SPOOF_UTTS // max(1, len(by_attack)))
        picked = []
        for aid, rows in sorted(by_attack.items()):
            n = min(K, len(rows))
            idx = rng.choice(len(rows), size=n, replace=False)
            picked.extend(rows[i] for i in idx)
        if len(picked) < MAX_SPOOF_UTTS:
            remainder = [r for rows in by_attack.values() for r in rows if r not in picked]
            need = min(MAX_SPOOF_UTTS - len(picked), len(remainder))
            if need > 0:
                idx = rng.choice(len(remainder), size=need, replace=False)
                picked.extend(remainder[i] for i in idx)
        train_s = picked
        cnt = Counter((r.get("attack_id") or "").strip().upper() for r in train_s)
        msg = ", ".join(f"{k or 'UNK'}:{v}" for k, v in sorted(cnt.items()))
        print(f"[pool] subsampled spoof utters to {len(train_s)} (per attack → {msg})")

    Xb_tmp,_ = pool_frames(train_b, f_tr, None, cap=min(50, cap), rng=rng)
    Xs_tmp,_ = pool_frames(train_s, f_tr, None, cap=min(50, cap), rng=rng)
    Xtmp = np.vstack([Xb_tmp, Xs_tmp]) if Xb_tmp.size and Xs_tmp.size else np.empty((0,0), np.float32)
    if Xtmp.size == 0:
        print(f"[ERROR] could not collect frames to fit scaler for {feat}")
        return
    scaler = StandardScaler().fit(Xtmp)
    print(f"[scaler] done  sample_frames={Xtmp.shape[0]} dims={Xtmp.shape[1]}")

    # --- pooled frames for GMMs ---
    print(f"[pool] collecting training frames (cap {cap}/utt) …")
    Xb,_ = pool_frames(train_b, f_tr, scaler, cap=cap, rng=rng)
    Xs,_ = pool_frames(train_s, f_tr, scaler, cap=cap, rng=rng)
    if Xb.size==0 or Xs.size==0:
        print(f"[ERROR] pooled frames missing for one class; skipping {feat}")
        return
    print(f"[pool] bonafide frames={Xb.shape[0]} dims={Xb.shape[1]}")
    print(f"[pool] spoof    frames={Xs.shape[0]} dims={Xs.shape[1]}")
    if Xs.shape[0] > MAX_SPOOF_FRAMES:
        i = rng.choice(Xs.shape[0], size=MAX_SPOOF_FRAMES, replace=False)
        Xs = Xs[i]
        print(f"[pool] trimmed spoof frames to {Xs.shape[0]} (global cap)")

    warn_zero_variance(Xb, "bonafide")
    warn_zero_variance(Xs, "spoof")

    # --- train GMMs (EM) ---
    print(f"[em] training GMMs (diag) with K={n_comp} …")
    Xb_fit = sanitize_for_gmm(Xb, "bonafide")
    Xs_fit = sanitize_for_gmm(Xs, "spoof")
    g_b = fit_gmm_safely(Xb_fit, seed=0, feat_tag=feat_tag, label_tag="bonafide", n_comp=n_comp)
    g_s = fit_gmm_safely(Xs_fit, seed=1, feat_tag=feat_tag, label_tag="spoof",    n_comp=n_comp)

    # --- DEV scoring / threshold ---
    print(f"[dev:{feat_tag}] scoring & finding min-EER threshold …")
    d_utts, d_sc, d_y = score_utts(dev, f_dev, scaler, g_b, g_s, feat_tag=feat_tag)
    if d_sc.size == 0:
        print(f"[ERROR] no DEV scores for {feat}")
        return
    d_eer, d_thr, (fpr_d, tpr_d, thr_d) = eer(d_sc, d_y)
    d_acc = float(accuracy_score(d_y, (d_sc >= d_thr).astype(int)))
    print(f"[dev:{feat_tag}] EER={d_eer*100:.2f}%  thr={d_thr:.3f}  ACC={d_acc*100:.2f}%")

    # --- EVAL scoring ---
    print(f"[eval:{feat_tag}] scoring with DEV threshold …")
    e_utts, e_sc, e_y = score_utts(eval_, f_ev, scaler, g_b, g_s, feat_tag=feat_tag)
    if e_sc.size:
        e_eer, _, (fpr_e, tpr_e, thr_e) = eer(e_sc, e_y)
        e_acc = float(accuracy_score(e_y, (e_sc >= d_thr).astype(int)))
        print(f"[eval:{feat_tag}] EER={e_eer*100:.2f}%  ACC={e_acc*100:.2f}%")
    else:
        e_eer = e_acc = None
        fpr_e = tpr_e = thr_e = []

    # --- save artifacts (robust) ---
    print("[save] writing models, scores, and metrics …")
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir/"standardizer.json").write_text(json.dumps({
        "with_mean": True, "with_std": True,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }, indent=2))
    joblib.dump(g_b, run_dir/"gmm_bonafide.joblib")
    joblib.dump(g_s, run_dir/"gmm_spoof.joblib")

    write_scores(run_dir, "dev",  d_utts, d_sc, d_y)
    if len(e_utts): write_scores(run_dir, "eval", e_utts, e_sc, e_y)
    write_roc(run_dir, "dev",  fpr_d, tpr_d, thr_d)
    if len(e_utts): write_roc(run_dir, "eval", fpr_e, tpr_e, thr_e)

    # persist run settings & metrics for full reproducibility
    (run_dir/"run_config.json").write_text(json.dumps({
        "seed": SEED,
        "feature": feat,
        "run_tag": run_tag,
        "n_components": int(n_comp),
        "cap_frames_per_utt": int(cap),
        "max_spoof_utts": int(MAX_SPOOF_UTTS),
        "max_spoof_frames": int(MAX_SPOOF_FRAMES),
        "em": {
            "max_iter": EM_MAX_ITER, "tol": EM_TOL, "n_init": EM_N_INIT,
            "cov_type": EM_COV_TYPE, "retry_regs": list(EM_RETRY_REGS)
        }
    }, indent=2))

    (run_dir/"metrics.json").write_text(json.dumps({
        "feature": feat,
        "train_frames": int(len(Xb) + len(Xs)),
        "gmm_components_per_class": int(n_comp),
        "cap_frames_per_utt": int(cap),
        "dev": {"eer": d_eer, "thr": d_thr, "acc": d_acc},
        "eval": {"eer": e_eer, "acc": e_acc},
        "feature_root": str(root.resolve()),
        "model_dir": str(run_dir.resolve()),
        "run_tag": run_tag,
        "finished_at": datetime.now().isoformat(timespec="seconds")
    }, indent=2))

    print(f"[done:{feat_tag}] saved metrics to {run_dir/'metrics.json'}  (total {time.time()-t_all:.1f}s)")
    print(f"[hint] Run dir: {run_dir}\n       Latest symlink: {base_model_dir/'latest'}")

# ---------- CLI / entry ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Train baseline GMMs (versioned runs, reproducible).")
    ap.add_argument("--ncomp", type=int, default=DEFAULT_NCOMP, help="GMM components per class")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP_FRAMES, help="frames per utterance cap (train)")
    ap.add_argument("--tag", type=str, default="", help="optional run tag; if empty, auto-tag by settings+timestamp")
    return ap.parse_args()

def main():
    args = parse_args()
    root=find_feature_root()
    feats=[]
    if (root/"lfcc").exists(): feats.append("lfcc")
    if (root/"mfcc").exists(): feats.append("mfcc")
    if not feats:
        print(f"No features under {root}. Run 04_extract_features.py first."); return
    for f in feats:
        run_one(f, root, n_comp=args.ncomp, cap=args.cap, tag=(args.tag or None))
    print("Evaluation complete.")

if __name__ == "__main__":
    main()
