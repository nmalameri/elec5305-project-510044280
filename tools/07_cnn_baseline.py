#!/usr/bin/env python3
# trains a tiny 2D-CNN on LFCC/MFCC with clean logs, reproducible artifacts,
# and robust handling of corrupt/truncated feature files.
from __future__ import annotations
from pathlib import Path
import os, sys, time, json, csv, math, shutil, warnings, re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, accuracy_score
from tqdm import tqdm

# -----------------------------
# Defaults (press-and-run)
# -----------------------------
FEATURE = "mfcc"           # "lfcc" or "mfcc"
T_CROP = 400               # frames per crop (≈4s if 10ms hop)
EPOCHS = 15                # set to 2 for a quick smoke test
BATCH_SIZE = 32
LR = 2e-3
WEIGHT_DECAY = 1e-4
PRINT_BATCHES = 0          # e.g., 100 to see per-batch running loss, 0 = off

# Make CNN training comparable to your GMM run:
MAX_SPOOF_UTTS = 6000      # cap spoof utterances on train
BALANCE_ATTACKS = True     # distribute spoof cap evenly over attacks (A01..A06)
SEED = 42

# -----------------------------
# Device
# -----------------------------
def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# -----------------------------
# Paths / basic IO
# -----------------------------
LA_UTT_RE = re.compile(r"^(LA_[A-Z]_\d{7})$", re.IGNORECASE)

def read_manifest(split):
    p = Path("results/manifests")/f"{split}.csv"
    rows=[]
    with p.open() as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append(d)
    return rows

def find_feature_root():
    cands=[Path("results/features"), Path.home()/ "asv_features"]
    vol=Path("/Volumes")
    if vol.exists():
        for d in vol.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                cands.append(d/"asv_features")
    existing=[c for c in cands if c.exists()]
    return existing[0] if existing else Path("results/features")

def feature_paths(root: Path, feat: str, split: str):
    d = root/feat/ split
    out={}
    if not d.exists(): return out
    for p in d.glob("LA_*_???????.npz"): out[p.stem]=p
    for p in d.glob("LA_*_???????.npy"): out.setdefault(p.stem, p)
    return out

# --- safe feature open ---
def _probe_feat_header(path: Path) -> tuple[bool, tuple|None]:
    """
    Fast, safe probe:
      - returns (ok, (T,D)) for .npy where possible
      - returns (ok, None) for .npz (we'll do a tiny read)
    """
    try:
        if path.suffix == ".npy":
            import numpy.lib.format as fmt
            with open(path, "rb") as f:
                version = fmt.read_magic(f)
                fmt._check_version(version)
                shape, fortran, dtype = fmt.read_array_header_1_0(f) if version == (1,0) else fmt.read_array_header_2_0(f)
                if not isinstance(shape, tuple) or len(shape) != 2:
                    return False, None
                return True, shape
        else:
            with np.load(path) as Z:
                if "feat" not in Z: return False, None
                M = Z["feat"]
                if M.ndim != 2: return False, None
                return True, M.shape
    except Exception:
        return False, None

def load_feat(path: Path):
    if path.suffix == ".npz":
        Z = np.load(path)
        return Z["feat"]
    return np.load(path, mmap_mode="r")

# -----------------------------
# Utility: metrics
# -----------------------------
def eer(scores: np.ndarray, y: np.ndarray):
    fpr,tpr,thr = roc_curve(y, scores, pos_label=1)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i]), (fpr, tpr, thr)

def op_metrics(scores, y, thr):
    pred = (scores >= thr).astype(int)
    acc = float(accuracy_score(y, pred))
    tp = int(((pred==1)&(y==1)).sum())
    tn = int(((pred==0)&(y==0)).sum())
    fp = int(((pred==1)&(y==0)).sum())
    fn = int(((pred==0)&(y==1)).sum())
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    bal = 0.5 * (tpr + (tn / max(1, tn + fp)))
    return acc, tpr, fpr, bal

# -----------------------------
# Dataset
# -----------------------------
class FrameCropDataset(Dataset):
    """
    Returns (X, y, utt_id) where:
      X: [1, T, D] (time, feat-dim)
      y: 0=bonafide, 1=spoof
    Train uses random crop; Dev/Eval uses center crop. One crop per sample.
    Assumes ids were pre-filtered to only include good feature files.
    """
    def __init__(self, rows, fpaths, scaler: StandardScaler, T=400, train=False):
        self.fpaths = fpaths
        self.scaler = scaler
        self.T = int(T)
        self.train = train

        ids = []
        labels = []
        for r in rows:
            u = r["utt_id"]
            if u in fpaths:
                ids.append(u)
                labels.append(0 if r["label"].lower()=="bonafide" else 1)

        self.ids = ids
        self.labels = np.array(labels, np.int64)

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        u = self.ids[i]
        y = self.labels[i]
        p = self.fpaths[u]
        try:
            X = load_feat(p)  # [T_all, D]
        except Exception:
            # fallback in case a file becomes corrupt between scan and load
            X = np.zeros((self.T, 60), np.float32)

        if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0:
            X = np.zeros((self.T, 60), np.float32)

        X = self.scaler.transform(X)

        T_all, D = X.shape
        T = min(self.T, T_all)
        if self.train:
            s = np.random.randint(0, max(1, T_all - T + 1)) if T_all > T else 0
        else:
            s = max(0, (T_all - T) // 2)
        crop = X[s:s+T]
        if crop.shape[0] < self.T:
            pad = np.zeros((self.T - crop.shape[0], D), dtype=crop.dtype)
            crop = np.vstack([crop, pad])

        crop = crop[np.newaxis, :, :]  # [1, T, D]
        return torch.from_numpy(crop).float(), torch.tensor(y, dtype=torch.long), u

# -----------------------------
# Model: tiny 2D CNN
# -----------------------------
class TinyCNN(nn.Module):
    """
    Input: [B, 1, T, D]
    Two conv blocks + global average pool (both axes) + linear head.
    """
    def __init__(self, in_ch=1, hid=32, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=(5,5), padding=2)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, hid, kernel_size=(3,3), padding=1)
        self.bn3   = nn.BatchNorm2d(hid)
        self.head  = nn.Linear(hid, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=(2,2))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(2,2))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=(-1,-2))  # global average pool [B, hid]
        logits = self.head(x).squeeze(-1)  # [B]
        return logits

# -----------------------------
# Helpers
# -----------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fit_scaler_light(train_b, train_s, f_tr, T=50, rng=np.random.default_rng(42)):
    feats=[]
    def take(rows):
        for r in rows:
            u = r["utt_id"]
            if u not in f_tr: continue
            try:
                X = load_feat(f_tr[u])
            except Exception:
                continue
            if isinstance(X, np.ndarray) and X.ndim==2 and X.shape[0]>0:
                idx = rng.choice(X.shape[0], size=min(T, X.shape[0]), replace=False)
                feats.append(X[idx])
    take(train_b); take(train_s)
    if not feats:
        raise RuntimeError("Could not collect frames to fit StandardScaler.")
    X = np.vstack(feats)
    scaler = StandardScaler().fit(X)
    return scaler

def evaluate(model, loader, device):
    model.eval()
    all_scores=[]; all_y=[]; all_u=[]
    with torch.no_grad():
        for xb, yb, u in loader:
            xb = xb.to(device)
            logits = model(xb)  # [B]
            all_scores.append(logits.detach().cpu().float().numpy())
            all_y.append(yb.numpy())
            all_u.extend(list(u))
    scores = np.concatenate(all_scores) if all_scores else np.empty((0,), np.float32)
    y      = np.concatenate(all_y)      if all_y      else np.empty((0,), np.int64)
    return all_u, scores, y

def make_run_dir(base_dir: Path, tag: str):
    rd = base_dir / f"cnn_{tag}"
    rd.mkdir(parents=True, exist_ok=True)
    latest = base_dir / "latest"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(rd.name)
    except Exception:
        pass
    return rd

# -----------------------------
# Robust pre-scan for corrupt features
# -----------------------------
def filter_good_utts(rows, fpaths):
    good=[]; bad=[]
    for r in rows:
        u = r["utt_id"]
        p = fpaths.get(u)
        if p is None:
            bad.append((u, "missing_path"))
            continue
        ok, shape = _probe_feat_header(p)
        if not ok:
            bad.append((u, "corrupt_or_bad_shape"))
            continue
        try:
            if p.stat().st_size == 0:
                bad.append((u, "zero_size"))
                continue
        except Exception:
            bad.append((u, "stat_error"))
            continue
        good.append(r)
    return good, bad

def write_skipped(run_dir: Path, name: str, bad_list):
    if not bad_list: return
    with (run_dir/f"skipped_{name}.txt").open("w") as f:
        for u, why in bad_list:
            f.write(f"{u}\t{why}\n")

# -----------------------------
# Per-attack helpers (FIXED)
# -----------------------------
def per_attack_eer_eval(eval_utts, eval_scores, eval_y, utt2attack):
    # build indices
    all_idx_bona = [i for i,y in enumerate(eval_y) if y==0]
    attack_to_spoof_idx = defaultdict(list)
    for i,u in enumerate(eval_utts):
        if eval_y[i]==1:
            attack_to_spoof_idx[utt2attack.get(u, "UNK")].append(i)

    rows=[]
    for aid in sorted(attack_to_spoof_idx.keys()):
        idx = attack_to_spoof_idx[aid] + all_idx_bona
        y = np.asarray(eval_y[idx])
        s = np.asarray(eval_scores[idx], dtype=np.float32)
        n_b = int((y==0).sum()); n_s = int((y==1).sum())
        if n_b > 0 and n_s > 0 and len(y) > 1:
            ee, _, _ = eer(s, y)     # <-- FIX: unpack 3-tuple, take first
            rows.append([aid, len(idx), n_b, n_s, round(100*ee,2)])
    return rows

def per_attack_op_eval(eval_utts, eval_scores, eval_y, utt2attack, thr):
    all_idx_bona = [i for i,y in enumerate(eval_y) if y==0]
    attack_to_spoof_idx = defaultdict(list)
    for i,u in enumerate(eval_utts):
        if eval_y[i]==1:
            attack_to_spoof_idx[utt2attack.get(u,"UNK")].append(i)

    rows=[]
    for aid in sorted(attack_to_spoof_idx.keys()):
        idx = attack_to_spoof_idx[aid] + all_idx_bona
        y = np.asarray(eval_y[idx]); s = np.asarray(eval_scores[idx], dtype=np.float32)
        if (y==0).sum() > 0 and (y==1).sum() > 0:
            acc,tpr,fpr,bal = op_metrics(s, y, thr)
            rows.append([aid, len(y), int((y==0).sum()), int((y==1).sum()),
                         round(acc,4), round(tpr,4), round(fpr,4)])
    return rows

# -----------------------------
# Main
# -----------------------------
def main():
    np.seterr(all="ignore")
    set_seed(SEED)
    device = pick_device()
    print(f"[device] {device.type}")

    # data
    train_rows = read_manifest("train")
    dev_rows   = read_manifest("dev")
    eval_rows  = read_manifest("eval")
    print(f"[data] train: bona={sum(r['label'].lower()=='bonafide' for r in train_rows)} "
          f"spoof={sum(r['label'].lower()=='spoof' for r in train_rows)}; "
          f"dev={len(dev_rows)} eval={len(eval_rows)}")

    # features
    feat_root = find_feature_root()
    f_tr  = feature_paths(feat_root, FEATURE, "train")
    f_dev = feature_paths(feat_root, FEATURE, "dev")
    f_ev  = feature_paths(feat_root, FEATURE, "eval")
    if not f_tr:
        print(f"[ERROR] no features at {feat_root/FEATURE/'train'}")
        sys.exit(1)

    # Subsample spoof utterances for train
    train_b = [r for r in train_rows if r["label"].lower()=="bonafide"]
    train_s_full = [r for r in train_rows if r["label"].lower()=="spoof"]
    if MAX_SPOOF_UTTS and len(train_s_full) > MAX_SPOOF_UTTS:
        if BALANCE_ATTACKS:
            by_attack = defaultdict(list)
            for r in train_s_full:
                a = r.get("attack_id") or "UNK"
                by_attack[a].append(r)
            target = MAX_SPOOF_UTTS
            keys = sorted(k for k in by_attack.keys() if k.startswith("A"))
            if not keys: keys = list(by_attack.keys())
            per = max(1, target // max(1,len(keys)))
            picked=[]
            for k in keys:
                rows = by_attack[k]
                picked.extend(rows[:per])
            if len(picked) < target:
                pool=[]
                for k in keys:
                    pool.extend(by_attack[k][per:])
                picked.extend(pool[:max(0,target-len(picked))])
            train_s = picked[:target]
            msg = ", ".join([f"{k}:{min(per,len(by_attack[k]))}" for k in keys[:6]])
            print(f"[subsample] spoof cap={target} (per attack → {msg})")
        else:
            train_s = train_s_full[:MAX_SPOOF_UTTS]
            print(f"[subsample] spoof cap={len(train_s)} (no attack balancing)")
    else:
        train_s = train_s_full

    # --- Pre-scan/filter corrupt features (train/dev/eval) ---
    base_dir = Path("results/models")/f"cnn_{FEATURE}"
    ts = time.strftime("%Y%m%d-%H%M")
    tag = f"T{T_CROP}_E{EPOCHS}_B{BATCH_SIZE}_S{MAX_SPOOF_UTTS}{'_BAL' if BALANCE_ATTACKS else ''}_{ts}"
    run_dir  = make_run_dir(base_dir, tag)
    print(f"[run_dir] {run_dir}")

    train_b_good, train_b_bad = filter_good_utts(train_b, f_tr)
    train_s_good, train_s_bad = filter_good_utts(train_s, f_tr)
    if train_b_bad or train_s_bad:
        print(f"[warn] skipping corrupt train feats: bona={len(train_b_bad)} spoof={len(train_s_bad)}")
        write_skipped(run_dir, "train", train_b_bad + train_s_bad)

    dev_good, dev_bad = filter_good_utts(dev_rows, f_dev)
    eval_good, eval_bad = filter_good_utts(eval_rows, f_ev)
    if dev_bad:
        print(f"[warn] skipping corrupt dev feats: {len(dev_bad)}")
        write_skipped(run_dir, "dev", dev_bad)
    if eval_bad:
        print(f"[warn] skipping corrupt eval feats: {len(eval_bad)}")
        write_skipped(run_dir, "eval", eval_bad)

    # scaler (light)
    print("[scaler] fitting on a light train sample …")
    scaler = fit_scaler_light(train_b_good, train_s_good, f_tr, T=50, rng=np.random.default_rng(SEED))
    pos_weight = torch.tensor(len(train_b_good) / max(1,len(train_s_good)), dtype=torch.float32, device=device)
    print(f"[loss] BCEWithLogitsLoss pos_weight={pos_weight.item():.3f}  "
          f"(bona:{len(train_b_good)}, spoof:{len(train_s_good)})")

    # save config & scaler
    (run_dir/"run_config.json").write_text(json.dumps({
        "feature": FEATURE, "T_crop": T_CROP, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "lr": LR, "weight_decay": WEIGHT_DECAY,
        "max_spoof_utts": MAX_SPOOF_UTTS, "balance_attacks": BALANCE_ATTACKS,
        "seed": SEED, "device": device.type,
        "feature_root": str(feat_root.resolve())
    }, indent=2))
    (run_dir/"scaler.json").write_text(json.dumps({
        "with_mean": True, "with_std": True,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }, indent=2))

    # datasets/loaders
    ds_tr  = FrameCropDataset(train_b_good + train_s_good, f_tr, scaler, T=T_CROP, train=True)
    ds_dev = FrameCropDataset(dev_good, f_dev, scaler, T=T_CROP, train=False)
    ds_ev  = FrameCropDataset(eval_good, f_ev, scaler, T=T_CROP, train=False)

    g = torch.Generator(); g.manual_seed(SEED)
    dl_tr  = DataLoader(ds_tr,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, generator=g)
    dl_dev = DataLoader(ds_dev, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    dl_ev  = DataLoader(ds_ev,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # model/optim
    model = TinyCNN().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    # training loop
    best = {"eer": 1.0, "thr": 0.0, "epoch": -1}
    for ep in range(1, EPOCHS+1):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        bcount = 0
        for xb, yb, _ in dl_tr:
            xb = xb.to(device)          # [B, 1, T, D]
            yb = yb.to(device).float()  # [B]
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            run_loss += loss.item()
            bcount += 1
            if PRINT_BATCHES and (bcount % PRINT_BATCHES == 0):
                print(f"[batch] ep {ep}  step {bcount}  loss={run_loss/bcount:.4f}")
        sched.step()
        tr_loss = run_loss / max(1,bcount)
        print(f"[train] ep {ep:02d}/{EPOCHS}  loss={tr_loss:.4f}  lr={optim.param_groups[0]['lr']:.2e}  time={time.time()-t0:.1f}s")

        # dev eval each epoch
        dev_utts, dev_scores, dev_y = evaluate(model, dl_dev, device)
        d_eer, d_thr, (fpr_d, tpr_d, thr_d) = eer(dev_scores, dev_y) if len(dev_y) else (1.0, 0.0, ([],[],[]))
        d_acc = float(accuracy_score(dev_y, (dev_scores >= d_thr).astype(int))) if len(dev_y) else 0.0
        star = ""
        if d_eer < best["eer"]:
            best.update({"eer": d_eer, "thr": d_thr, "epoch": ep})
            star = "  ⭐ best"
            # checkpoint best
            torch.save(model.state_dict(), run_dir/"model.pt")
            torch.save(optim.state_dict(), run_dir/"optim.pt")
            # persist DEV artifacts
            with (run_dir/"scores_dev.csv").open("w", newline="") as f:
                w = csv.writer(f); w.writerow(["utt_id","score","label_int"])
                for u,s,l in zip(dev_utts, dev_scores, dev_y):
                    w.writerow([u, float(s), int(l)])
            with (run_dir/"roc_dev.csv").open("w", newline="") as f:
                w = csv.writer(f); w.writerow(["fpr","tpr","thr"])
                for a,b,c in zip(fpr_d, tpr_d, thr_d): w.writerow([float(a),float(b),float(c)])
        tpr_now = (dev_scores>=d_thr).astype(int)[dev_y==1].mean()*100 if (dev_y==1).any() else 0.0
        fpr_now = (dev_scores>=d_thr).astype(int)[dev_y==0].mean()*100 if (dev_y==0).any() else 0.0
        print(f"[dev]   ep {ep:02d}  EER={d_eer*100:.2f}%  thr={d_thr:.3f}  ACC={d_acc*100:.2f}%  "
              f"TPR={tpr_now:.2f}%  FPR={fpr_now:.2f}%{star}")

    # final: reload best model for eval
    state = torch.load(run_dir/"model.pt", map_location=device)
    model.load_state_dict(state)

    # EVAL scoring
    eval_utts, eval_scores, eval_y = evaluate(model, dl_ev, device)
    with (run_dir/"scores_eval.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["utt_id","score","label_int"])
        for u,s,l in zip(eval_utts, eval_scores, eval_y):
            w.writerow([u, float(s), int(l)])

    # DEV EER/THR (best) — reload stored dev scores for exactness
    dev_rows=[]
    with (run_dir/"scores_dev.csv").open() as f:
        r = csv.DictReader(f)
        for d in r: dev_rows.append(d)
    dev_scores = np.array([float(d["score"]) for d in dev_rows], np.float32) if dev_rows else np.empty((0,), np.float32)
    dev_y_int  = np.array([int(d["label_int"]) for d in dev_rows], np.int64)   if dev_rows else np.empty((0,), np.int64)
    d_eer, d_thr, (fpr_d, tpr_d, thr_d) = eer(dev_scores, dev_y_int) if len(dev_y_int) else (1.0, 0.0, ([],[],[]))
    with (run_dir/"roc_dev.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["fpr","tpr","thr"])
        for a,b,c in zip(fpr_d, tpr_d, thr_d): w.writerow([float(a),float(b),float(c)])

    # EVAL EER (threshold-free) + ROC
    e_eer, _, (fpr_e, tpr_e, thr_e) = eer(eval_scores, eval_y) if len(eval_y) else (1.0, 0.0, ([],[],[]))
    with (run_dir/"roc_eval.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["fpr","tpr","thr"])
        for a,b,c in zip(fpr_e, tpr_e, thr_e): w.writerow([float(a),float(b),float(c)])

    # Operating point @ DEV threshold (now includes EERs)
    ev_acc, ev_tpr, ev_fpr, ev_bacc = op_metrics(eval_scores, eval_y, d_thr) if len(eval_y) else (0,0,0,0)
    op_csv  = run_dir / "op_eval_at_dev_threshold.csv"
    op_json = run_dir / "op_eval_at_dev_threshold.json"
    with op_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["acc","tpr","fpr","bal_acc","thr_dev","eval_eer","dev_eer"])
        w.writerow([ev_acc, ev_tpr, ev_fpr, ev_bacc, float(d_thr), float(e_eer), float(d_eer)])
    op_json.write_text(json.dumps({
        "acc": ev_acc, "tpr": ev_tpr, "fpr": ev_fpr, "bal_acc": ev_bacc,
        "thr_dev": float(d_thr),
        "eval_eer": float(e_eer),
        "dev_eer": float(d_eer)
    }, indent=2))
    print(f"[save] {op_csv}")
    print(f"[save] {op_json}")
    print(f"[eval] EER={e_eer*100:.2f}%  ACC={ev_acc*100:.2f}%  TPR={ev_tpr*100:.2f}%  FPR={ev_fpr*100:.2f}%  (at DEV thr)")

    # Per-attack analysis
    mrows=[]
    with (Path("results/manifests")/"eval.csv").open() as f:
        r = csv.DictReader(f)
        for d in r: mrows.append(d)
    utt2attack = { d["utt_id"]: (d.get("attack_id") or "UNK") for d in mrows }

    pa_eer_rows = per_attack_eer_eval(eval_utts, eval_scores, eval_y, utt2attack)
    pa_op_rows  = per_attack_op_eval(eval_utts, eval_scores, eval_y, utt2attack, d_thr)

    pa_eer_csv = run_dir / "per_attack_eer_eval.csv"
    with pa_eer_csv.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["attack_id","n_total","n_bonafide","n_spoof","eer_percent"])
        for row in pa_eer_rows: w.writerow(row)
    print(f"[save] {pa_eer_csv}")

    pa_op_csv = run_dir / "per_attack_op_eval_at_dev_threshold.csv"
    with pa_op_csv.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["attack_id","n_total","n_bonafide","n_spoof","acc","tpr","fpr"])
        for row in pa_op_rows: w.writerow(row)
    print(f"[save] {pa_op_csv}")

    # metrics.json (summary)
    (run_dir/"metrics.json").write_text(json.dumps({
        "feature": FEATURE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "T_crop": T_CROP,
        "max_spoof_utts": MAX_SPOOF_UTTS,
        "balance_attacks": BALANCE_ATTACKS,
        "dev": {"eer": float(d_eer), "thr": float(d_thr)},
        "eval": {"eer": float(e_eer), "acc": float(ev_acc), "tpr": float(ev_tpr), "fpr": float(ev_fpr), "bal_acc": float(ev_bacc)},
        "run_dir": str(run_dir.resolve()),
        "feature_root": str(find_feature_root().resolve()),
        "device": device.type
    }, indent=2))

    print(f"[done] saved to {run_dir}")
    print(f"[dev]  EER={d_eer*100:.2f}%  thr={d_thr:.3f}")
    print(f"[eval] EER={e_eer*100:.2f}%  ACC={ev_acc*100:.2f}%  TPR={ev_tpr*100:.2f}%  FPR={ev_fpr*100:.2f}%  (at DEV thr)")

if __name__ == "__main__":
    main()