"""
rawnet2_infer.py

Inference for RawNet2 using:
    - model.py (RawNet class) from https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2
    - model_config_RawNet.yaml
    - pre_trained_DF_RawNet2.pth from https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip

Outputs:
    - scores_dev.csv
    - scores_eval.csv
    - metrics.json
    - run_config.json

FAST MODE:
    python pretrained_models/rawnet2/rawnet2_infer.py --max_utt 200

FULL RUN:
    python pretrained_models/rawnet2/rawnet2_infer.py
"""

import argparse
import csv
import json
import os
import numpy as np
import torch
import torchaudio
import soundfile as sf
from datetime import datetime
from typing import List, Tuple

# importing RawNet from model.py
from model import RawNet


# -------------------------
# Loading YAML config 
# -------------------------
def load_yaml_config(path):
    """
    Minimal YAML parser for config file.
    Only handles simple key:value and list entries.
    """
    import yaml
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["model"]


# ----------------
# Audio loading
# ----------------
def load_waveform(path, target_sr=16000):
    """
    Loading audio using soundfile (works with FLAC/WAV on all OS because I faced issues on running this on conda prior).
    Returns mono waveform [1, T].
    """
    wav_np, sr = sf.read(path)
    if wav_np.ndim == 1:
        wav_np = wav_np[None, :]
    else:
        wav_np = wav_np.T
        wav_np = wav_np.mean(axis=0, keepdims=True)

    wav = torch.tensor(wav_np, dtype=torch.float32)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav  # [1, T]


# ---------------------------------
# Preprocessing RawNet2 waveform
# ---------------------------------
def preprocess_rawnet2(wav, nb_samp):
    """
    RawNet2 requires:
        - [B, T] waveform
        - exactly nb_samp samples (64600)
        - mean-variance normalisation
    """
    wav = wav.squeeze(0)          # [T]

    # Crop or pad
    if wav.shape[0] > nb_samp:
        wav = wav[:nb_samp]
    else:
        pad_len = nb_samp - wav.shape[0]
        wav = torch.nn.functional.pad(wav, (0, pad_len))

    # Mean-variance norm
    mean = wav.mean()
    std = wav.std()
    wav = (wav - mean) / (std + 1e-5)

    return wav.unsqueeze(0)       # [1, T]


# --------------------
# EER / ROC helpers
# --------------------
def compute_roc(scores, labels):
    idx = np.argsort(-scores)
    s = scores[idx]
    l = labels[idx]

    P = (l == 1).sum()
    N = (l == 0).sum()

    tprs, fprs, thresholds = [], [], []
    tp = fp = 0
    prev = None

    for i in range(len(s)):
        if prev is None or s[i] != prev:
            tprs.append(tp / P if P > 0 else 0)
            fprs.append(fp / N if N > 0 else 0)
            thresholds.append(s[i])
            prev = s[i]
        if l[i] == 1:
            tp += 1
        else:
            fp += 1

    return np.array(fprs), np.array(tprs), np.array(thresholds)


def compute_eer(scores, labels):
    fprs, tprs, thr = compute_roc(scores, labels)
    fnrs = 1 - tprs
    idx = np.argmin(np.abs(fnrs - fprs))
    return float((fnrs[idx] + fprs[idx]) / 2), float(thr[idx])


def compute_acc(scores, labels, thr):
    preds = (scores > thr).astype(int)
    return float((preds == labels).sum() / len(labels))


# -----------------------
# Inference on one split
# -----------------------
def run_split(
    model,
    device,
    manifest_csv,
    out_scores_csv,
    max_utt,
    nb_samp
):
    """
    Processing DEV or EVAL split.
    Writes scores CSV:
        utt_id,score,label_int
    """
    os.makedirs(os.path.dirname(out_scores_csv), exist_ok=True)
    entries = []

    with open(manifest_csv, "r") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    print(f"Found {total} utterances in {manifest_csv}")

    count = 0
    with open(out_scores_csv, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["utt_id", "score", "label_int"])

        for row in rows:
            if max_utt is not None and count >= max_utt:
                print(f"Reached max_utt={max_utt}, stopping early.")
                break

            if count % 100 == 0:
                print(f"  Processed {count}/{total}")

            utt_id = row["utt_id"]
            label_str = row["label"].lower()
            label_int = 1 if label_str == "spoof" else 0
            path = row["path"]

            # load + preprocess
            wav = load_waveform(path)
            wav = preprocess_rawnet2(wav, nb_samp)
            wav = wav.to(device)

            # forward pass
            with torch.no_grad():
                out = model(wav)       # [1, 2] log-softmax
                # spoof = class index 1
                score = float(out[0, 1].item())

            writer.writerow([utt_id, f"{score:.9f}", label_int])
            entries.append((score, label_int))
            count += 1

    return entries


# -----
# Main
# -----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests_root", type=str, default="results/manifests")
    parser.add_argument("--dev_csv", type=str, default="dev.csv")
    parser.add_argument("--eval_csv", type=str, default="eval.csv")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pretrained_models/rawnet2/models/pre_trained_DF_RawNet2.pth",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pretrained_models/rawnet2/model_config_RawNet.yaml",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/models/rawnet2",
    )
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--max_utt", type=int, default=None)
    args = parser.parse_args()

    # device (CUDA or CPU), MPS does not work
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # loading config
    cfg = load_yaml_config(args.config)
    nb_samp = cfg["nb_samp"]

    # preparing run dir
    if args.run_tag is None:
        run_tag = f"rawnet2_pretrained_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    else:
        run_tag = args.run_tag

    run_dir = os.path.join(args.out_dir, run_tag)
    os.makedirs(run_dir, exist_ok=True)

    # saving run_config.json
    run_config = {
        "model_type": "rawnet2",
        "checkpoint": os.path.abspath(args.checkpoint),
        "config": os.path.abspath(args.config),
        "manifests_root": os.path.abspath(args.manifests_root),
        "dev_csv": args.dev_csv,
        "eval_csv": args.eval_csv,
        "device": str(device),
        "run_tag": run_tag,
        "run_dir": os.path.abspath(run_dir),
        "max_utt": args.max_utt,
    }
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # loading model
    print("Loading RawNet2 model...")
    model = RawNet(cfg, device=device).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    # trying simple load
    try:
        model.load_state_dict(ckpt)
    except:
        # trying common dict keys
        possible_keys = ["state_dict", "model_state", "model", "rawnet", "net"]
        loaded = False
        for k in possible_keys:
            if k in ckpt:
                model.load_state_dict(ckpt[k])
                loaded = True
                break
        if not loaded:
            raise RuntimeError(f"Cannot load checkpoint: keys={ckpt.keys()}")

    model.eval()
    print("Model loaded.")

    # paths
    dev_manifest = os.path.join(args.manifests_root, args.dev_csv)
    eval_manifest = os.path.join(args.manifests_root, args.eval_csv)

    # DEV inference
    print("\nRunning DEV inference...")
    dev_entries = run_split(
        model=model,
        device=device,
        manifest_csv=dev_manifest,
        out_scores_csv=os.path.join(run_dir, "scores_dev.csv"),
        max_utt=args.max_utt,
        nb_samp=nb_samp,
    )

    dev_scores = np.array([e[0] for e in dev_entries])
    dev_labels = np.array([e[1] for e in dev_entries])

    dev_eer, dev_thr = compute_eer(dev_scores, dev_labels)
    dev_acc = compute_acc(dev_scores, dev_labels, dev_thr)

    print(f"DEV EER: {dev_eer*100:.2f}%, threshold={dev_thr}, acc={dev_acc*100:.2f}%")

    # EVAL inference
    print("\nRunning EVAL inference...")
    eval_entries = run_split(
        model=model,
        device=device,
        manifest_csv=eval_manifest,
        out_scores_csv=os.path.join(run_dir, "scores_eval.csv"),
        max_utt=args.max_utt,
        nb_samp=nb_samp,
    )

    eval_scores = np.array([e[0] for e in eval_entries])
    eval_labels = np.array([e[1] for e in eval_entries])

    eval_eer, _ = compute_eer(eval_scores, eval_labels)
    eval_acc = compute_acc(eval_scores, eval_labels, dev_thr)

    print(f"EVAL EER: {eval_eer*100:.2f}%, acc@dev_thr={eval_acc*100:.2f}%")

    # writing metrics.json
    metrics = {
        "model_type": "rawnet2",
        "checkpoint": os.path.abspath(args.checkpoint),
        "dev": {
            "eer": dev_eer,
            "thr": dev_thr,
            "acc": dev_acc
        },
        "eval": {
            "eer": eval_eer,
            "acc": eval_acc
        },
        "run_tag": run_tag,
        "model_dir": os.path.abspath(run_dir),
        "finished_at": datetime.now().isoformat(timespec="seconds")
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDONE. Outputs saved in: {run_dir}")


if __name__ == "__main__":
    main()