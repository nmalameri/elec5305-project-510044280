"""
aasist_infer.py

Inference script for pretrained AASIST-L (Logical Access) on ASVspoof 2019 LA.

Inputs:
    - pretrained_models/aasist/AASIST.py
    - pretrained_models/aasist/AASIST-L.pth
    - pretrained_models/aasist/AASIST-L.conf
    - results/manifests/dev.csv
    - results/manifests/eval.csv

Outputs (under results/models/aasist/<run_tag>/):
    - scores_dev.csv
    - scores_eval.csv
    - metrics.json
    - run_config.json

Supports FAST MODE via:
    --max_utt 200
"""

import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Import AASIST model
from AASIST import Model  # YOUR architecture file


# ------------------------------------------------------------
# Load AASIST-L.conf
# ------------------------------------------------------------
def load_conf(path):
    with open(path, "r") as f:
        conf = json.load(f)
    return conf["model_config"]


# ------------------------------------------------------------
# Audio loading (safe for mac/windows/linux)
# ------------------------------------------------------------
def load_waveform(path, target_sr=16000):
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


# ------------------------------------------------------------
# Preprocess audio for AASIST
# - pad/crop to nb_samp
# - mean-variance normalize
# ------------------------------------------------------------
def preprocess_aasist(wav, nb_samp):
    wav = wav.squeeze(0)  # [T]
    T = wav.shape[0]

    if T > nb_samp:
        wav = wav[:nb_samp]
    else:
        pad = nb_samp - T
        wav = torch.nn.functional.pad(wav, (0, pad))

    # normalize
    mean = wav.mean()
    std = wav.std()
    wav = (wav - mean) / (std + 1e-5)

    return wav.unsqueeze(0)  # [1, nb_samp]


# ------------------------------------------------------------
# EER utilities
# ------------------------------------------------------------
def compute_roc(scores, labels):
    idx = np.argsort(-scores)
    s = scores[idx]
    l = labels[idx]

    P = (l == 1).sum()
    N = (l == 0).sum()

    tprs, fprs, thrs = [], [], []
    tp = fp = 0
    prev = None

    for i in range(len(s)):
        if prev is None or s[i] != prev:
            tprs.append(tp / P if P > 0 else 0)
            fprs.append(fp / N if N > 0 else 0)
            thrs.append(s[i])
            prev = s[i]
        if l[i] == 1:
            tp += 1
        else:
            fp += 1

    return np.array(fprs), np.array(tprs), np.array(thrs)


def compute_eer(scores, labels):
    fprs, tprs, thrs = compute_roc(scores, labels)
    fnrs = 1 - tprs
    idx = np.argmin(np.abs(fnrs - fprs))
    return float((fnrs[idx] + fprs[idx]) / 2), float(thrs[idx])


def compute_acc(scores, labels, thr):
    preds = (scores > thr).astype(int)
    return float((preds == labels).sum() / len(labels))


# ------------------------------------------------------------
# Run inference for one split (dev or eval)
# ------------------------------------------------------------
def run_split(model, device, manifest_csv, out_csv, max_utt, nb_samp):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    with open(manifest_csv, "r") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    print(f"Found {total} utterances in {manifest_csv}")

    entries = []
    count = 0

    with open(out_csv, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["utt_id", "score", "label_int"])

        for row in rows:
            if max_utt is not None and count >= max_utt:
                print(f"Reached max_utt={max_utt}, stopping early.")
                break

            if count % 100 == 0:
                print(f"  Processed {count}/{total}")

            utt_id = row["utt_id"]
            label = 1 if row["label"].lower() == "spoof" else 0
            path = row["path"]

            # load + preprocess
            wav = load_waveform(path)
            wav = preprocess_aasist(wav, nb_samp)
            wav = wav.to(device)

            # forward: AASIST returns (embedding, logits)
            with torch.no_grad():
                _, out = model(wav)       # out shape: [1, 2]
                score = float(out[0, 1].item())  # spoof logit

            writer.writerow([utt_id, f"{score:.9f}", label])
            entries.append((score, label))
            count += 1

    return entries


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests_root", type=str, default="results/manifests")
    parser.add_argument("--dev_csv", type=str, default="dev.csv")
    parser.add_argument("--eval_csv", type=str, default="eval.csv")
    parser.add_argument("--conf", type=str, default="pretrained_models/aasist/AASIST-L.conf")
    parser.add_argument("--checkpoint", type=str, default="pretrained_models/aasist/AASIST-L.pth")
    parser.add_argument("--out_dir", type=str, default="results/models/aasist")
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--max_utt", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load conf
    cfg = load_conf(args.conf)
    nb_samp = cfg["nb_samp"]

    # prepare run dir
    if args.run_tag is None:
        run_tag = f"aasist_LA_pretrained_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    else:
        run_tag = args.run_tag

    run_dir = os.path.join(args.out_dir, run_tag)
    os.makedirs(run_dir, exist_ok=True)

    # save run_config.json
    run_config = {
        "model_type": "aasist-LA",
        "checkpoint": os.path.abspath(args.checkpoint),
        "conf": os.path.abspath(args.conf),
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

    # load model
    print("Loading AASIST-L model...")
    model = Model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    try:
        model.load_state_dict(ckpt)
    except:
        # some checkpoints nest params under "model" or "state_dict"
        for key in ["model", "state_dict", "net"]:
            if key in ckpt:
                model.load_state_dict(ckpt[key])
                break
    model.eval()
    print("Model loaded.")

    # paths
    dev_manifest = os.path.join(args.manifests_root, args.dev_csv)
    eval_manifest = os.path.join(args.manifests_root, args.eval_csv)

    # DEV inference
    print("\nRunning DEV inference...")
    dev_entries = run_split(
        model, device, dev_manifest,
        os.path.join(run_dir, "scores_dev.csv"),
        args.max_utt, nb_samp
    )
    dev_scores = np.array([s for s, l in dev_entries])
    dev_labels = np.array([l for s, l in dev_entries])

    dev_eer, dev_thr = compute_eer(dev_scores, dev_labels)
    dev_acc = compute_acc(dev_scores, dev_labels, dev_thr)

    print(f"DEV EER: {dev_eer*100:.2f}%, ACC: {dev_acc*100:.2f}%, thr={dev_thr}")

    # EVAL inference
    print("\nRunning EVAL inference...")
    eval_entries = run_split(
        model, device, eval_manifest,
        os.path.join(run_dir, "scores_eval.csv"),
        args.max_utt, nb_samp
    )
    eval_scores = np.array([s for s, l in eval_entries])
    eval_labels = np.array([l for s, l in eval_entries])

    eval_eer, _ = compute_eer(eval_scores, eval_labels)
    eval_acc = compute_acc(eval_scores, eval_labels, dev_thr)

    print(f"EVAL EER: {eval_eer*100:.2f}%, ACC: {eval_acc*100:.2f}%")

    # metrics.json
    metrics = {
        "model_type": "aasist-LA",
        "checkpoint": os.path.abspath(args.checkpoint),
        "dev": {"eer": dev_eer, "thr": dev_thr, "acc": dev_acc},
        "eval": {"eer": eval_eer, "acc": eval_acc},
        "run_tag": run_tag,
        "model_dir": os.path.abspath(run_dir),
        "finished_at": datetime.now().isoformat(timespec='seconds')
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDONE. Outputs saved under:\n  {run_dir}")


if __name__ == "__main__":
    main()