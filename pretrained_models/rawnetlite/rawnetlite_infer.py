"""
rawnetlite_infer.py

Inference for RawNetLite using:
    - RawNetLite.py from https://github.com/adipiz99/RawNetLite
    - Pretrained weights rawnet_lite.pt

Outputs (saved under results/models/rawnetlite/<run_tag>/):
    - scores_dev.csv       # per-utterance scores for DEV
    - scores_eval.csv      # per-utterance scores for EVAL
    - metrics.json         # EER, threshold, and ACC for DEV/EVAL
    - run_config.json      # configuration used for this inference run

FAST MODE:
    python pretrained_models/rawnetlite/rawnetlite_infer.py --max_utt 200

FULL RUN:
    python pretrained_models/rawnetlite/rawnetlite_infer.py
"""


import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import torch
import torchaudio
import soundfile as sf # For CUDA implementation

from RawNetLite import RawNetLite 
import audio_preprocessor


# ----------------
# Data structures
# ----------------

@dataclass
class ScoreEntry:
    utt_id: str
    score: float
    label_int: int


# -------------------
# EER / ROC utilities
# -------------------

def compute_roc(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computing ROC curve and thresholds.

    Args:
        scores: shape [N], higher = more spoof-like
        labels: shape [N], 1 = spoof, 0 = bona fide

    Returns:
        fprs, tprs, thresholds
    """
    # sorting scores descending
    desc_idx = np.argsort(-scores)
    scores_sorted = scores[desc_idx]
    labels_sorted = labels[desc_idx]

    P = np.sum(labels_sorted == 1)
    N = np.sum(labels_sorted == 0)

    tprs = []
    fprs = []
    thresholds = []

    tp = 0
    fp = 0
    prev_score = None

    for i in range(len(scores_sorted)):
        s = scores_sorted[i]
        y = labels_sorted[i]

        if prev_score is None or s != prev_score:
            if P > 0 and N > 0:
                tprs.append(tp / P)
                fprs.append(fp / N)
                thresholds.append(s)
            prev_score = s

        if y == 1:
            tp += 1
        else:
            fp += 1

    if P > 0 and N > 0:
        tprs.append(tp / P)
        fprs.append(fp / N)
        thresholds.append(scores_sorted[-1])

    return np.array(fprs), np.array(tprs), np.array(thresholds)


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Computing Equal Error Rate (EER) and threshold.
    EER is the point where FPR ~= FNR, searched along the ROC curve.
    """
    fprs, tprs, thresholds = compute_roc(scores, labels)
    fnrs = 1.0 - tprs
    diffs = np.abs(fnrs - fprs)
    idx = np.argmin(diffs)
    eer = (fnrs[idx] + fprs[idx]) / 2.0
    thr = thresholds[idx]
    return float(eer), float(thr)


def compute_acc(scores: np.ndarray, labels: np.ndarray, thr: float) -> float:
    """
    Computing accuracy at a given threshold.

    score > thr -> spoof (1)
    score <= thr -> bona fide (0)
    """
    preds = (scores > thr).astype(int)
    correct = (preds == labels).sum()
    return float(correct) / len(labels) if len(labels) > 0 else 0.0


# ------------------------------
# Audio loading / preprocessing
# ------------------------------

# For CUDA:
def load_waveform(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Loading waveform from disk as mono, resample to target_sr if needed.

    Using soundfile for broad format support (FLAC, WAV) and torchaudio only for resampling.
    """
    wav_np, sr = sf.read(path)  # wav_np: [T] or [T, C]
    if wav_np.ndim == 1:
        wav_np = wav_np[None, :]  # [1, T]
    else:
        wav_np = wav_np.T  # [C, T]
        wav_np = wav_np.mean(axis=0, keepdims=True)  # converting to mono [1, T]

    wav = torch.from_numpy(wav_np.astype(np.float32))  # [1, T]

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav


# FOR MAC/ CPU ONLY
# def load_waveform(path: str, target_sr: int = 16000) -> torch.Tensor:
#     wav, sr = torchaudio.load(path)  # [C, T]
#     if wav.shape[0] > 1:
#         wav = wav.mean(dim=0, keepdim=True)
#     if sr != target_sr:
#         wav = torchaudio.functional.resample(wav, sr, target_sr)
#     return wav


def preprocess_waveform_for_rawnetlite(wav: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """
    Applying RawNetLite-style preprocessing.
    """
    with torch.no_grad():
        mean = wav.mean()
        std = wav.std()
        if std > 0:
            wav = (wav - mean) / std
        else:
            wav = wav * 0.0
    return wav


# --------------------------
# RawNetLite model loading
# --------------------------

def load_rawnetlite_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loading the pretrained RawNetLite model.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    # checkpoint is a plain state_dict
    try:
        model = RawNetLite()
        model.load_state_dict(ckpt)
    except Exception:
        # checkpoint is a dict with a 'model_state' or 'state_dict' key
        if isinstance(ckpt, dict):
            possible_keys = ["state_dict", "model_state", "model", "rawnet", "net"]
            state = None
            for k in possible_keys:
                if k in ckpt:
                    state = ckpt[k]
                    break
            if state is None:
                raise RuntimeError(
                    f"Could not find model state_dict in checkpoint keys: {ckpt.keys()}"
                )
            model = RawNetLite()
            model.load_state_dict(state)
        else:
            raise RuntimeError(
                "Unexpected checkpoint format. Please inspect the checkpoint "
                "and adjust load_rawnetlite_model accordingly."
            )

    model.to(device)
    model.eval()
    return model


# ---------------------------------------
# Inference on one split (dev or eval)
# ---------------------------------------

def run_split_inference(
    model: torch.nn.Module,
    device: torch.device,
    manifest_csv: str,
    out_scores_csv: str,
    max_utt: Optional[int] = None,
) -> List[ScoreEntry]:
    """
    Running RawNetLite inference for all utterances in a manifest CSV.
    """
    entries: List[ScoreEntry] = []

    os.makedirs(os.path.dirname(out_scores_csv), exist_ok=True)

    with open(manifest_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise RuntimeError(f"No header found in manifest CSV: {manifest_csv}")
        required = {"utt_id", "label", "path"}
        if not required.issubset(set(fieldnames)):
            raise RuntimeError(
                f"Manifest {manifest_csv} missing required columns {required}, "
                f"found: {fieldnames}"
            )

        rows = list(reader)
        total = len(rows)
        print(f"Found {total} utterances in {manifest_csv}")

        for i, row in enumerate(rows):
            if max_utt is not None and i >= max_utt:
                print(f"  Reached max_utt={max_utt}, stopping early for this split.")
                break

            if i % 100 == 0:
                print(f"  Processed {i}/{total} utterances...", flush=True)

            utt_id = row["utt_id"]
            label_str = row["label"].strip().lower()
            path = row["path"]

            label_int = 1 if label_str == "spoof" else 0

            # loading and preprocess audio
            wav = load_waveform(path, target_sr=16000)      # [1, T]
            wav = preprocess_waveform_for_rawnetlite(wav)  # still [1, T]
            wav = wav.to(device)

            with torch.no_grad():
                # RawNetLite expects input shape [B, 1, T]
                if wav.dim() == 1:
                    batch = wav.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
                elif wav.dim() == 2:
                    batch = wav.unsqueeze(1)               # [1, 1, T]
                elif wav.dim() == 3:
                    batch = wav                            # assuming [B, 1, T]
                else:
                    raise RuntimeError(f"Unexpected waveform shape: {wav.shape}")

                logits = model(batch)  # expecting [B, 1] or [B]
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                if logits.dim() == 2 and logits.size(1) == 1:
                    logits = logits.squeeze(1)
                score = float(logits.squeeze().item())

            entries.append(ScoreEntry(utt_id=utt_id, score=score, label_int=label_int))

    # writing scores CSV
    with open(out_scores_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utt_id", "score", "label_int"])
        for e in entries:
            writer.writerow([e.utt_id, f"{e.score:.9f}", e.label_int])

    print(f"  Wrote scores to {out_scores_csv}")
    return entries


# --------------
# Main script
# --------------

def main():
    parser = argparse.ArgumentParser(description="RawNetLite inference on ASVspoof 2019 LA")
    parser.add_argument(
        "--manifests_root",
        type=str,
        default="results/manifests",
        help="Root directory containing train/dev/eval manifest CSVs.",
    )
    parser.add_argument(
        "--dev_csv",
        type=str,
        default="dev.csv",
        help="DEV manifest filename (under manifests_root).",
    )
    parser.add_argument(
        "--eval_csv",
        type=str,
        default="eval.csv",
        help="EVAL manifest filename (under manifests_root).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pretrained_models/rawnetlite/models/rawnet_lite.pt",
        help="Path to pretrained RawNetLite checkpoint (.pt).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/models/rawnetlite",
        help="Base output directory for this RawNetLite run.",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional run tag for this experiment. If not set, a timestamp-based "
             "tag will be created.",
    )
    parser.add_argument(
        "--max_utt",
        type=int,
        default=None,
        help="Optional cap on number of utterances per split (DEV/EVAL) for fast testing.",
    )
    args = parser.parse_args()

    # CUDA if available, otherwise CPU (avoiding MPS)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Prepare run directory
    if args.run_tag is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_tag = f"rawnetlite_pretrained_{timestamp}"
    else:
        run_tag = args.run_tag

    run_dir = os.path.join(args.out_dir, run_tag)
    os.makedirs(run_dir, exist_ok=True)

    # saving a minimal run_config.json
    run_config = {
        "model_type": "rawnetlite",
        "checkpoint": os.path.abspath(args.checkpoint),
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
    print(f"Loading RawNetLite from {args.checkpoint} on device {device} ...")
    model = load_rawnetlite_model(args.checkpoint, device)

    # DEV and EVAL manifests
    dev_manifest = os.path.join(args.manifests_root, args.dev_csv)
    eval_manifest = os.path.join(args.manifests_root, args.eval_csv)

    # running inference on DEV
    print(f"\nRunning DEV inference using manifest: {dev_manifest}")
    dev_scores = run_split_inference(
        model=model,
        device=device,
        manifest_csv=dev_manifest,
        out_scores_csv=os.path.join(run_dir, "scores_dev.csv"),
        max_utt=args.max_utt,
    )
    dev_scores_arr = np.array([e.score for e in dev_scores], dtype=np.float64)
    dev_labels_arr = np.array([e.label_int for e in dev_scores], dtype=np.int32)

    # computing DEV EER and threshold
    dev_eer, dev_thr = compute_eer(dev_scores_arr, dev_labels_arr)
    dev_acc = compute_acc(dev_scores_arr, dev_labels_arr, dev_thr)
    print(f"\nDEV EER: {dev_eer*100:.2f}% at threshold {dev_thr:.6f}, ACC: {dev_acc*100:.2f}%")

    # running inference on EVAL
    print(f"\nRunning EVAL inference using manifest: {eval_manifest}")
    eval_scores = run_split_inference(
        model=model,
        device=device,
        manifest_csv=eval_manifest,
        out_scores_csv=os.path.join(run_dir, "scores_eval.csv"),
        max_utt=args.max_utt,
    )
    eval_scores_arr = np.array([e.score for e in eval_scores], dtype=np.float64)
    eval_labels_arr = np.array([e.label_int for e in eval_scores], dtype=np.int32)

    # computing EVAL EER at DEV threshold
    eval_eer, _ = compute_eer(eval_scores_arr, eval_labels_arr)
    eval_acc = compute_acc(eval_scores_arr, eval_labels_arr, dev_thr)
    print(f"\nEVAL EER: {eval_eer*100:.2f}%, ACC at DEV threshold: {eval_acc*100:.2f}%")

    # writing metrics.json
    metrics = {
        "model_type": "rawnetlite",
        "checkpoint": os.path.abspath(args.checkpoint),
        "dev": {
            "eer": dev_eer,
            "thr": dev_thr,
            "acc": dev_acc,
        },
        "eval": {
            "eer": eval_eer,
            "acc": eval_acc,
        },
        "run_tag": run_tag,
        "model_dir": os.path.abspath(run_dir),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nRawNetLite inference complete. Results saved under:\n  {run_dir}")


if __name__ == "__main__":
    main()