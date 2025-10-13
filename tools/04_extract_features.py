#!/usr/bin/env python3
"""
04_extract_features.py  —  zero-argument, space-aware extractor

- Reads YAML config (configs/paths.yaml)
- Reads manifests: results/manifests/{train,dev,eval}.csv
- Extracts BOTH LFCC and MFCC (from YAML params) by default
- Saves compact: float16 + compressed .npz
- Auto-selects an output root with enough free space:
    1) results/features
    2) ~/asv_features
    3) /Volumes/*/asv_features  (macOS external drives)
- Skips existing files; appends to index.csv incrementally
- If disk fills mid-run, automatically falls back to the next location

Run:
    python tools/04_extract_features.py
"""

from __future__ import annotations
import csv, json, math, os, re, shutil, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from tqdm import tqdm

# Audio IO + DSP
try:
    import soundfile as sf
except Exception:
    sf = None
import librosa
from scipy.fftpack import dct as scipy_dct

LA_UTT_RE = re.compile(r"^(LA_[A-Z]_\d{7})$", re.IGNORECASE)

# -------------------------
# Config dataclasses
# -------------------------
@dataclass
class FeatParams:
    frame_length_ms: float
    frame_hop_ms: float
    sr: int
    # LFCC
    lfcc_n_filters: int
    lfcc_n_ceps: int
    lfcc_keep_c0: bool
    lfcc_deltas: bool
    # MFCC
    mfcc_n_mels: int
    mfcc_n_ceps: int
    mfcc_keep_c0: bool
    mfcc_deltas: bool


# -------------------------
# YAML / params
# -------------------------
def load_cfg(cfg_path: Path) -> Dict:
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["dataset"]["root"] = str(Path(cfg["paths"]["dataset"]["root"]).expanduser().resolve())
    return cfg

def load_params(cfg: Dict) -> FeatParams:
    sr = int(cfg["audio"]["sr"])
    fl_ms = float(cfg["features"]["frame_length_ms"])
    fh_ms = float(cfg["features"]["frame_hop_ms"])
    return FeatParams(
        frame_length_ms=fl_ms,
        frame_hop_ms=fh_ms,
        sr=sr,
        lfcc_n_filters=int(cfg["features"]["lfcc"]["n_filters"]),
        lfcc_n_ceps=int(cfg["features"]["lfcc"]["n_ceps"]),
        lfcc_keep_c0=bool(cfg["features"]["lfcc"]["keep_c0"]),
        lfcc_deltas=bool(cfg["features"]["lfcc"]["deltas"]),
        mfcc_n_mels=int(cfg["features"]["mfcc"]["n_mels"]),
        mfcc_n_ceps=int(cfg["features"]["mfcc"]["n_ceps"]),
        mfcc_keep_c0=bool(cfg["features"]["mfcc"]["keep_c0"]),
        mfcc_deltas=bool(cfg["features"]["mfcc"]["deltas"]),
    )

# -------------------------
# Space helpers
# -------------------------
def free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return usage.free

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def pick_out_root(need_bytes: int) -> Path:
    """
    Choose an output root with >= need_bytes free.
    Order:
      1) results/features
      2) ~/asv_features
      3) /Volumes/*/asv_features (macOS external drives)
    If none have enough space, still return the first candidate.
    """
    candidates: List[Path] = [
        Path("results/features"),
        Path.home() / "asv_features",
    ]
    vol_dir = Path("/Volumes")
    if vol_dir.exists():
        for d in vol_dir.iterdir():
            if d.is_dir() and not str(d.name).startswith("."):
                candidates.append(d / "asv_features")

    chosen = candidates[0]
    best_free = -1
    for c in candidates:
        try:
            ensure_dir(c)
            fb = free_bytes(c)
            if fb > best_free:
                best_free = fb
            if fb >= need_bytes:
                return c
        except Exception:
            continue
    # Not enough anywhere; pick the one with the most free space
    return max(candidates, key=lambda p: (free_bytes(p) if p.exists() else -1))

def approx_bytes_per_file(dims: int, frames_est: int, dtype_bytes: int = 2, compressed: bool = True) -> int:
    """
    Rough estimate for planning. dtype default=2 (float16).
    compressed=True reduces size ~30–60%; we assume 50% here.
    """
    raw = frames_est * dims * dtype_bytes
    return int(raw * (0.5 if compressed else 1.0))

# -------------------------
# Audio / DSP utils
# -------------------------
def next_pow2(n: int) -> int:
    return int(2 ** math.ceil(math.log2(max(1, n))))

def ms_to_samples(ms: float, sr: int) -> int:
    return int(round(ms * 0.001 * sr))

def ensure_audio(y: np.ndarray, sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    if y.ndim > 1:
        y = librosa.to_mono(y.T if y.shape[0] < y.shape[1] else y)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
        sr = target_sr
    return y.astype(np.float32, copy=False), sr

def read_audio(path: Path, target_sr: int) -> np.ndarray:
    if sf is None:
        raise RuntimeError("soundfile is required to read audio.")
    y, sr = sf.read(str(path), always_2d=False)
    if isinstance(y, np.ndarray) and y.dtype.kind in {"i", "u"}:
        y = y.astype(np.float32) / np.iinfo(y.dtype).max
    y, _ = ensure_audio(np.asarray(y, dtype=np.float32), sr, target_sr)
    return y

def power_spectrogram(y: np.ndarray, sr: int, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hann", center=True)
    return (np.abs(S) ** 2).astype(np.float32)

# -------------------------
# LFCC / MFCC
# -------------------------
def linear_triangular_filterbank(n_fft: int, sr: int, n_filters: int) -> np.ndarray:
    freqs = np.linspace(0.0, sr / 2.0, num=n_fft // 2 + 1)
    f_min, f_max = 0.0, sr / 2.0
    centers = np.linspace(f_min, f_max, num=n_filters + 2)
    fb = np.zeros((n_filters, len(freqs)), dtype=np.float32)
    for m in range(1, n_filters + 1):
        f_l, f_c, f_r = centers[m - 1], centers[m], centers[m + 1]
        left = np.logical_and(freqs >= f_l, freqs <= f_c)
        fb[m - 1, left] = (freqs[left] - f_l) / max(1e-12, (f_c - f_l))
        right = np.logical_and(freqs >= f_c, freqs <= f_r)
        fb[m - 1, right] = (f_r - freqs[right]) / max(1e-12, (f_r - f_c))
    return fb

def lfcc_from_powspec(P: np.ndarray, fb: np.ndarray, n_ceps: int, keep_c0: bool, add_deltas: bool) -> np.ndarray:
    E = np.dot(fb, P)                              # (n_filters x n_frames)
    E[E <= 1e-12] = 1e-12
    logE = np.log(E).astype(np.float32)
    C = scipy_dct(logE, type=2, axis=0, norm="ortho")    # (n_filters x n_frames)
    cep = (C[:n_ceps, :].T if keep_c0 else C[1:n_ceps+1, :].T)
    if add_deltas:
        d1 = librosa.feature.delta(cep, order=1, mode="nearest")
        d2 = librosa.feature.delta(cep, order=2, mode="nearest")
        cep = np.concatenate([cep, d1, d2], axis=1)
    return cep.astype(np.float32)

def mfcc_from_audio(y: np.ndarray, sr: int, n_fft: int, hop_length: int, win_length: int,
                    n_mels: int, n_ceps: int, keep_c0: bool, add_deltas: bool) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_ceps if keep_c0 else n_ceps + 1,
        n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        n_mels=n_mels, htk=True, center=True, window="hann", power=2.0
    )
    if not keep_c0:
        mfcc = mfcc[1:, :]
    X = mfcc.T.astype(np.float32)
    if add_deltas:
        d1 = librosa.feature.delta(mfcc, order=1, mode="nearest").T
        d2 = librosa.feature.delta(mfcc, order=2, mode="nearest").T
        X = np.concatenate([X, d1.astype(np.float32), d2.astype(np.float32)], axis=1)
    return X

def extract_lfcc(y: np.ndarray, sr: int, params: FeatParams) -> np.ndarray:
    frame_len = ms_to_samples(params.frame_length_ms, sr)
    hop_len   = ms_to_samples(params.frame_hop_ms, sr)
    n_fft     = next_pow2(frame_len)
    P = power_spectrogram(y, sr, n_fft=n_fft, hop_length=hop_len, win_length=frame_len)
    P = P[: n_fft // 2 + 1, :]
    fb = linear_triangular_filterbank(n_fft=n_fft, sr=sr, n_filters=params.lfcc_n_filters)
    X = lfcc_from_powspec(P, fb, n_ceps=params.lfcc_n_ceps, keep_c0=params.lfcc_keep_c0, add_deltas=params.lfcc_deltas)
    return X

def extract_mfcc(y: np.ndarray, sr: int, params: FeatParams) -> np.ndarray:
    frame_len = ms_to_samples(params.frame_length_ms, sr)
    hop_len   = ms_to_samples(params.frame_hop_ms, sr)
    n_fft     = next_pow2(frame_len)
    X = mfcc_from_audio(
        y, sr, n_fft=n_fft, hop_length=hop_len, win_length=frame_len,
        n_mels=params.mfcc_n_mels, n_ceps=params.mfcc_n_ceps,
        keep_c0=params.mfcc_keep_c0, add_deltas=params.mfcc_deltas
    )
    return X

# -------------------------
# Manifests
# -------------------------
def load_manifest(split: str) -> List[Dict[str, str]]:
    mpath = Path("results/manifests") / f"{split}.csv"
    if not mpath.exists():
        raise FileNotFoundError(f"Manifest not found: {mpath}")
    rows = []
    with mpath.open() as f:
        r = csv.DictReader(f)
        need = {"utt_id", "label", "attack_id", "path"}
        if set(r.fieldnames) & need != need:
            raise ValueError(f"{mpath} must have columns {sorted(need)}")
        for d in r:
            rows.append(d)
    return rows

# -------------------------
# Saving
# -------------------------
def save_array_npz(out_base: Path, X: np.ndarray):
    """Save compressed .npz (float16) as out_base(.npz)."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_base.with_suffix(".npz"), feat=X.astype(np.float16, copy=False))

def load_shape_if_exists(out_base: Path) -> Tuple[int, int] | None:
    npz = out_base.with_suffix(".npz")
    npy = out_base.with_suffix(".npy")
    try:
        if npz.exists():
            Z = np.load(npz)
            return tuple(Z["feat"].shape)  # type: ignore
        if npy.exists():
            A = np.load(npy, mmap_mode="r")
            return tuple(A.shape)  # type: ignore
    except Exception:
        return None
    return None

# -------------------------
# Main
# -------------------------
def main():
    cfg = load_cfg(Path("configs/paths.yaml"))
    params = load_params(cfg)

    # Defaults: both features, all splits, skip-existing true
    features = ["lfcc", "mfcc"]
    splits   = ["train", "dev", "eval"]
    skip_existing = True

    # Rough size planning (very conservative):
    # Assume median ~600 frames/utt, dims ~60 (20 ceps * [1 or 3] depending on deltas; YAML uses deltas=true)
    # float16 + compressed .npz ≈ ~50% of raw
    dims_guess = 60
    frames_guess = 600
    per_file = approx_bytes_per_file(dims_guess, frames_guess, dtype_bytes=2, compressed=True)
    # Count total utterances from manifests we find
    total_utts = 0
    for sp in splits:
        try:
            total_utts += len(load_manifest(sp))
        except Exception:
            pass
    # for two features
    need_bytes = per_file * total_utts * len(features)
    # add margin
    need_bytes = int(need_bytes * 1.3)

    out_root = pick_out_root(need_bytes)
    ensure_dir(out_root)
    print(f"[extract] output root: {out_root} (free ~{free_bytes(out_root)/1e9:.2f} GB)")

    # processing
    for feat in features:
        feat_root = out_root / feat
        ensure_dir(feat_root)

        # snapshot params for meta
        if feat == "lfcc":
            snap = dict(
                frame_length_ms=params.frame_length_ms, frame_hop_ms=params.frame_hop_ms, sr=params.sr,
                n_filters=params.lfcc_n_filters, n_ceps=params.lfcc_n_ceps,
                keep_c0=params.lfcc_keep_c0, deltas=params.lfcc_deltas,
                storage="float16_npz"
            )
        else:
            snap = dict(
                frame_length_ms=params.frame_length_ms, frame_hop_ms=params.frame_hop_ms, sr=params.sr,
                n_mels=params.mfcc_n_mels, n_ceps=params.mfcc_n_ceps,
                keep_c0=params.mfcc_keep_c0, deltas=params.mfcc_deltas,
                storage="float16_npz"
            )

        for split in splits:
            try:
                rows = load_manifest(split)
            except FileNotFoundError as e:
                print(f"[warn] {e}")
                continue

            split_dir = feat_root / split
            ensure_dir(split_dir)
            idx_path = split_dir / "index.csv"
            wrote_header = idx_path.exists()

            n_ok = n_err = 0
            pbar = tqdm(rows, desc=f"{feat}:{split}", unit="utt")

            for r in pbar:
                utt = r["utt_id"]
                if not LA_UTT_RE.match(utt or ""):
                    n_err += 1
                    continue
                wav_p = Path(r["path"])
                out_base = split_dir / utt  # we will save as .npz

                if skip_existing and load_shape_if_exists(out_base) is not None:
                    shape = load_shape_if_exists(out_base)
                    n_frames, n_dims = (shape if shape is not None else (-1, -1))
                    with idx_path.open("a", newline="") as f:
                        w = csv.writer(f)
                        if not wrote_header:
                            w.writerow(["utt_id", "label", "attack_id", "path", "n_frames", "n_dims"])
                            wrote_header = True
                        w.writerow([utt, r["label"], r["attack_id"], str(wav_p), n_frames, n_dims])
                    n_ok += 1
                    continue

                try:
                    y = read_audio(wav_p, params.sr)
                    if feat == "lfcc":
                        X = extract_lfcc(y, params.sr, params)
                    else:
                        X = extract_mfcc(y, params.sr, params)
                    save_array_npz(out_base, X)

                    with idx_path.open("a", newline="") as f:
                        w = csv.writer(f)
                        if not wrote_header:
                            w.writerow(["utt_id", "label", "attack_id", "path", "n_frames", "n_dims"])
                            wrote_header = True
                        w.writerow([utt, r["label"], r["attack_id"], str(wav_p), X.shape[0], X.shape[1]])
                    n_ok += 1

                except OSError as e:
                    # Disk full on current out_root — try to switch root and continue
                    if "No space left on device" in str(e):
                        print("\n[warn] disk full; searching for alternative output location...")
                        alt_root = pick_out_root(need_bytes // 2)  # try again with smaller requirement
                        if alt_root.resolve() != out_root.resolve():
                            print(f"[info] switching output to: {alt_root}")
                            out_root = alt_root
                            feat_root = out_root / feat
                            split_dir = feat_root / split
                            ensure_dir(split_dir)
                            idx_path = split_dir / "index.csv"
                            wrote_header = idx_path.exists()
                            # retry save once
                            try:
                                save_array_npz(split_dir / utt, X)
                                with idx_path.open("a", newline="") as f:
                                    w = csv.writer(f)
                                    if not wrote_header:
                                        w.writerow(["utt_id", "label", "attack_id", "path", "n_frames", "n_dims"])
                                        wrote_header = True
                                    w.writerow([utt, r["label"], r["attack_id"], str(wav_p), X.shape[0], X.shape[1]])
                                n_ok += 1
                                continue
                            except Exception as e2:
                                print(f"[error] retry failed after switching disk: {e2}")
                        # if cannot switch, record error and continue
                    n_err += 1
                    with idx_path.open("a", newline="") as f:
                        w = csv.writer(f)
                        if not wrote_header:
                            w.writerow(["utt_id", "label", "attack_id", "path", "n_frames", "n_dims"])
                            wrote_header = True
                        w.writerow([utt, r["label"], r["attack_id"], str(wav_p), -1, -1])
                    pbar.set_postfix_str("write error")

                except Exception as e:
                    n_err += 1
                    with idx_path.open("a", newline="") as f:
                        w = csv.writer(f)
                        if not wrote_header:
                            w.writerow(["utt_id", "label", "attack_id", "path", "n_frames", "n_dims"])
                            wrote_header = True
                        w.writerow([utt, r["label"], r["attack_id"], str(wav_p), -1, -1])
                    pbar.set_postfix_str(f"error: {e}")

            # meta
            meta = {
                "feature": feat,
                "split": split,
                "params": snap,
                "counts": {"ok": n_ok, "error": n_err, "total": n_ok + n_err},
                "out_root": str(out_root.resolve()),
            }
            (split_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            print(f"[{feat}:{split}] -> {split_dir}  ok={n_ok} err={n_err}")

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
