"""
01_verify_asvspoof.py

Purpose:
    To verify that the ASVspoof 2019 LA dataset is correctly installed and configured in configs/paths.yaml
    This script checks:
      - all protocol files exist
      - all audio paths listed in the protocols exist
      - formats and directory structures are valid

Inputs:
    configs/paths.yaml  (points to dataset root)

Outputs:
    Printed summary of dataset status (missing files, errors, sample checks)

Notes:
    This script does not modify any data; it only verifies the environment.
"""

from __future__ import annotations
import argparse
import random
import re
from collections import Counter
from pathlib import Path

import yaml

try:
    import soundfile as sf
except Exception:
    sf = None

# accepting LA_[DTI]_####### with optional "-N" and optional ".flac" (case-insensitive)
LA_UTT_RE = re.compile(r"^(LA_[DTE]_\d{7})(?:-\d+)?(?:\.flac)?$", re.IGNORECASE)

def norm_utt(token: str) -> str:
    """
    Normalize variations:
      'LA_E_1234567-3.flac' -> 'LA_E_1234567'
      'la_d_7654321.flac'   -> 'LA_D_7654321'
    """
    t = token.strip()
    m = LA_UTT_RE.match(t)
    if not m:
        return t
    return m.group(1).upper()

def load_cfg(cfg_path: Path) -> dict:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["dataset"]["root"] = str(Path(cfg["paths"]["dataset"]["root"]).expanduser().resolve())
    return cfg

def read_cm_protocol_lines(p: Path):
    """
    Expected (per README):
      SPEAKER_ID  AUDIO_FILE_NAME  -  SYSTEM_ID  KEY
    Extract:
      utt_id   := AUDIO_FILE_NAME (normalized, extension stripped)
      system_id:= SYSTEM_ID (A01..A19 or '-'; may be None if irregular)
      label    := KEY ('bonafide' | 'spoof'); may be None on eval releases
    """
    rows = []
    with p.open() as f:
        for line in f:
            ln = line.strip()
            if not ln or ln.startswith("#"):
                continue
            toks = ln.split()

            # finding utt token by regex (robust to columns)
            utt_idx, utt = None, None
            for i, t in enumerate(toks):
                if LA_UTT_RE.match(t):
                    utt_idx, utt = i, norm_utt(t)
                    break

            system_id, attack_id, label = None, None, None
            if utt_idx is not None:
                # canonical layout after utt: '-' SYSTEM_ID KEY
                post = toks[utt_idx + 1:]
                # finding a '-' and then take next as system, last as key
                if post:
                    # system id in LA is typically 'A01'..'A19' or '-' (bonafide)
                    # attack_id isn't a separate column in LA cm readme; many papers refer to SYSTEM_ID as attack ID.
                    # I'll treat system_id as the attack code (Axx).
                    # trying canonical positions:
                    if len(post) >= 3 and post[0] == "-":
                        system_id = post[1].upper()
                        label = post[2].lower()
                    else:
                        # fallback: scan post for first token that looks like Axx or '-'
                        for t in post:
                            tu = t.upper()
                            if tu == "-" or (tu.startswith("A") and tu[1:].isdigit()):
                                system_id = tu
                                break
                        label = post[-1].lower() if post else None

                    # attack_id equals Axx for LA; keeping both keys for convenience
                    if system_id and system_id.upper().startswith("A") and system_id[1:].isdigit():
                        attack_id = system_id.upper()

            # sanitizing the label
            if label not in {"bonafide", "spoof"}:
                label = None

            rows.append({
                "utt_id": utt,
                "label": label,
                "system_id": system_id,
                "attack_id": attack_id,
                "raw_tokens": toks,
                "line": ln,
            })
    return rows

def build_audio_index(audio_dir: Path):
    idx = {}
    for p in audio_dir.glob("*.flac"):
        idx[norm_utt(p.name)] = p
    return idx

def verify_split(split: str, ds_root: Path, audio_rel: str, proto_dir_rel: str, proto_fname: str):
    audio_dir = ds_root / audio_rel
    proto_path = ds_root / proto_dir_rel / proto_fname

    problems = []

    if not audio_dir.exists():
        problems.append(f"[{split}] audio dir missing: {audio_dir}")
    if not proto_path.exists():
        problems.append(f"[{split}] protocol missing: {proto_path}")
    if problems:
        return {"split": split, "ok": False, "problems": problems}

    rows = read_cm_protocol_lines(proto_path)
    idx = build_audio_index(audio_dir)

    with_utt = [r for r in rows if r["utt_id"]]
    labels = Counter([r["label"] for r in with_utt if r["label"] is not None])
    prot_utts = [r["utt_id"] for r in with_utt]
    aud_utts = set(idx.keys())

    missing_audio = [u for u in prot_utts if u not in aud_utts]
    extra_audio = [u for u in aud_utts if u not in set(prot_utts)]

    spoof_rows = [r for r in with_utt if r.get("label") == "spoof"]
    sys_counts = Counter([r.get("system_id") for r in spoof_rows if r.get("system_id")])

    summary = {
        "split": split,
        "protocol_file": str(proto_path),
        "audio_dir": str(audio_dir),
        "n_protocol_lines": len(rows),
        "n_protocol_with_utt": len(with_utt),
        "label_counts": dict(labels),
        "n_audio_files": len(aud_utts),
        "n_missing_audio": len(missing_audio),
        "n_extra_audio": len(extra_audio),
        "missing_audio_examples": missing_audio[:5],
        "extra_audio_examples": extra_audio[:5],
        "system_counts": dict(sys_counts),
        "ok": len(problems) == 0 and len(missing_audio) == 0,
        "problems": problems,
    }
    return summary

def audio_spotcheck(sr_expected: int, audio_paths, k=5):
    if sf is None:
        return {"ran": False, "error": "soundfile not available"}
    out = {"ran": True, "checked": 0, "mismatch": []}
    sample = random.sample(audio_paths, min(k, len(audio_paths)))
    for p in sample:
        try:
            info = sf.info(p)
            if info.samplerate != sr_expected:
                out["mismatch"].append({"path": str(p), "sr": info.samplerate})
            out["checked"] += 1
        except Exception as e:
            out["mismatch"].append({"path": str(p), "error": str(e)})
    return out

def main():
    ap = argparse.ArgumentParser(description="Verify ASVspoof2019 LA dataset using YAML config.")
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--audio-check", action="store_true",
                    help="Randomly open a few files per split and verify sample rate.")
    ap.add_argument("--n-audio-check", type=int, default=8)
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    ds = cfg["paths"]["dataset"]
    cm = cfg["data"]["cm_protocol_files"]
    ds_root = Path(ds["root"])

    splits = [("train", ds["train_audio"], cm["train"]),
              ("dev",   ds["dev_audio"],   cm["dev"]),
              ("eval",  ds["eval_audio"],  cm["eval"])]

    print("=== ASVspoof2019 LA verification ===")
    print(f"Root: {ds_root}\n")

    all_ok = True
    audio_paths_for_spot = []

    for split, audio_rel, proto_name in splits:
        res = verify_split(split, ds_root, audio_rel, ds["cm_protocols"], proto_name)
        all_ok = all_ok and res["ok"]
        print(f"[{split.upper()}]")
        print(f"  audio dir     : {res.get('audio_dir')}")
        print(f"  protocol file : {res.get('protocol_file')}")
        print(f"  protocol lines: {res.get('n_protocol_lines')}  (with utt: {res.get('n_protocol_with_utt')})")
        print(f"  audio files   : {res.get('n_audio_files')}")
        print(f"  labels        : {res.get('label_counts')}")
        print(f"  missing audio : {res.get('n_missing_audio')}")
        if res["missing_audio_examples"]:
            print(f"    e.g. {res['missing_audio_examples']}")
        print(f"  extra audio   : {res.get('n_extra_audio')}")
        if res["extra_audio_examples"]:
            print(f"    e.g. {res['extra_audio_examples']}")
        if res["system_counts"]:
            # showing top 10 by count for verification
            top_sys = sorted(res["system_counts"].items(), key=lambda x: -x[1])[:10]
            print(f"  spoof by system (top): {top_sys}")
        if res["problems"]:
            for p in res["problems"]:
                print("  !", p)
        print()

        if args.audio_check:
            audio_dir = Path(res["audio_dir"])
            audio_paths_for_spot.extend(list(audio_dir.glob("*.flac")))

    if args.audio_check and audio_paths_for_spot:
        random.shuffle(audio_paths_for_spot)
        out = audio_spotcheck(sr_expected=int(cfg["audio"]["sr"]),
                              audio_paths=audio_paths_for_spot,
                              k=args.n_audio_check)
        print("[Audio spot-check]")
        if not out["ran"]:
            print("  (skipped) soundfile not installed.")
        else:
            print(f"  checked: {out['checked']} files at {cfg['audio']['sr']} Hz")
            if out["mismatch"]:
                print("  mismatches:")
                for m in out["mismatch"]:
                    print("   -", m)
                all_ok = False
            else:
                print("all good")
        print()

    print("=== SUMMARY ===")
    print("Status:", "OK" if all_ok else "Issues found")
    exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()