#!/usr/bin/env python3
"""
Build protocol-driven manifests for ASVspoof2019 LA.

Outputs (per split):
- results/manifests/<split>.csv        (utt_id,label,system_id,attack_id,path)
- results/manifests/<split>.lock.json  (protocol path + SHA256)
"""

from __future__ import annotations
import argparse, csv, hashlib, json, re
from pathlib import Path
from collections import Counter

import yaml

# Accept LA_* with optional -N and optional .flac (case-insensitive)
LA_UTT_RE = re.compile(r"^(LA_[A-Z]_\d{7})(?:-\d+)?(?:\.flac)?$", re.IGNORECASE)

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def norm_utt(token: str) -> str:
    m = LA_UTT_RE.match(token.strip())
    return m.group(1).upper() if m else token

def load_cfg(cfg_path: Path) -> dict:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["dataset"]["root"] = str(Path(cfg["paths"]["dataset"]["root"]).expanduser().resolve())
    return cfg

def read_cm_protocol_lines(p: Path):
    """
    README format: SPEAKER_ID  AUDIO_FILE_NAME  -  SYSTEM_ID  KEY
    Extract:
      utt_id   := AUDIO_FILE_NAME (normalized, ext stripped)
      system_id:= 'Axx' or '-' (bonafide)
      label    := 'bonafide' | 'spoof' (eval may hide in some releases; your copy has them)
    """
    rows = []
    with p.open() as f:
        for line in f:
            ln = line.strip()
            if not ln or ln.startswith("#"):
                continue
            toks = ln.split()

            utt_idx, utt = None, None
            for i, t in enumerate(toks):
                if LA_UTT_RE.match(t):
                    utt_idx, utt = i, norm_utt(t)
                    break

            system_id, label = None, None
            if utt_idx is not None:
                post = toks[utt_idx + 1:]
                if len(post) >= 3 and post[0] == "-":
                    system_id = post[1].upper()
                    label = post[2].lower()
                else:
                    # fallback scan
                    for t in post:
                        tu = t.upper()
                        if tu == "-" or (tu.startswith("A") and tu[1:].isdigit()):
                            system_id = tu
                            break
                    label = post[-1].lower() if post else None

            if label not in {"bonafide","spoof"}:
                label = None

            attack_id = system_id.upper() if system_id and system_id.upper().startswith("A") and system_id[1:].isdigit() else ""
            rows.append({"utt_id": utt, "label": label or "", "attack_id": attack_id})
    return rows

def map_utts_to_paths(audio_dir: Path):
    return {norm_utt(p.name): p for p in audio_dir.glob("*.flac")}

def build_manifest_for_split(split: str, cfg: dict) -> dict:
    ds = cfg["paths"]["dataset"]; cm = cfg["data"]["cm_protocol_files"]
    root = Path(ds["root"])
    audio_dir = root / ds[f"{split}_audio"]
    proto_path = root / ds["cm_protocols"] / cm[split]

    problems = []
    if not audio_dir.exists(): problems.append(f"[{split}] missing audio dir: {audio_dir}")
    if not proto_path.exists(): problems.append(f"[{split}] missing protocol: {proto_path}")
    if problems: return {"ok": False, "problems": problems}

    rows = read_cm_protocol_lines(proto_path)
    idx  = map_utts_to_paths(audio_dir)

    out_rows, missing = [], []
    for r in rows:
        u = r["utt_id"]
        if not u: continue
        p = idx.get(u)
        if p is None:
            missing.append(u)
            continue
        out_rows.append({**r, "path": str(p)})

    counts = Counter([r["label"] for r in out_rows if r["label"]])
    return {
        "ok": len(missing) == 0,
        "split": split,
        "protocol_path": str(proto_path.resolve()),
        "protocol_sha256": sha256_of(proto_path),
        "audio_dir": str(audio_dir.resolve()),
        "n_protocol": len(rows),
        "n_manifest": len(out_rows),
        "label_counts": dict(counts),
        "missing_utts": missing,
        "rows": out_rows,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--splits", nargs="+", default=["train","dev","eval"])
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    out_root = Path("results/manifests"); out_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        res = build_manifest_for_split(split, cfg)
        print(f"[{split.upper()}]")
        if not res["ok"]:
            print("  ! problems:", res["problems"])
            continue
        print(f"  protocol: {res['protocol_path']}")
        print(f"  audio dir: {res['audio_dir']}")
        print(f"  rows: {res['n_manifest']}  labels: {res['label_counts']}")

        # write CSV manifest
        csv_path = out_root / f"{split}.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["utt_id","label","attack_id","path"])
            for r in res["rows"]:
                w.writerow([r["utt_id"], r["label"], r["attack_id"], r["path"]])

        # write protocol lock
        lock_path = out_root / f"{split}.lock.json"
        with lock_path.open("w") as f:
            json.dump({
                "split": split,
                "protocol_file": res["protocol_path"],
                "sha256": res["protocol_sha256"]
            }, f, indent=2)

    print("Manifests ready")

if __name__ == "__main__":
    main()
