#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, re, sys, hashlib, random
from pathlib import Path
from collections import Counter

# Optional audio probe
try:
    import soundfile as sf
except Exception:
    sf = None

LA_UTT_RE = re.compile(r"^(LA_[A-Z]_\d{7})(?:-\d+)?(?:\.flac)?$", re.IGNORECASE)
ATTACK_RE = re.compile(r"^A\d{2}$")

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def norm_utt(token: str) -> str:
    m = LA_UTT_RE.match(token.strip())
    return m.group(1).upper() if m else token

def parse_protocol(protocol_path: Path):
    """Return dict utt_id -> (label, attack_id) from CM protocol (robust to spacing)."""
    rows = {}
    with protocol_path.open() as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"): continue
            toks = s.split()
            # find audio token
            utt_idx, utt = None, None
            for i, t in enumerate(toks):
                if LA_UTT_RE.match(t):
                    utt_idx, utt = i, norm_utt(t)
                    break
            if utt is None: continue
            # canonical: after utt -> "-" SYSTEM_ID KEY
            label, attack = None, ""
            post = toks[utt_idx+1:]
            if len(post) >= 3 and post[0] == "-":
                sysid = post[1].upper()
                label = post[2].lower()
            else:
                # fallback scan
                sysid = None
                for t in post:
                    tu = t.upper()
                    if tu == "-" or ATTACK_RE.match(tu):
                        sysid = tu; break
                label = (post[-1].lower() if post else None)
            if label not in {"bonafide","spoof"}:
                label = None
            attack = sysid if (label=="spoof" and ATTACK_RE.match(sysid or "")) else ""
            rows[utt] = (label or "", attack)
    return rows

def load_manifest(split: str):
    mpath = Path("results/manifests")/f"{split}.csv"
    lpath = Path("results/manifests")/f"{split}.lock.json"
    if not mpath.exists() or not lpath.exists():
        raise FileNotFoundError(f"Missing manifest or lock for split '{split}'")
    lock = json.loads(lpath.read_text())
    rows = []
    with mpath.open() as f:
        r = csv.DictReader(f)
        need = {"utt_id","label","attack_id","path"}
        if set(r.fieldnames) & need != need:
            raise ValueError(f"{mpath} must have columns {sorted(need)}")
        for d in r:
            rows.append(d)
    return rows, lock, mpath, lpath

def spotcheck_audio(paths, k=8):
    if sf is None:
        return {"ran": False, "error": "soundfile not installed"}
    out = {"ran": True, "checked": 0, "errors": []}
    sample = random.sample(paths, min(k, len(paths)))
    for p in sample:
        try:
            _ = sf.info(str(p))
            out["checked"] += 1
        except Exception as e:
            out["errors"].append((str(p), str(e)))
    return out

def verify_split(split: str, strict: bool, audio_check: int|None):
    issues = []
    rows, lock, mpath, lpath = load_manifest(split)

    # 1) Lock file verification
    protocol_path = Path(lock["protocol_file"])
    sha_cur = sha256_of(protocol_path) if protocol_path.exists() else None
    if not protocol_path.exists():
        issues.append(f"[{split}] protocol file missing: {protocol_path}")
    elif sha_cur != lock.get("sha256"):
        issues.append(f"[{split}] protocol SHA mismatch: lock={lock.get('sha256')} cur={sha_cur}")

    # 2) Reload protocol to cross-check
    proto_map = parse_protocol(protocol_path) if protocol_path.exists() else {}

    # 3) Row-by-row checks
    seen_utts, seen_paths = set(), set()
    n_ok = 0
    label_counts = Counter()
    attack_counts = Counter()

    for i, r in enumerate(rows, 1):
        utt = r["utt_id"].strip()
        lab = r["label"].strip().lower()
        atk = r["attack_id"].strip().upper()
        pth = Path(r["path"])

        # utt format
        if not LA_UTT_RE.match(utt):
            issues.append(f"[{split}] bad utt_id format at row {i}: {utt}")

        # label validity
        if lab not in {"bonafide","spoof"}:
            issues.append(f"[{split}] invalid label at row {i}: {lab}")

        # attack/label consistency
        if lab == "bonafide":
            if atk != "":
                issues.append(f"[{split}] bonafide must have empty attack_id at row {i}: got {atk}")
        else:
            if not ATTACK_RE.match(atk):
                issues.append(f"[{split}] spoof must have attack_id Axx at row {i}: got {atk}")

        # file existence
        if not pth.exists():
            issues.append(f"[{split}] missing file at row {i}: {pth}")

        # utt ↔ filename consistency
        stem = pth.stem
        if norm_utt(stem) != utt:
            issues.append(f"[{split}] utt_id/path mismatch at row {i}: utt={utt}, file={stem}")

        # duplicates
        if utt in seen_utts:
            issues.append(f"[{split}] duplicate utt_id: {utt}")
        if str(pth) in seen_paths:
            issues.append(f"[{split}] duplicate path: {pth}")
        seen_utts.add(utt); seen_paths.add(str(pth))

        # protocol agreement (if available)
        if proto_map:
            plab, patk = proto_map.get(utt, ("",""))
            if plab != lab or patk != atk:
                issues.append(f"[{split}] protocol mismatch for {utt}: manifest=({lab},{atk}) proto=({plab},{patk})")

        n_ok += 1
        if lab: label_counts[lab]+=1
        if atk: attack_counts[atk]+=1

    # 4) sanity vs protocol totals
    if proto_map and strict:
        if len(proto_map) != len(rows):
            issues.append(f"[{split}] manifest row count {len(rows)} != protocol utts {len(proto_map)}")

    # 5) optional audio spotcheck
    if audio_check and audio_check > 0 and sf is not None:
        probe = spotcheck_audio([Path(r["path"]) for r in rows], k=audio_check)
        if probe["ran"] and probe["errors"]:
            for p,e in probe["errors"]:
                issues.append(f"[{split}] audio read error: {p} -> {e}")

    # report
    print(f"[{split.upper()}]")
    print(f"  rows         : {n_ok}")
    print(f"  labels       : {dict(label_counts)}")
    if attack_counts:
        items = sorted(attack_counts.items(), key=lambda x: (-x[1], x[0]))
        print(f"  attacks(all) : {items}")
    if issues:
        print(f" {len(issues)} issues:")
        for msg in issues[:20]:
            print("   -", msg)
        if len(issues) > 20:
            print(f"   ... (+{len(issues)-20} more)")
    else:
        print("manifest OK")
    return 0 if not issues else 1

def main():
    ap = argparse.ArgumentParser(description="Verify results/manifests/*.csv correctness.")
    ap.add_argument("--splits", nargs="+", default=["train","dev","eval"])
    ap.add_argument("--strict", action="store_true",
                    help="fail if manifest row count != protocol utts; otherwise only content mismatches fail")
    ap.add_argument("--audio-check", type=int, default=0,
                    help="spot-check N random files via soundfile")
    args = ap.parse_args()

    rc = 0
    for sp in args.splits:
        rc |= verify_split(sp, strict=args.strict, audio_check=args.audio_check)
    # cross-split overlap check
    if rc == 0:
        # ensure no utt_id overlap across splits
        all_utts = {}
        for sp in args.splits:
            rows, _, _, _ = load_manifest(sp)
            for r in rows:
                u = r["utt_id"]
                all_utts.setdefault(u, []).append(sp)
        dups = {u:sps for u,sps in all_utts.items() if len(sps)>1}
        if dups:
            print("\n cross-split overlap detected:", list(dups.items())[:10])
            rc = 1
        else:
            print("\n no cross-split overlap")
    sys.exit(rc)

if __name__ == "__main__":
    main()