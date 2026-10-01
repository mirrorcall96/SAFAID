#!/usr/bin/env python3
"""Check the data on disk against the shipped manifests (read-only, stdlib only).

  train subset : SHA256 of all 4,000 images vs manifests/safaid_train_subset_v2_seed0_sha256.csv
  eval images  : size + CRC32 of all 82,353 Table 1 images vs manifests/eval_members_manifest.csv
                 (the CRC32 values are the ones stored in the original zip central directories)

usage: python scripts/verify_data.py [--data $SAFAID_DATA] [--skip_train] [--skip_eval]
Uses the same layout as safaid/fetch_data.py: <data>/progan_train_subset/<name>, <data>/forensynths/<member>
(CNN_synth_testset.zip) and <data>/<member> (diffusion_datasets.zip). Exit code 1 if anything is missing or differs.
"""
import argparse, csv, hashlib, os, sys, zlib
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKDIR = os.environ.get("SAFAID_WORKDIR", "/workspace/safaid")
DATA = os.environ.get("SAFAID_DATA", os.path.join(WORKDIR, "data"))
MANIFESTS = os.environ.get("SAFAID_MANIFESTS", os.path.join(REPO, "manifests"))


def read(path):
    with open(path, "rb") as f:
        return f.read()


def check_train(data):
    rows = list(csv.DictReader(open(os.path.join(MANIFESTS, "safaid_train_subset_v2_seed0_sha256.csv"))))

    def one(r):
        p = os.path.join(data, "progan_train_subset", r["name"])
        if not os.path.exists(p):
            return "missing", p
        return ("ok" if hashlib.sha256(read(p)).hexdigest() == r["sha256"] else "sha256 mismatch"), p

    return rows, list(ThreadPoolExecutor(16).map(one, rows))


def check_eval(data):
    rows = [r for r in csv.DictReader(open(os.path.join(MANIFESTS, "eval_members_manifest.csv")))
            if r["table1_row"] != "(not in Table 1)"]
    roots = {"CNN_synth_testset.zip": os.path.join(data, "forensynths"), "diffusion_datasets.zip": data}

    def one(r):
        p = os.path.join(roots[r["zip"]], r["member"])
        if not os.path.exists(p):
            return "missing", p
        b = read(p)
        if len(b) != int(r["size"]):
            return "size mismatch", p
        return ("ok" if format(zlib.crc32(b), "08x") == r["crc32"] else "crc32 mismatch"), p

    return rows, list(ThreadPoolExecutor(16).map(one, rows))


def report(name, rows, results):
    bad = [(s, p) for s, p in results if s != "ok"]
    print(f"{name}: {len(rows) - len(bad)}/{len(rows)} ok")
    for s, p in bad[:10]:
        print(f"  {s}: {p}")
    if len(bad) > 10:
        print(f"  ... and {len(bad) - 10} more")
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    a = ap.parse_args()
    ok = True
    if not a.skip_train:
        ok &= report("train subset (SHA256)", *check_train(a.data))
    if not a.skip_eval:
        ok &= report("eval images (size + CRC32)", *check_eval(a.data))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
