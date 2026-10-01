#!/usr/bin/env python3
"""Fetch the data for the revision runs onto the instance.
  1. ProGAN train subset (4,000 imgs): HTTP range reads from HF sywang/CNNDetection@f287cf3c (no 75 GB download),
     CRC32 + SHA256 checked against manifest/safaid_train_subset_v2_seed0_sha256.csv
  2. ForenSynths test (CNN_synth_testset.zip, 20 GB, HF, same file as the Drive copy): only Table 1 members extracted
  3. UniverDiffu (diffusion_datasets.zip, Ojha et al., Google Drive): fully extracted
"""
import os, sys, csv, struct, zlib, hashlib, time, zipfile, subprocess
from concurrent.futures import ThreadPoolExecutor

# ---- paths (release) ----
# Defaults reproduce the layout of the paper runs (vast.ai, /workspace/safaid); override with env vars or CLI flags.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKDIR = os.environ.get("SAFAID_WORKDIR", "/workspace/safaid")
DATA = os.environ.get("SAFAID_DATA", os.path.join(WORKDIR, "data"))
MANIFESTS = os.environ.get("SAFAID_MANIFESTS", os.path.join(REPO, "manifests"))
sys.path.insert(0, os.path.join(REPO, "manifests", "scripts"))
from multipart import MultiRange, ZIP_START  # noqa: E402

HF_REV = "f287cf3c3e92ca2e5451ce200bab8b6f5e0175b2"
UD_DRIVE_ID = "1FXlGIRh_Ud3cScMgSVDbEWmPDmjcrm1t"


def fetch_train_subset():
    out_dir = os.path.join(DATA, "progan_train_subset")
    rows = list(csv.DictReader(open(os.path.join(MANIFESTS, "safaid_train_subset_v2_seed0_sha256.csv"))))
    mr = MultiRange()
    [mr._url(i) for i in range(7)]

    def get(r):
        dst = os.path.join(out_dir, r["name"])
        if os.path.exists(dst) and hashlib.sha256(open(dst, "rb").read()).hexdigest() == r["sha256"]:
            return 0
        off, cs = int(r["offset"]), int(r["csize"])
        h = mr._raw(ZIP_START + off, ZIP_START + off + 29)
        n, x = struct.unpack("<HH", h[26:30])
        raw = mr._raw(ZIP_START + off + 30 + n + x, ZIP_START + off + 30 + n + x + cs - 1)
        data = zlib.decompress(raw, -15) if int(r["method"]) == 8 else raw
        assert format(zlib.crc32(data), "08x") == r["crc"], r["name"]
        assert hashlib.sha256(data).hexdigest() == r["sha256"], r["name"]
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "wb").write(data)
        return 1

    t = time.time()
    with ThreadPoolExecutor(48) as ex:
        n = sum(ex.map(get, rows))
    print(f"train subset: {len(rows)} verified ({n} downloaded) in {time.time() - t:.0f}s", flush=True)


def fetch_testset():
    out_dir = os.path.join(DATA, "forensynths")
    want = {}
    for r in csv.DictReader(open(os.path.join(MANIFESTS, "eval_members_manifest.csv"))):
        if r["zip"] == "CNN_synth_testset.zip" and r["table1_row"] != "(not in Table 1)":
            want[r["member"]] = r
    if all(os.path.exists(os.path.join(out_dir, m)) for m in want):
        print("forensynths test: already extracted", flush=True)
        return
    from huggingface_hub import hf_hub_download
    t = time.time()
    z = hf_hub_download("sywang/CNNDetection", "CNN_synth_testset.zip", repo_type="dataset", revision=HF_REV,
                        local_dir=os.path.join(DATA, "_dl"))
    print(f"downloaded {z} in {time.time() - t:.0f}s", flush=True)
    with zipfile.ZipFile(z) as zf:
        names = [i for i in zf.infolist() if i.filename in want]
        assert len(names) == len(want), (len(names), len(want))
        for i in names:
            zf.extract(i, out_dir)
    os.remove(z)
    print(f"forensynths test: extracted {len(want)} Table 1 images", flush=True)


def fetch_univerdiffu():
    want = [r["member"] for r in csv.DictReader(open(os.path.join(MANIFESTS, "eval_members_manifest.csv")))
            if r["zip"] == "diffusion_datasets.zip"]
    if all(os.path.exists(os.path.join(DATA, m)) for m in want):
        print("univerdiffu: already extracted", flush=True)
        return
    z = os.path.join(DATA, "_dl", "diffusion_datasets.zip")
    os.makedirs(os.path.dirname(z), exist_ok=True)
    if not os.path.exists(z):
        subprocess.check_call([sys.executable, "-m", "gdown", UD_DRIVE_ID, "-O", z])
    with zipfile.ZipFile(z) as zf:
        zf.extractall(DATA)
    missing = [m for m in want if not os.path.exists(os.path.join(DATA, m))]
    assert not missing, (len(missing), missing[:3])
    os.remove(z)
    print(f"univerdiffu: {len(want)} images ok", flush=True)


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    fetch_train_subset()
    fetch_univerdiffu()
    fetch_testset()
    open(os.path.join(DATA, "DATA_READY"), "w").write("ok\n")
    print("DATA_READY", flush=True)
