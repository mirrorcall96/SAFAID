#!/usr/bin/env python3
"""Extract ONLY the manifest-listed members from a local progan_train.zip / progan_val.zip
(no extractall). Verifies size, CRC32 (zipfile does this on read) and SHA256 from the manifest.
usage: python extract_subset_from_local_zip.py progan_train.zip safaid_train_subset_v2_seed0_sha256.csv out/ [--source progan_train]
Note: the HF release ships the train zip inside a store-mode 7z split (progan_train.7z.001-007).
      `cat progan_train.7z.00* > t.7z && 7z x t.7z` yields progan_train.zip (74,924,002,746 bytes)."""
import csv, hashlib, os, sys, zipfile
zpath, man, out = sys.argv[1:4]
src = sys.argv[5] if len(sys.argv) > 5 else None
rows = [r for r in csv.DictReader(open(man)) if src is None or r.get("source", "progan_train") == src]
with zipfile.ZipFile(zpath) as z:
    for r in rows:
        data = z.read(r["name"])                                 # random access via central directory
        assert len(data) == int(r["size"]), r["name"]
        if "sha256" in r: assert hashlib.sha256(data).hexdigest() == r["sha256"], r["name"]
        dst = os.path.join(out, r["split"], r["category"], r["label"], os.path.basename(r["name"]))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f: f.write(data)
print(f"extracted {len(rows)} files to {out}")
