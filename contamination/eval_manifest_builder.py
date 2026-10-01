#!/usr/bin/env python3
"""Build the eval manifest for the post-cutoff contamination test (and generation_meta.json).

Same columns as manifests/eval_members_manifest.csv:
    benchmark,table1_row,folder,subclass,label,zip,member,size,crc32

The two generator groups get DIFFERENT benchmark labels, so no script can average them by accident;
the shared reals get a third label:
  benchmark=PostCutoff                   FLUX.2-klein-4B, ERNIE-Image-Turbo (post-cutoff model + FLUX.2 VAE)
  benchmark=PostCutoffModelPreCutoffVAE  Z-Image-Turbo (post-cutoff model, FLUX.1-family VAE)
  benchmark=PostCutoffReal               Wikimedia Commons photos taken >= 2025-11-01, table1_row "real pool post-cutoff"
  fakes: table1_row=<generator display name>, folder=<generator key>, subclass=<prompt category>, label=1_fake
  zip="postcutoff", member=<folder>/<label>/<file>; eval_post.py maps zip "postcutoff" to <data_root>/postcutoff.

--verify re-checks every PNG: crc32 against metadata.jsonl / attribution.csv, 512x512, and no ancillary
PNG chunks (only IHDR/IDAT/IEND). Files that fail are left out of the manifest and listed.

    python eval_manifest_builder.py --root DATA/postcutoff --out DATA/postcutoff/postcutoff_manifest.csv \
        --verify --meta_out RUNS/postcutoff/generation_meta.json
"""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from postcutoff_common import CROP, OUT_RES, file_digests, png_info  # noqa: E402

HEADER = ["benchmark", "table1_row", "folder", "subclass", "label", "zip", "member", "size", "crc32"]
REAL_BENCHMARK = "PostCutoffReal"
REAL_ROW = "real pool post-cutoff"
# fallback if a generator folder has no run_info.json (gen_fakes.py writes "benchmark" there)
GROUP_OF = {"flux2_klein_4b": "PostCutoff", "ernie_image_turbo": "PostCutoff",
            "z_image_turbo": "PostCutoffModelPreCutoffVAE"}
DISPLAY_OF = {"flux2_klein_4b": "FLUX.2-klein-4B", "ernie_image_turbo": "ERNIE-Image-Turbo",
              "z_image_turbo": "Z-Image-Turbo"}
GROUPS = ["PostCutoff", "PostCutoffModelPreCutoffVAE"]


def load_jsonl(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                out[r["file"]] = r
            except (json.JSONDecodeError, KeyError):
                pass
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="the --out of gen_fakes.py / fetch_recent_reals.py")
    p.add_argument("--out", required=True, help="manifest CSV to write")
    p.add_argument("--zip", default="postcutoff")
    p.add_argument("--reals_subdir", default="wikimedia_2026")
    p.add_argument("--out_res", type=int, default=OUT_RES, help="expected PNG size (tests only)")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--expect_n", type=int, default=0, help="warn if a generator or the real pool has fewer")
    p.add_argument("--meta_out", help="also write generation_meta.json here")
    a = p.parse_args()

    rows, problems, gen_info = [], [], {}
    for folder in sorted(os.listdir(a.root)):
        d = os.path.join(a.root, folder)
        if not os.path.isdir(d) or folder.startswith(("_", ".")) or folder.endswith("_dryrun"):
            continue
        for label in ("1_fake", "0_real"):
            ld = os.path.join(d, label)
            if not os.path.isdir(ld):
                continue
            files = sorted(f for f in os.listdir(ld) if f.lower().endswith(".png"))
            if label == "1_fake":
                info_p = os.path.join(d, "run_info.json")
                info = json.load(open(info_p)) if os.path.exists(info_p) else {}
                bench = info.get("benchmark") or GROUP_OF.get(folder)
                if bench not in GROUPS:
                    print(f"SKIP {folder}: unknown generator group {bench!r} (not in {GROUPS})")
                    continue
                row_name = info.get("display") or DISPLAY_OF.get(folder, folder)
                meta = load_jsonl(os.path.join(d, "metadata.jsonl"))
                gen_info[folder] = info
            else:
                if folder != a.reals_subdir:
                    print(f"SKIP {folder}/0_real: reals are only taken from {a.reals_subdir}/")
                    continue
                bench, row_name, meta = REAL_BENCHMARK, REAL_ROW, {}
                ap = os.path.join(d, "attribution.csv")
                if os.path.exists(ap):
                    meta = {r["file"]: {"crc32": r["out_crc32"]} for r in csv.DictReader(open(ap, newline=""))}
            for f in files:
                path = os.path.join(ld, f)
                size, crc, _ = file_digests(path)
                m = meta.get(f)
                if a.verify:
                    why = None
                    if m is None:
                        why = "no metadata record"
                    elif m.get("crc32") != crc:
                        why = "crc mismatch"
                    else:
                        w, h, chunks = png_info(path)
                        if (w, h) != (a.out_res, a.out_res):
                            why = f"size {w}x{h}"
                        elif chunks != ["IHDR", "IDAT", "IEND"]:
                            why = f"PNG chunks {chunks}"
                    if why:
                        problems.append(f"{folder}/{label}/{f}: {why}")
                        continue
                rows.append({"benchmark": bench, "table1_row": row_name, "folder": folder,
                             "subclass": (m or {}).get("category", "") if label == "1_fake" else "",
                             "label": label, "zip": a.zip, "member": f"{folder}/{label}/{f}",
                             "size": size, "crc32": crc})
    n_real = sum(r["label"] == "0_real" for r in rows)
    n_fake = sum(r["label"] == "1_fake" for r in rows)
    if problems:
        print(f"VERIFY: {len(problems)} file(s) left out, e.g. {problems[:5]}")
    if not n_real or not n_fake:
        sys.exit(f"need both classes: {n_real} reals, {n_fake} fakes under {a.root}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    c = collections.Counter((r["benchmark"], r["table1_row"], r["label"]) for r in rows)
    for (bench, row, label), n in sorted(c.items()):
        flag = "  <-- fewer than expected" if a.expect_n and n < a.expect_n else ""
        print(f"{n:6d}  {label}  {bench:28s} {row}{flag}")
    print(f"wrote {len(rows)} rows ({n_fake} fakes, {n_real} reals) to {a.out} (zip={a.zip!r})")

    if a.meta_out:
        reals_info_p = os.path.join(a.root, a.reals_subdir, "run_info.json")
        meta = {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "root": os.path.abspath(a.root), "manifest": os.path.abspath(a.out),
            "manifest_sha256": hashlib.sha256(open(a.out, "rb").read()).hexdigest(),
            "cutoff": {"model": "Qwen3-VL-4B-Instruct", "released": "2025-10-15", "reals_taken_from": "2025-11-01"},
            "processing": {"reals": f"EXIF orientation, sRGB, center crop {CROP}x{CROP} at native resolution "
                                    f"(shortest side >= {CROP}), Lanczos {CROP}->{OUT_RES}",
                           "fakes": f"generated at {CROP}x{CROP}, Lanczos {CROP}->{OUT_RES}",
                           "scale_factor": CROP / OUT_RES, "format": "PNG, no ancillary chunks"},
            "groups": {g: sorted({r["table1_row"] for r in rows if r["benchmark"] == g}) for g in GROUPS},
            "group_note": "groups are reported separately and never averaged together",
            "counts": {f"{b} | {row} | {lab}": n for (b, row, lab), n in sorted(c.items())},
            "verify": {"enabled": a.verify, "left_out": problems},
            "generators": {k: {kk: vv for kk, vv in v.items() if kk not in ("host", "weights_path")}
                           for k, v in gen_info.items()},
            "reals": json.load(open(reals_info_p)) if os.path.exists(reals_info_p) else None,
        }
        os.makedirs(os.path.dirname(os.path.abspath(a.meta_out)), exist_ok=True)
        json.dump(meta, open(a.meta_out, "w"), indent=1, ensure_ascii=False)
        print(f"wrote {a.meta_out}")


if __name__ == "__main__":
    main()
