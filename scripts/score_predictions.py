#!/usr/bin/env python3
"""Score any detector's per-image predictions with the SAFAID protocol (stdlib only).

usage: python scripts/score_predictions.py preds.csv [--json metrics.json]

preds.csv needs the columns  table1_row, label, path, pred  (extra columns are ignored), one row per image of
manifests/eval_members_manifest.csv that is in Table 1 (82,353 rows):
  table1_row  the manifest's table1_row (UniverDiffu reals: "real pool for ADM only" or
              "real pool for all LDM/Glide/DALL-E rows")
  label       0_real | 1_fake     (as in the manifest)
  path        the manifest's member path (a prefix such as <data_root>/forensynths/ is allowed)
  pred        real | fake | unknown   (unknown = unparseable answer, scored as an error)
This is the format safaid/eval.py writes (preds_<prompt>.csv), so its files can be re-scored as they are.
Accuracy per row = (TP + TN) / (N_real + N_fake); each UniverDiffu generator is scored on its 1,000 fakes plus its
official 1,000-image real pool; Average (19) is the unweighted mean of the 19 row accuracies.
"""
import argparse, csv, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_results import ORDER, N_FS, recompute  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFESTS = os.environ.get("SAFAID_MANIFESTS", os.path.join(REPO, "manifests"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds")
    ap.add_argument("--manifest", default=os.path.join(MANIFESTS, "eval_members_manifest.csv"))
    ap.add_argument("--json", default=None, help="also write the per-row metrics here")
    a = ap.parse_args()

    want = {}
    for r in csv.DictReader(open(a.manifest, newline="")):
        if r["table1_row"] != "(not in Table 1)":
            want[r["member"]] = (r["table1_row"], r["label"])
    preds, seen, bad = [], set(), 0
    for p in csv.DictReader(open(a.preds, newline="", encoding="utf-8")):
        member = next((m for m in (p["path"], *(p["path"].split("/", k)[-1] for k in range(1, 6))) if m in want), None)
        if member is None or member in seen:
            bad += 1
            continue
        if (p["table1_row"], p["label"]) != want[member]:
            sys.exit(f"row/label of {p['path']} do not match the manifest: {(p['table1_row'], p['label'])} vs {want[member]}")
        if p["pred"] not in ("real", "fake", "unknown"):
            sys.exit(f"pred must be real, fake or unknown, got {p['pred']!r} for {p['path']}")
        seen.add(member)
        preds.append(p)
    missing = len(want) - len(seen)
    if missing or bad:
        sys.exit(f"{missing} manifest images have no prediction; {bad} prediction rows are not (or twice) in the manifest")

    rows, n = recompute(preds)
    print(f"{'Generator':<16}{'acc':>8}{'real':>8}{'fake':>8}{'unknown':>9}")
    for i, row in enumerate(ORDER):
        if i in (0, N_FS):
            print("-- ForenSynths" if i == 0 else "-- UniverDiffu")
        m = rows[row]
        print(f"{row:<16}{m['acc']:>8.2f}{m['real_acc']:>8.2f}{m['fake_acc']:>8.2f}{m['unknown']:>9d}")
    avg = round(sum(rows[r]["acc"] for r in ORDER) / len(ORDER), 2)
    print(f"{'Average (19)':<16}{avg:>8.2f}   ({n:,} images)")
    if a.json:
        json.dump({"rows": rows, "average_19": avg, "n_images": n}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
