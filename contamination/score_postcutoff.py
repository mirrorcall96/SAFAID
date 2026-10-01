#!/usr/bin/env python3
"""Aggregate the post-cutoff contamination test over models into RESULTS_postcutoff.md (+ JSON).

Input: one eval_post.py run dir per model (preds_<tag>.csv + metrics_<tag>.json). For every model:
  * answers are RE-PARSED from the response column with "first verdict wins": whichever of the fake
    keywords ("ai-generated", "ai generated", "fake") or the real keyword ("real") occurs first decides;
    no keyword = unknown = error. eval.py's parse (fake keywords checked first) is kept for comparison,
    and the number of answers where the two parses disagree is reported.
  * the share of responses that are not exactly "This is a real image." / "This is an AI-generated image."
  * per generator: balanced accuracy (real_acc + fake_acc) / 2, real_acc, fake_acc (the 1,000 reals are
    shared by all generators, so plain accuracy would weight them 3x); per-group mean of the per-generator
    numbers. Group PostCutoff (FLUX.2-klein-4B, ERNIE-Image-Turbo) and group PostCutoffModelPreCutoffVAE
    (Z-Image-Turbo) are NEVER averaged together.
Across models: SAFAID mean +- sample sd over its seeds vs the CatAID-prompt control mean over its seeds
(family and seed are read from the run-dir name: "safaid" / "catAIDprompt", "seed<k>"). For context, the
same models' fake_acc on ForenSynths (11 Table-1 rows) and UniverDiffu (8 rows) is read from each model's
own Table-1 metrics_<prompt>.json (next to its checkpoint, or under --context_root/<run name>/).

    python contamination/score_postcutoff.py --runs_root $SAFAID_RUNS/postcutoff \
        --context_root $SAFAID_WORKDIR/runs_prev --context_root $SAFAID_RUNS \
        --attribution .../attribution.csv --generation_meta .../generation_meta.json \
        --out_md .../RESULTS_postcutoff.md --out_json .../results_postcutoff.json
"""
import argparse
import collections
import csv
import datetime as dt
import glob
import json
import math
import os
import re
import statistics

GROUPS = ["PostCutoff", "PostCutoffModelPreCutoffVAE"]
GROUP_DESC = {"PostCutoff": "post-cutoff model + post-cutoff decoder (FLUX.2 VAE, shared)",
              "PostCutoffModelPreCutoffVAE": "post-cutoff model, pre-cutoff decoder (FLUX.1-family VAE)"}
REAL_BENCHMARK = "PostCutoffReal"
CANONICAL = ("This is a real image.", "This is an AI-generated image.")
FS_ROWS = ["ProGAN", "CycleGAN", "BigGAN", "StyleGAN", "GauGAN", "SAN", "StarGAN", "DeepFakes", "SITD", "CRN", "IMLE"]
UD_ROWS = ["ADM", "LDM (200)", "LDM (CFG)", "LDM (100)", "Glide (100-27)", "Glide (50-27)", "Glide (100-10)", "DALL-E"]
FAKE_KEYS = ("ai-generated", "ai generated", "fake")


def parse_evalpy(response):  # identical to eval.py parse()
    r = response.lower()
    if "ai-generated" in r or "ai generated" in r or "fake" in r:
        return "fake"
    elif "real" in r:
        return "real"
    return "unknown"


def parse_first(response):
    """First verdict wins: the earliest fake keyword vs the earliest 'real'."""
    r = response.lower()
    pf = min((i for i in (r.find(k) for k in FAKE_KEYS) if i >= 0), default=-1)
    pr = r.find("real")
    if pf < 0 and pr < 0:
        return "unknown"
    if pr < 0 or (0 <= pf < pr):
        return "fake"
    return "real"


def family_of(name):
    n = name.lower()
    if "catai" in n:
        return "control"
    if "safaid" in n:
        return "safaid"
    return "other"


def seed_of(name):
    m = re.search(r"seed(\d+)", name)
    return int(m[1]) if m else None


def rate(k, n):
    return 100.0 * k / n if n else float("nan")


def score_model(preds_path, parser):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(preds_path, newline="")):
        by[(r["benchmark"], r["table1_row"])].append(r)
    reals = [r for (b, _), v in by.items() if b == REAL_BENCHMARK for r in v if r["label"] == "0_real"]
    def pred(r):
        return "unknown" if r["response"] == "<load error>" else parser(r["response"])
    nr = len(reals)
    tn = sum(pred(r) == "real" for r in reals)
    out = {"n_real": nr, "real_acc": rate(tn, nr), "real_unknown": sum(pred(r) == "unknown" for r in reals),
           "generators": {}, "groups": {}}
    for (b, row), v in sorted(by.items()):
        if b not in GROUPS:
            continue
        fk = [r for r in v if r["label"] == "1_fake"]
        nf = len(fk)
        tp = sum(pred(r) == "fake" for r in fk)
        pr_, pf_ = tn / max(nr, 1), tp / max(nf, 1)
        se = 50 * math.sqrt(pr_ * (1 - pr_) / max(nr, 1) + pf_ * (1 - pf_) / max(nf, 1))
        out["generators"][row] = {"group": b, "n_fake": nf, "fake_acc": rate(tp, nf), "real_acc": out["real_acc"],
                                  "balanced_acc": 50 * (pr_ + pf_), "balanced_ci95": 1.96 * se,
                                  "acc": rate(tp + tn, nf + nr), "fake_unknown": sum(pred(r) == "unknown" for r in fk)}
    for g in GROUPS:
        gv = [v for v in out["generators"].values() if v["group"] == g]
        if gv:
            out["groups"][g] = {"n_generators": len(gv), "balanced_acc": statistics.fmean(v["balanced_acc"] for v in gv),
                                "fake_acc": statistics.fmean(v["fake_acc"] for v in gv), "real_acc": out["real_acc"]}
    return out, by


def response_format(by):
    out = {}
    allr = []
    for (b, row), v in sorted(by.items()):
        key = "real pool" if b == REAL_BENCHMARK else row
        k = sum(r["response"].strip() not in CANONICAL for r in v)
        dis = sum((("unknown" if r["response"] == "<load error>" else parse_first(r["response"])) != r["pred"]) for r in v)
        out[key] = {"n": len(v), "noncanonical": k, "share_noncanonical": rate(k, len(v)), "parse_disagreements": dis}
        allr += v
    k = sum(r["response"].strip() not in CANONICAL for r in allr)
    dis = sum(x["parse_disagreements"] for x in out.values())
    out["all"] = {"n": len(allr), "noncanonical": k, "share_noncanonical": rate(k, len(allr)), "parse_disagreements": dis}
    examples = collections.Counter(r["response"].strip() for r in allr if r["response"].strip() not in CANONICAL)
    out["top_noncanonical_examples"] = [{"response": t[:160], "count": c} for t, c in examples.most_common(5)]
    return out


def context_metrics(m_post, run_name, context_roots):
    prompt = m_post.get("test_prompt_name") or ("vanilla" if m_post.get("test_prompt", "").startswith("Is this") else "investigative")
    cands = []
    mp = m_post.get("model_path") or ""
    if mp:
        cands.append(os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(mp))), f"metrics_{prompt}.json"))
    cands += [os.path.join(r, run_name, f"metrics_{prompt}.json") for r in context_roots]
    for c in cands:
        if os.path.exists(c):
            rows = json.load(open(c)).get("rows", {})
            out = {"source": c}
            for name, keys in (("ForenSynths", FS_ROWS), ("UniverDiffu", UD_ROWS)):
                have = [rows[k] for k in keys if k in rows]
                if len(have) == len(keys):
                    out[name] = {"fake_acc": statistics.fmean(v["fake_acc"] for v in have),
                                 "real_acc": statistics.fmean(v["real_acc"] for v in have),
                                 "acc": statistics.fmean(v["acc"] for v in have), "n_rows": len(keys)}
            return out
    return {"source": None, "tried": cands}


def agg(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "values": []}
    return {"n": len(vals), "mean": statistics.fmean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else None,
            "values": vals}


def fmt_sd(a):
    if not a["n"]:
        return "n/a"
    return f"{a['mean']:.1f} ± {a['sd']:.1f}" if a["sd"] is not None else f"{a['mean']:.1f}"


def fmt_vals(a):
    if not a["n"]:
        return "n/a"
    return f"{a['mean']:.1f} [{', '.join(f'{v:.1f}' for v in a['values'])}]"


def fmt_diff(a, b):
    return f"{a['mean'] - b['mean']:+.1f}" if a["n"] and b["n"] else "n/a"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_root", help="directory whose subdirectories are eval_post.py run dirs")
    p.add_argument("--run_dirs", nargs="*", default=[])
    p.add_argument("--context_root", action="append", default=[], help="where <run name>/metrics_<prompt>.json lives")
    p.add_argument("--attribution", help="attribution.csv of the reals (summarised in the header)")
    p.add_argument("--generation_meta", help="generation_meta.json (summarised in the header)")
    p.add_argument("--expected_models", type=int, default=5)
    p.add_argument("--out_md", required=True)
    p.add_argument("--out_json")
    a = p.parse_args()

    dirs = list(a.run_dirs)
    if a.runs_root:
        dirs += sorted(d for d in glob.glob(os.path.join(a.runs_root, "*")) if os.path.isdir(d))
    models, skipped = [], []
    for d in dirs:
        mets = sorted(f for f in glob.glob(os.path.join(d, "metrics_*.json")) if "_smoke" not in f)
        if not mets:
            skipped.append(os.path.basename(os.path.normpath(d)))
            continue
        mpath = mets[0]
        tag = os.path.basename(mpath)[len("metrics_"):-len(".json")]
        preds = os.path.join(d, f"preds_{tag}.csv")
        if not os.path.exists(preds):
            skipped.append(os.path.basename(os.path.normpath(d)))
            continue
        name = os.path.basename(os.path.normpath(d))
        m_post = json.load(open(mpath))
        fv, by = score_model(preds, parse_first)
        ev, _ = score_model(preds, parse_evalpy)
        models.append({"name": name, "family": family_of(name), "seed": seed_of(name), "tag": tag,
                       "complete": os.path.exists(os.path.join(d, f"EVAL_DONE_{tag}")),
                       "test_prompt": m_post.get("test_prompt_name"), "model_path": m_post.get("model_path"),
                       "inference_code_check": m_post.get("inference_code_check"),
                       "first_verdict": fv, "evalpy_parse": ev, "response_format": response_format(by),
                       "context": context_metrics(m_post, name, a.context_root)})
    if not models:
        raise SystemExit(f"no eval_post.py run dirs with metrics under {dirs}")
    gens = collections.OrderedDict()
    for g in GROUPS:
        for m in models:
            for row, v in m["first_verdict"]["generators"].items():
                if v["group"] == g:
                    gens.setdefault(row, g)
    fam = {f: sorted([m for m in models if m["family"] == f], key=lambda m: (m["seed"] or 0)) for f in ("safaid", "control")}
    FORDER = {"safaid": 0, "control": 1, "other": 2}
    by_family = sorted(models, key=lambda m: (FORDER[m["family"]], m["seed"] or 0))

    def short(name):
        return name.split("_qwen3vl")[0]

    def A(f, getter):
        return agg([getter(m) for m in fam[f]])

    def G(row, key):
        return lambda m: m["first_verdict"]["generators"].get(row, {}).get(key)

    def GR(g, key):
        return lambda m: m["first_verdict"]["groups"].get(g, {}).get(key)

    L = []
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L.append("# Post-cutoff contamination test: results\n")
    L.append(f"Generated {now} by score_postcutoff.py from {len(models)} model run(s) "
             f"(SAFAID seeds {[m['seed'] for m in fam['safaid']]}, CatAID-prompt control seeds "
             f"{[m['seed'] for m in fam['control']]}).\n")
    n_real = max(m["first_verdict"]["n_real"] for m in models)
    nf = {row: max(m["first_verdict"]["generators"].get(row, {}).get("n_fake", 0) for m in models) for row in gens}
    L.append("**Setup.** Detector backbone Qwen3-VL-4B-Instruct was released 2025-10-15. "
             f"Reals: {n_real} Wikimedia Commons photographs taken on or after 2025-11-01, shared by every generator. "
             "Fakes: " + ", ".join(f"{row} ({nf[row]})" for row in gens) + ". "
             "Both classes: 1024x1024 native square (reals: center crop at native resolution; fakes: generated at 1024), "
             "then one Lanczos 2x downscale to 512, PNG without metadata.\n")
    L.append("**Groups (never averaged together).** " + "; ".join(
        f"`{g}` = {', '.join(r for r, gg in gens.items() if gg == g) or 'none'}: {GROUP_DESC[g]}" for g in GROUPS) + ".\n")
    L.append("**Metric.** Balanced accuracy = (real_acc + fake_acc) / 2, in %. Answers re-parsed from the response text "
             "with *first verdict wins*; an answer with no verdict counts as an error. SAFAID: mean ± sample sd over seeds; "
             "control: mean [per-seed values].\n")
    if a.attribution and os.path.exists(a.attribution):
        at = list(csv.DictReader(open(a.attribution, newline="")))
        if at:
            dates = sorted(r["exif_datetime_original"][:10].replace(":", "-") for r in at)
            lic = collections.Counter(r["license_short"] for r in at).most_common(4)
            cams = len({(r["camera_make"], r["camera_model"]) for r in at})
            L.append(f"**Reals (attribution.csv).** {len(at)} files, capture dates {dates[0]} .. {dates[-1]}, "
                     f"{len({r['uploader'] for r in at})} uploaders, {cams} camera models; licenses "
                     + ", ".join(f"{k} {v}" for k, v in lic) + ".\n")
    if a.generation_meta and os.path.exists(a.generation_meta):
        gm = json.load(open(a.generation_meta))
        parts = []
        for k, v in (gm.get("generators") or {}).items():
            cal = v.get("calibration") or {}
            parts.append(f"{v.get('display', k)}: {v.get('repo')}@{str(v.get('revision'))[:8]}, {v.get('call')}, "
                         f"VAE sha256 {str(v.get('vae_sha256', ''))[:12]}, {cal.get('steady_sec_per_img', '?')} s/img")
        if parts:
            L.append("**Generators.** " + "; ".join(parts) + ".\n")
    missing = a.expected_models - len(models)
    incomplete = [m["name"] for m in models if not m["complete"]]
    if missing > 0 or skipped or incomplete:
        L.append(f"**Warning.** {max(missing, 0)} of {a.expected_models} expected models have no results"
                 + (f" (no metrics in: {', '.join(skipped)})" if skipped else "")
                 + (f"; incomplete runs: {', '.join(incomplete)}" if incomplete else "") + ".\n")

    L.append("## 1. Balanced accuracy (%)\n")
    L.append("| Set | Group | SAFAID (mean ± sd) | CatAID-prompt control (mean [seeds]) | SAFAID − control |")
    L.append("|---|---|---|---|---|")
    for g in GROUPS:
        rows_g = [r for r, gg in gens.items() if gg == g]
        for row in rows_g:
            s, c = A("safaid", G(row, "balanced_acc")), A("control", G(row, "balanced_acc"))
            L.append(f"| {row} | {g} | {fmt_sd(s)} | {fmt_vals(c)} | {fmt_diff(s, c)} |")
        if len(rows_g) > 1:
            s, c = A("safaid", GR(g, "balanced_acc")), A("control", GR(g, "balanced_acc"))
            L.append(f"| **{g} mean** ({', '.join(rows_g)}) | {g} | **{fmt_sd(s)}** | **{fmt_vals(c)}** | {fmt_diff(s, c)} |")
    L.append("")

    L.append("## 2. Real and fake accuracy (%)\n")
    L.append("| Set | Group | SAFAID real_acc | SAFAID fake_acc | control real_acc | control fake_acc |")
    L.append("|---|---|---|---|---|---|")
    s, c = A("safaid", lambda m: m["first_verdict"]["real_acc"]), A("control", lambda m: m["first_verdict"]["real_acc"])
    L.append(f"| Real pool (n={n_real}, shared) | {REAL_BENCHMARK} | {fmt_sd(s)} | – | {fmt_vals(c)} | – |")
    for g in GROUPS:
        for row in [r for r, gg in gens.items() if gg == g]:
            s, c = A("safaid", G(row, "fake_acc")), A("control", G(row, "fake_acc"))
            L.append(f"| {row} | {g} | – | {fmt_sd(s)} | – | {fmt_vals(c)} |")
        if sum(1 for gg in gens.values() if gg == g) > 1:
            s, c = A("safaid", GR(g, "fake_acc")), A("control", GR(g, "fake_acc"))
            L.append(f"| **{g} mean** | {g} | – | **{fmt_sd(s)}** | – | **{fmt_vals(c)}** |")
    L.append("")

    L.append("## 3. Context: the same models on the Table-1 benchmarks (%)\n")
    L.append("Table-1 numbers come from each model's own `metrics_<prompt>.json` (eval.py parse). "
             "ForenSynths = mean over its 11 Table-1 rows, UniverDiffu = mean over its 8 rows.\n")
    L.append("| Set | SAFAID fake_acc | control fake_acc | SAFAID real_acc | control real_acc |")
    L.append("|---|---|---|---|---|")
    for name in ("ForenSynths", "UniverDiffu"):
        def ctx(key, name=name):
            return lambda m: (m["context"].get(name) or {}).get(key)
        L.append(f"| {name} (Table 1) | {fmt_sd(A('safaid', ctx('fake_acc')))} | {fmt_vals(A('control', ctx('fake_acc')))} | "
                 f"{fmt_sd(A('safaid', ctx('real_acc')))} | {fmt_vals(A('control', ctx('real_acc')))} |")
    for g in GROUPS:
        if any(gg == g for gg in gens.values()):
            L.append(f"| {g} (post-cutoff, this test) | {fmt_sd(A('safaid', GR(g, 'fake_acc')))} | "
                     f"{fmt_vals(A('control', GR(g, 'fake_acc')))} | {fmt_sd(A('safaid', GR(g, 'real_acc')))} | "
                     f"{fmt_vals(A('control', GR(g, 'real_acc')))} |")
    no_ctx = [short(m["name"]) for m in by_family if not m["context"].get("source")]
    part_ctx = [short(m["name"]) for m in by_family if m["context"].get("source")
                and not (m["context"].get("ForenSynths") and m["context"].get("UniverDiffu"))]
    if no_ctx:
        L.append(f"\nNo Table-1 metrics found for: {', '.join(no_ctx)}.")
    if part_ctx:
        L.append(f"\nIncomplete Table-1 metrics (not all rows present) for: {', '.join(part_ctx)}.")
    L.append("")

    L.append("## 4. Per model\n")
    head = ["Model", "Family", "Seed", "Prompt", "real_acc"]
    for row in gens:
        head += [f"{row} bal. (±95% CI)", f"{row} fake"]
    head += [f"{g} mean bal." for g in GROUPS if sum(1 for gg in gens.values() if gg == g) > 1]
    head += ["non-canonical %", "parse disagreements", "code check"]
    L.append("| " + " | ".join(head) + " |")
    L.append("|" + "---|" * len(head))
    for m in by_family:
        fvm = m["first_verdict"]
        cells = [short(m["name"]), m["family"], str(m["seed"]), str(m["test_prompt"]), f"{fvm['real_acc']:.1f}"]
        for row in gens:
            v = fvm["generators"].get(row)
            cells += [f"{v['balanced_acc']:.1f} ± {v['balanced_ci95']:.1f}", f"{v['fake_acc']:.1f}"] if v else ["n/a", "n/a"]
        for g in GROUPS:
            if sum(1 for gg in gens.values() if gg == g) > 1:
                v = fvm["groups"].get(g)
                cells.append(f"{v['balanced_acc']:.1f}" if v else "n/a")
        rf = m["response_format"]["all"]
        cells += [f"{rf['share_noncanonical']:.2f}", str(rf["parse_disagreements"]),
                  "identical" if m["inference_code_check"] == "IDENTICAL" else str(m["inference_code_check"])]
        L.append("| " + " | ".join(cells) + " |")
    L.append("\n±95% CI: 1.96 × binomial standard error of the balanced accuracy within one model (image sampling only).\n")

    L.append("## 5. Response format and parsing\n")
    L.append("Share of responses that are not exactly `This is a real image.` or `This is an AI-generated image.`, and the "
             "number of answers where *first verdict wins* and eval.py's parse (fake keywords first) disagree.\n")
    sets = ["real pool"] + list(gens)
    L.append("| Model | " + " | ".join(f"{s} non-canon. %" for s in sets) + " | all non-canon. % | disagreements |")
    L.append("|" + "---|" * (len(sets) + 3))
    for m in by_family:
        rf = m["response_format"]
        L.append(f"| {short(m['name'])} | " + " | ".join(f"{rf[s]['share_noncanonical']:.2f}" if s in rf else "n/a" for s in sets)
                 + f" | {rf['all']['share_noncanonical']:.2f} | {rf['all']['parse_disagreements']} |")
    ex = collections.Counter()
    for m in models:
        for e in m["response_format"]["top_noncanonical_examples"]:
            ex[e["response"]] += e["count"]
    if ex:
        L.append("\nMost frequent non-canonical responses (all models): " +
                 "; ".join(f"`{t}` ×{c}" for t, c in ex.most_common(5)))
    ev_parts = []
    for g in GROUPS:
        if any(gg == g for gg in gens.values()):
            s_ev = A("safaid", lambda m, g=g: m["evalpy_parse"]["groups"].get(g, {}).get("balanced_acc"))
            c_ev = A("control", lambda m, g=g: m["evalpy_parse"]["groups"].get(g, {}).get("balanced_acc"))
            ev_parts.append(f"`{g}` SAFAID {fmt_sd(s_ev)}, control {fmt_vals(c_ev)}")
    L.append("\nWith eval.py's parse instead, the group mean balanced accuracy is: " + "; ".join(ev_parts) + ".\n")

    L.append("## Notes\n")
    L.append("- Reals are shared: all generators within one model use the same real_acc, so differences between "
             "generators are differences in fake_acc.")
    L.append("- The two groups answer different questions: `PostCutoff` removes both the generator and its decoder from "
             "the pretraining window; `PostCutoffModelPreCutoffVAE` keeps a decoder whose outputs could have been seen "
             "in pretraining. They are therefore not pooled.")
    L.append("- Wikimedia reals are camera JPEGs; fakes were never JPEG-compressed. Both went through the same crop/resize "
             "and PNG step, but compression history differs between the classes.")
    open(a.out_md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    if a.out_json:
        summary = {"generated_utc": now, "groups": GROUPS, "generators": gens, "models": models, "skipped": skipped,
                   "aggregate": {}}
        for f in ("safaid", "control"):
            d = {}
            for row in gens:
                for k in ("balanced_acc", "fake_acc", "real_acc"):
                    d[f"{row}|{k}"] = A(f, G(row, k))
            for g in GROUPS:
                for k in ("balanced_acc", "fake_acc", "real_acc"):
                    d[f"{g}|{k}"] = A(f, GR(g, k))
            summary["aggregate"][f] = d
        json.dump(summary, open(a.out_json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
