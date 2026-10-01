#!/usr/bin/env python3
"""Rebuild every results table of the SAFAID release from the run folders (stdlib only).

usage: python scripts/summarize_results.py [results_dir] [--out results/RESULTS.md] [--no_check_predictions]

results_dir holds one folder per run (default: results/ of this repository; on the GPU machine: $SAFAID_RUNS).
Each folder has config.json (written by safaid/train.py) and metrics_<test_prompt>.json (safaid/eval.py); the
optional trainer_state.json, train_log.txt and eval_log_*.txt add epochs, trainable parameters and timings.
The role of a run is read from its config.json, not from its name:

  SAFAID      investigative prompt, fake-only augmentation, full fine-tuning     (r11, r3, r4 = runs 1, 2, 3)
  control     CatAID category prompt for training, tested without the category   (r2, r5, r6)
  QLoRA       as SAFAID, but 4-bit QLoRA (r=16, alpha=16, LR 2e-4)               (r16, seed of SAFAID run 1)
  ablations   plain prompt (r12) / uniform augmentation (r13) / no augmentation (r14), seed of SAFAID run 1

The seed of every run is in its config.json and in the first table. The SAFAID run that shares its seed with the
ablations is the reference run and is listed first ("run 1"); the other runs follow by seed. std is the sample
standard deviation over runs (ddof=1). The SAFAID-vs-control test is Welch's two-sample t-test on the per-run
19-generator averages. Unless --no_check_predictions is given, every run's 19 row accuracies are
also recomputed from its per-image predictions (preds_*.csv in the run folder, or results/predictions.zip) and must
equal the stored metrics exactly.
"""
import argparse, collections, csv, glob, io, json, math, os, re, statistics, zipfile

ORDER = ["ProGAN", "CycleGAN", "BigGAN", "StyleGAN", "GauGAN", "SAN", "StarGAN", "DeepFakes", "SITD", "CRN", "IMLE",
         "ADM", "LDM (200)", "LDM (CFG)", "LDM (100)", "Glide (100-27)", "Glide (50-27)", "Glide (100-10)", "DALL-E"]
N_FS = 11  # the first 11 rows are ForenSynths (GANs etc.), the last 8 UniverDiffu (diffusion)
UD_REAL = {"ADM": "real pool for ADM only"}  # as in safaid/eval.py
UD_REAL_DEFAULT = "real pool for all LDM/Glide/DALL-E rows"

# Published accuracies (%), in ORDER. All from Table 1 of CatAID (Cai et al., ICCVW 2025), except GADNet
# (Li et al., 2026; UniverDiffu rows only). None = not evaluated. Averages are recomputed over the 19 rows.
BASELINES = collections.OrderedDict([
    ("CNN-aug",      [100.0, 90.88, 82.40, 93.11, 93.52, 57.04, 87.27, 62.48, 76.67, 95.28, 96.93,
                      65.20, 63.15, 62.39, 61.50, 65.36, 69.52, 66.18, 60.10]),
    ("GADNet",       [None] * 11 + [84.60, 97.10, 97.60, 97.80, 94.50, 94.20, 93.80, 84.70]),
    ("Patch",        [75.03, 68.97, 68.47, 79.16, 64.23, 75.28, 63.94, 75.54, 75.14, 72.33, 55.30,
                      67.41, 76.50, 76.10, 75.77, 74.81, 73.28, 68.52, 67.91]),
    ("Co-occur",     [97.70, 63.15, 53.75, 92.50, 51.10, 55.85, 54.70, 57.10, 63.06, 65.65, 65.80,
                      60.50, 70.70, 70.55, 71.00, 70.25, 69.60, 69.90, 67.55]),
    ("Spec",         [49.90, 99.90, 50.50, 49.90, 50.30, 48.00, 99.70, 50.10, 50.00, 50.60, 50.10,
                      50.90, 50.40, 50.40, 50.30, 51.70, 51.40, 50.40, 50.00]),
    ("CLIP-ViT",     [99.54, 93.49, 88.63, 80.75, 97.11, 61.00, 98.97, 84.50, 71.50, 69.27, 79.21,
                      71.06, 91.29, 72.02, 91.29, 89.05, 90.67, 90.08, 81.47]),
    ("NPR",          [99.99, 86.83, 87.70, 96.90, 84.49, 70.09, 99.52, 66.88, 64.44, 50.00, 50.00,
                      74.65, 90.90, 93.15, 92.20, 95.20, 95.95, 95.90, 87.95]),
    ("DIRE",         [100.0, 67.73, 64.78, 83.08, 65.30, 60.96, 100.0, 94.75, 57.62, 62.36, 62.31,
                      83.20, 82.70, 84.05, 84.25, 84.25, 87.10, 90.80, 90.25]),
    ("Flamingo",     [35.35, 38.65, 27.80, 32.80, 28.79, 33.41, 39.57, 45.14, 23.06, 32.69, 33.46,
                      19.30, 27.65, 24.55, 27.40, 24.90, 25.30, 24.10, 22.25]),
    ("DFLIP",        [99.98, 98.46, 98.35, 85.19, 99.70, 48.93, 97.92, 51.79, 65.56, 82.40, 90.13,
                      56.50, 84.50, 70.70, 84.30, 69.05, 71.70, 70.15, 86.10]),
    ("InstructBLIP", [82.79, 78.09, 81.77, 72.92, 90.11, 74.94, 89.47, 79.94, 65.00, 50.00, 50.00,
                      78.05, 82.60, 70.35, 82.65, 87.00, 86.85, 86.95, 81.70]),
    ("CatAID",       [99.86, 98.83, 97.00, 96.63, 96.24, 57.28, 99.35, 71.53, 82.22, 79.05, 79.87,
                      78.80, 92.35, 88.45, 92.50, 88.80, 89.30, 89.25, 91.35]),
])
# 19-row averages as printed in the paper (used as a transcription check of the table above)
BASELINE_AVG = {"CNN-aug": 76.26, "Patch": 71.25, "Co-occur": 66.86, "Spec": 55.50, "CLIP-ViT": 84.26, "NPR": 83.30,
                "DIRE": 79.24, "Flamingo": 29.80, "DFLIP": 79.55, "InstructBLIP": 77.43, "CatAID": 87.82}

ROLE_LABEL = {"safaid": "SAFAID", "control": "CatAID-prompt control", "qlora": "SAFAID, QLoRA",
              "abl_noprompt": "ablation: w/o investigative prompt", "abl_uniform": "ablation: uniform augmentation",
              "abl_noaug": "ablation: no augmentation"}
PROMPT_TEXT = {"investigative": "investigative", "category": "category (CatAID)", "vanilla": "plain question"}


# ---------------------------------------------------------------- loading
def role_of(cfg):
    method = str(cfg.get("method", ""))
    aug = cfg.get("aug_mode", "fake_only")  # r2-r4 predate the --aug_mode flag (always fake-only)
    if method.lower().startswith("qlora"):
        return "qlora"
    assert method.startswith("full"), method
    if cfg["train_prompt"] == "category":
        return "control"
    if cfg["train_prompt"] == "vanilla":
        return "abl_noprompt"
    return {"fake_only": "safaid", "uniform": "abl_uniform", "none": "abl_noaug"}[aug]


def load_runs(results_dir):
    runs = []
    for cfg_path in sorted(glob.glob(os.path.join(results_dir, "*", "config.json"))):
        d = os.path.dirname(cfg_path)
        metrics = [m for m in sorted(glob.glob(os.path.join(d, "metrics_*.json"))) if not m.endswith("_smoke.json")]
        if not metrics:
            continue
        assert len(metrics) == 1, metrics
        cfg = json.load(open(cfg_path))
        name = os.path.basename(d)
        m = re.match(r"r(\d+)_", name)
        r = {"dir": d, "name": name, "id": f"r{m.group(1)}" if m else name, "num": int(m.group(1)) if m else 999,
             "cfg": cfg, "seed": cfg["seed"], "role": role_of(cfg), "metrics": json.load(open(metrics[0])),
             "tag": os.path.basename(metrics[0])[len("metrics_"):-len(".json")]}
        ts = os.path.join(d, "trainer_state.json")
        if os.path.exists(ts):
            t = json.load(open(ts))
            r["epoch"], r["step"] = t.get("epoch"), t.get("global_step")
        tl = os.path.join(d, "train_log.txt")
        if os.path.exists(tl):
            mm = re.search(r"Trainable parameters = ([\d,]+) of ([\d,]+)", open(tl, errors="replace").read())
            if mm:
                r["trainable"], r["total_params"] = (int(x.replace(",", "")) for x in mm.groups())
        el = glob.glob(os.path.join(d, "eval_log_*.txt"))
        if el:
            secs = [int(x) for x in re.findall(r": \d+ images in (\d+)s", open(el[0], errors="replace").read())]
            r["eval_min"] = sum(secs) / 60 if secs else None
        runs.append(r)
    runs.sort(key=lambda r: (r["num"], r["name"]))
    return runs


# ---------------------------------------------------------------- helpers
def acc(r, row, key="acc"):
    return r["metrics"]["rows"][row][key]


def avg(r, rows=ORDER, key="acc"):
    return sum(acc(r, row, key) for row in rows) / len(rows)


def avg19(r):
    return round(avg(r), 2)  # as safaid/eval.py: mean of the rounded row accuracies, rounded


def mean_std(vals):
    vals = list(vals)
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else float("nan"))


def f2(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def f1(x):
    return "—" if x is None else f"{x:.1f}"


def ms(vals):
    m, s = mean_std(vals)
    return f"{m:.2f} ± {s:.2f}" if len(vals) > 1 else f"{m:.2f}"


def signed(x, nd=2):
    return f"{x:+.{nd}f}".replace("-", "−")


def rdiff(a, b):
    """Difference of two means as displayed (each rounded to 2 decimals first), so the table adds up."""
    return round(a, 2) - round(b, 2)


def table(head, rows, align=None):
    align = align or ["---"] + ["---:"] * (len(head) - 1)
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def section_rows(n_cols):
    return {0: ["*ForenSynths*"] + [""] * (n_cols - 1), N_FS: ["*UniverDiffu*"] + [""] * (n_cols - 1)}


# Student t / Welch test without scipy: two-sided p = I_{df/(df+t^2)}(df/2, 1/2)
def _betacf(a, b, x):
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return h


def betai(a, b, x):
    if x <= 0.0 or x >= 1.0:
        return 0.0 if x <= 0.0 else 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def welch(x, y):
    mx, my, vx, vy, nx, ny = statistics.mean(x), statistics.mean(y), statistics.variance(x), statistics.variance(y), len(x), len(y)
    se2 = vx / nx + vy / ny
    t = (mx - my) / math.sqrt(se2)
    df = se2 ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    return t, df, betai(df / 2.0, 0.5, df / (df + t * t))


# ---------------------------------------------------------------- predictions check
def iter_preds(r, results_dir):
    local = os.path.join(r["dir"], f"preds_{r['tag']}.csv")
    if os.path.exists(local):
        return csv.DictReader(open(local, newline="", encoding="utf-8"))
    zp = os.path.join(results_dir, "predictions.zip")
    if os.path.exists(zp):
        z = zipfile.ZipFile(zp)
        member = f"{r['name']}/preds_{r['tag']}.csv"
        if member in z.namelist():
            return csv.DictReader(io.TextIOWrapper(z.open(member), encoding="utf-8", newline=""))
    return None


def recompute(preds):
    by_row = collections.defaultdict(list)
    for p in preds:
        by_row[p["table1_row"]].append(p)
    out = {}
    for row in ORDER:
        ps = by_row[row] + (by_row[UD_REAL.get(row, UD_REAL_DEFAULT)] if ORDER.index(row) >= N_FS else [])
        tp = sum(1 for p in ps if p["label"] == "1_fake" and p["pred"] == "fake")
        tn = sum(1 for p in ps if p["label"] == "0_real" and p["pred"] == "real")
        nf = sum(1 for p in ps if p["label"] == "1_fake")
        nr = sum(1 for p in ps if p["label"] == "0_real")
        out[row] = {"n": nf + nr, "acc": round(100 * (tp + tn) / (nf + nr), 2),
                    "real_acc": round(100 * tn / nr, 2), "fake_acc": round(100 * tp / nf, 2),
                    "unknown": sum(1 for p in ps if p["pred"] == "unknown")}
    return out, sum(len(v) for v in by_row.values())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    ap.add_argument("--out", default=None, help="write Markdown here instead of stdout")
    ap.add_argument("--no_check_predictions", action="store_true")
    a = ap.parse_args()

    runs = load_runs(a.results_dir)
    by = collections.defaultdict(list)
    for r in runs:
        by[r["role"]].append(r)
    for k in by:
        by[k].sort(key=lambda r: r["seed"])
    saf, ctl = by["safaid"], by["control"]
    assert saf, "no SAFAID runs found"
    abl_runs = [by[k][0] for k in ("abl_noprompt", "abl_uniform", "abl_noaug") if by[k]]
    abl_seeds = {r["seed"] for r in abl_runs}
    assert len(abl_seeds) <= 1, f"the ablations use different seeds: {abl_seeds}"
    # reference run: the SAFAID run with the seed of the ablations; it is listed first ("run 1")
    ref_run = next((r for r in saf if r["seed"] in abl_seeds), saf[0])
    assert not abl_runs or ref_run["seed"] in abl_seeds, "no SAFAID run with the seed of the ablations"
    saf = [ref_run] + [r for r in saf if r is not ref_run]
    run_no = lambda rs: ", ".join(str(i + 1) for i in range(len(rs)))

    # ---- consistency checks
    checks = []
    for r in runs:
        stored = r["metrics"].get("average_19")
        assert stored is None or abs(stored - avg19(r)) < 1e-9, (r["name"], stored, avg19(r))
    checks.append(f"- `average_19` stored by `eval.py` equals the mean of the 19 row accuracies for all {len(runs)} runs.")
    for name, vals in BASELINES.items():
        if name in BASELINE_AVG:
            got = round(sum(vals) / len(vals), 2)
            assert abs(got - BASELINE_AVG[name]) < 0.006, (name, got, BASELINE_AVG[name])
    checks.append("- The recomputed 19-row averages of the published baselines match the averages printed in the paper.")
    if not a.no_check_predictions:
        n_checked = 0
        for r in runs:
            preds = iter_preds(r, a.results_dir)
            if preds is None:
                continue
            got, n_img = recompute(preds)
            for row in ORDER:
                ref = r["metrics"]["rows"][row]
                for k in ("n", "acc", "real_acc", "fake_acc", "unknown"):
                    assert got[row][k] == ref[k], (r["name"], row, k, got[row][k], ref[k])
            assert n_img == 82353, (r["name"], n_img)
            n_checked += 1
        if n_checked:
            checks.append(f"- For {n_checked} of {len(runs)} runs, all 19 row accuracies (and real/fake accuracies and "
                          f"unknown counts) were recomputed from the 82,353 per-image predictions and equal the stored metrics.")
    unk = {r["id"]: sum(acc(r, row, "unknown") for row in ORDER) for r in runs}
    detail = ", ".join(f"{k}: {v}" for k, v in unk.items() if v) or "none"
    checks.append(f"- Unparseable answers (scored as errors): {sum(unk.values())} of {82353 * len(runs):,} answers "
                  f"across all {len(runs)} runs ({detail}).")

    L = ["# SAFAID results", "",
         "Generated by `python scripts/summarize_results.py --out results/RESULTS.md` from the run folders in `results/`.",
         "Do not edit by hand.", "",
         "Accuracy (%) on the full test protocol: all 72,353 ForenSynths Table 1 images and all 10,000 UniverDiffu images "
         "(each UniverDiffu generator scored on its 1,000 fakes plus its official 1,000-image real pool). Every run is "
         "Qwen3-VL-4B-Instruct trained on the same 4,000-image ProGAN subset and evaluated at optimizer step 1,600 of a "
         "3,750-step cosine schedule. Mean ± std: sample standard deviation over training runs (ddof = 1).", ""]

    # ---- 1. overview of all runs
    L += ["## 1. All runs", ""]
    rows = []
    for r in runs:
        c = r["cfg"]
        aug = {"fake_only": "fake-only", "uniform": "uniform", "none": "none"}[c.get("aug_mode", "fake_only")]
        tr = r.get("trainable")
        rows.append([r["id"], ROLE_LABEL[r["role"]], r["seed"], PROMPT_TEXT[c["train_prompt"]],
                     PROMPT_TEXT.get(r["tag"], r["tag"]),
                     "QLoRA 4-bit" if r["role"] == "qlora" else "full FT", aug, f"{c['num_train_samples']:,}",
                     f"{c['lr']:.0e}".replace("e-0", "e-"), f2(r.get("epoch")), f"**{f2(avg19(r))}**",
                     f2(avg(r, ORDER[:N_FS])), f2(avg(r, ORDER[N_FS:])),
                     f1(avg(r, key="real_acc")), f1(avg(r, key="fake_acc")),
                     "—" if tr is None else (f"{tr / 1e9:.2f} B" if tr > 1e9 else f"{tr / 1e6:.1f} M"),
                     f1(c.get("train_minutes")), f1(c.get("peak_mem_gb")), f1(r.get("eval_min"))])
    L += table(["Run", "Role", "Seed", "Train prompt", "Test prompt", "Method", "Aug.", "Samples", "LR",
                "Epochs @1600", "Avg (19)", "ForenSynths (11)", "UniverDiffu (8)", "Real acc", "Fake acc",
                "Trained params", "Train min", "Peak mem GB", "Eval min"], rows,
               ["---", "---", "---:", "---", "---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:",
                "---:", "---:", "---:", "---:", "---:", "---:"])
    L += ["", "Real / fake acc: accuracy on authentic / generated images, averaged over the 19 rows. Train min: wall "
          "time to step 1,600 (the runs were made on three vast.ai instances with the same GPU model, so wall times "
          "differ between hosts). "
          "Peak mem: `torch.cuda.max_memory_allocated` during training. Eval min: sum of the per-dataset times in "
          "`eval_log_*.txt` (82,353 images, model loading excluded).", ""]

    # ---- 2. headline
    sv = [avg19(r) for r in saf]
    L += ["## 2. Headline", ""]
    hl = [["SAFAID (this code)", len(saf), ", ".join(f2(v) for v in sv), f"**{ms(sv)}**"]]
    if ctl:
        cv = [avg19(r) for r in ctl]
        hl.append(["CatAID-prompt control, same backbone and recipe", len(ctl), ", ".join(f2(v) for v in cv), ms(cv)])
    hl.append(["CatAID, published (OpenFlamingo-9B)", "—", "—", f2(BASELINE_AVG["CatAID"])])
    for r in by["qlora"]:
        hl.append([f"SAFAID with QLoRA ({r['id']})", 1, f2(avg19(r)), f2(avg19(r))])
    L += table(["Model", "Runs", "Per run", "Average (19), mean ± std"], hl,
               ["---", "---:", "---", "---:"])
    L += [""]
    if ctl and len(saf) > 1 and len(ctl) > 1:
        cv = [avg19(r) for r in ctl]
        t, df, p = welch(sv, cv)
        d_row = [rdiff(statistics.mean(acc(r, row) for r in saf), statistics.mean(acc(r, row) for r in ctl)) for row in ORDER]
        wins = sum(1 for x in d_row if x > 0)
        L += [f"- SAFAID − control: {signed(rdiff(statistics.mean(sv), statistics.mean(cv)))} points on the 19-generator "
              f"average; SAFAID has the higher mean on **{wins} of 19** generators.",
              f"- Welch's t-test on the per-run averages ({len(saf)} vs {len(ctl)} runs): t = {t:.2f}, df = {df:.2f}, "
              f"two-sided p = {p:.2f}. " + ("**The difference is not statistically significant.**" if p >= 0.05 else
                                           "The difference is statistically significant at the 5 % level."),
              f"- SAFAID − published CatAID: {signed(rdiff(statistics.mean(sv), BASELINE_AVG['CatAID']))} points "
              f"(standard deviation over the SAFAID runs: {mean_std(sv)[1]:.2f})."]
        big = sorted(zip(d_row, ORDER), reverse=True)
        L += [f"- Largest margins for SAFAID: " + ", ".join(f"{g} {signed(x)}" for x, g in big[:4]) +
              "; for the control: " + ", ".join(f"{g} {signed(x)}" for x, g in big[::-1][:2] if x < 0) + "."]
    real = statistics.mean(acc(r, row, "real_acc") for r in saf for row in ORDER)
    fake = statistics.mean(acc(r, row, "fake_acc") for r in saf for row in ORDER)
    L += [f"- SAFAID classifies {real:.1f} % of authentic and {fake:.1f} % of generated images correctly "
          f"(mean over the 19 rows and {len(saf)} runs): most errors are generated images labelled real.", ""]

    # ---- 3. Table 1
    L += ["## 3. Table 1: accuracy per generator", "",
          f"SAFAID and the control: mean ± std over {len(saf)} runs. QLoRA: single run. Published baselines: CatAID paper "
          "(Cai et al., ICCVW 2025, Table 1), GADNet from Li et al. (2026), UniverDiffu only; averages recomputed over "
          "the 19 rows. **Bold**: best among SAFAID and the published baselines (the control and QLoRA are not ranked).", ""]
    head = ["Generator", "SAFAID", "Control (same backbone)", "Δ SAFAID − control"] + \
           (["QLoRA"] if by["qlora"] else []) + list(BASELINES)
    rows = []
    q = by["qlora"][0] if by["qlora"] else None
    for i, row in enumerate(ORDER):
        if i in section_rows(len(head)):
            rows.append(section_rows(len(head))[i])
        m_s = statistics.mean(acc(r, row) for r in saf)
        base = {k: v[i] for k, v in BASELINES.items()}
        best = max([round(m_s, 2)] + [v for v in base.values() if v is not None])
        bold = lambda v, s: f"**{s}**" if v is not None and abs(round(v, 2) - best) < 1e-9 else s
        cells = [row, bold(m_s, ms([acc(r, row) for r in saf]))]
        cells += [ms([acc(r, row) for r in ctl]) if ctl else "—",
                  signed(rdiff(m_s, statistics.mean(acc(r, row) for r in ctl))) if ctl else "—"]
        if q:
            cells.append(f2(acc(q, row)))
        cells += [bold(v, f2(v)) for v in base.values()]
        rows.append(cells)
    m_all = statistics.mean(sv)
    base_avg = {k: (round(sum(v) / len(v), 2) if None not in v else None) for k, v in BASELINES.items()}
    best = max([round(m_all, 2)] + [v for v in base_avg.values() if v is not None])
    bold = lambda v, s: f"**{s}**" if v is not None and abs(round(v, 2) - best) < 1e-9 else s
    cells = ["**Average (19)**", bold(m_all, ms(sv)),
             ms([avg19(r) for r in ctl]) if ctl else "—",
             signed(rdiff(m_all, statistics.mean(avg19(r) for r in ctl))) if ctl else "—"]
    if q:
        cells.append(f2(avg19(q)))
    cells += [bold(v, f2(v)) for v in base_avg.values()]
    rows.append(cells)
    L += table(head, rows) + [""]

    # ---- 4. seeds
    L += ["## 4. Run-by-run accuracy", "",
          "The runs use different training seeds. They share the training subset and the data order (Unsloth's fixed "
          "`data_seed`) and differ only in the augmented copies and other random state, so these spreads are training "
          "noise only.", ""]
    head = ["Generator"] + [f"SAFAID run {i + 1} ({r['id']})" for i, r in enumerate(saf)] + \
           ["SAFAID mean ± std", "SAFAID range"] + \
           [f"Control run {i + 1} ({r['id']})" for i, r in enumerate(ctl)] + (["Control mean ± std"] if ctl else [])
    rows = []
    for i, row in enumerate(ORDER):
        if i in section_rows(len(head)):
            rows.append(section_rows(len(head))[i])
        v = [acc(r, row) for r in saf]
        c = [acc(r, row) for r in ctl]
        rows.append([row] + [f2(x) for x in v] + [ms(v), f"{max(v) - min(v):.2f}"] + [f2(x) for x in c] +
                    ([ms(c)] if ctl else []))
    v = [avg19(r) for r in saf]
    c = [avg19(r) for r in ctl]
    rows.append(["**Average (19)**"] + [f"**{f2(x)}**" for x in v] + [f"**{ms(v)}**", f"{max(v) - min(v):.2f}"] +
                [f2(x) for x in c] + ([ms(c)] if ctl else []))
    L += table(head, rows) + [""]
    sg = lambda key: ", ".join(f2(acc(r, key)) for r in saf)
    L += [f"StyleGAN by subclass (SAFAID runs {run_no(saf)}): car + cat only {sg('StyleGAN (car+cat only, old Table 1)')}; "
          f"bedroom {sg('StyleGAN/bedroom')}; car {sg('StyleGAN/car')}; cat {sg('StyleGAN/cat')}.", ""]

    # ---- 5. ablations
    abl = [("Full SAFAID", ref_run)] + [(ROLE_LABEL[k].replace("ablation: ", ""), by[k][0])
                                   for k in ("abl_noprompt", "abl_uniform", "abl_noaug") if by[k]]
    if len(abl) > 1:
        L += ["## 5. Ablations", "",
              f"All with the seed of SAFAID run 1 ({ref_run['id']}) and the same 3,750-step learning-rate schedule, evaluated "
              f"at step 1,600. Single runs: differences are indicative, compare them with the run-to-run spread of the "
              f"full model ({ms(sv)}).", ""]
        rows = []
        for name, r in abl:
            c = r["cfg"]
            rows.append([f"{name} ({r['id']})", f"**{f2(avg19(r))}**",
                         "—" if r is ref_run else signed(avg19(r) - avg19(ref_run)),
                         f1(avg(r, key="real_acc")), f1(avg(r, key="fake_acc")),
                         PROMPT_TEXT[c["train_prompt"]], {"fake_only": "fake only (3 copies per fake)",
                                                          "uniform": "all images (3 copies each)",
                                                          "none": "none"}[c.get("aug_mode", "fake_only")],
                         f"{c['num_train_samples']:,}", f2(r.get("epoch"))])
        L += table(["Configuration", "Avg (19)", "Δ vs full", "Real acc", "Fake acc", "Prompt (train = test)",
                    "Augmentation", "Samples", "Epochs @1600"], rows,
                   ["---", "---:", "---:", "---:", "---:", "---", "---", "---:", "---:"]) + [""]
        head = ["Generator"] + [f"{n} ({r['id']})" for n, r in abl]
        rows = []
        for i, row in enumerate(ORDER):
            if i in section_rows(len(head)):
                rows.append(section_rows(len(head))[i])
            rows.append([row] + [f2(acc(r, row)) for _, r in abl])
        rows.append(["**Average (19)**"] + [f"**{f2(avg19(r))}**" for _, r in abl])
        L += table(head, rows) + [""]
        L += ["The control's test prompt is CatAID's category-free question; the \"w/o investigative prompt\" ablation "
              "uses that same plain question for training and testing.", ""]

    # ---- 6. QLoRA
    if q:
        cq = q["cfg"]
        pair = next((r for r in saf if r["seed"] == q["seed"]), None)  # full fine-tuning run with the QLoRA seed
        full = [pair] if pair else saf
        c = full[0]["cfg"]
        fm = lambda fn: statistics.mean(fn(r) for r in full)
        full_name = f"Full fine-tuning ({pair['id']})" if pair else f"Full fine-tuning (mean of {len(saf)} runs)"
        L += ["## 6. Full fine-tuning vs QLoRA", "",
              ("Same data, prompts, augmentation, schedule and seed" if pair else
               "Same data, prompts, augmentation and schedule") + ". QLoRA: 4-bit base "
              "model, rank 16, alpha 16, adapters on all attention and MLP layers of the vision and language modules, "
              "LR 2e-4, 8-bit AdamW; evaluated after merging the adapters into 16-bit weights."
              + ("" if pair else f" QLoRA is a single run ({q['id']}); the full fine-tuning column is the mean over "
                                 f"the {len(saf)} SAFAID runs."), ""]
        tr = lambda r: "—" if r.get("trainable") is None else f"{r['trainable']:,} of {r['total_params']:,}"
        tmin = [r["cfg"].get("train_minutes") for r in full if r["cfg"].get("train_minutes") is not None]
        tmin_s = "—" if not tmin else (f1(tmin[0]) if len(tmin) == 1 else f"{min(tmin):.1f}–{max(tmin):.1f}")
        a_full, a_q = round(fm(avg19), 2), avg19(q)
        san_full, san_q = round(fm(lambda r: acc(r, "SAN")), 2), acc(q, "SAN")
        real_full, fake_full = fm(lambda r: avg(r, key="real_acc")), fm(lambda r: avg(r, key="fake_acc"))
        rows = [["Avg (19)", f"**{f2(a_full)}**", f"**{f2(a_q)}**", signed(a_q - a_full)],
                ["SAN", f2(san_full), f2(san_q), signed(san_q - san_full)],
                ["Real acc (mean of 19 rows)", f1(real_full), f1(avg(q, key='real_acc')),
                 signed(round(avg(q, key='real_acc'), 1) - round(real_full, 1), 1)],
                ["Fake acc (mean of 19 rows)", f1(fake_full), f1(avg(q, key='fake_acc')),
                 signed(round(avg(q, key='fake_acc'), 1) - round(fake_full, 1), 1)],
                ["Peak training memory (GB)", f2(c.get("peak_mem_gb")), f2(cq.get("peak_mem_gb")), ""],
                ["Trainable parameters", tr(full[0]), tr(q), ""],
                ["Training time to step 1,600 (min)", tmin_s, f1(cq.get("train_minutes")), ""],
                ["Learning rate / optimizer", f"{c['lr']:.0e} / {c['optim']}", f"{cq['lr']:.0e} / {cq['optim']}", ""]]
        L += table(["", full_name, f"QLoRA ({q['id']})", "Δ QLoRA − full"], rows,
                   ["---", "---:", "---:", "---:"]) + [""]
        head = ["Generator", "Full FT acc", "QLoRA acc", "Δ", "Full FT real / fake", "QLoRA real / fake"]
        rows = []
        for i, row in enumerate(ORDER):
            if i in section_rows(len(head)):
                rows.append(section_rows(len(head))[i])
            a_row = round(fm(lambda r: acc(r, row)), 2)
            rows.append([row, f2(a_row), f2(acc(q, row)), signed(acc(q, row) - a_row),
                         f"{f1(fm(lambda r: acc(r, row, 'real_acc')))} / {f1(fm(lambda r: acc(r, row, 'fake_acc')))}",
                         f"{f1(acc(q, row, 'real_acc'))} / {f1(acc(q, row, 'fake_acc'))}"])
        L += table(head, rows) + [""]

    # ---- 7. real vs fake
    L += ["## 7. Accuracy on authentic vs generated images", "",
          "Per-generator accuracy on the authentic (real) and generated (fake) images of each row, mean over runs; "
          "min–max is over the SAFAID runs. UniverDiffu rows share their real pools (ImageNet for ADM, LAION for the "
          "others).", ""]
    head = ["Generator", "SAFAID real", "SAFAID fake", "SAFAID fake min–max"] + \
           (["Control real", "Control fake"] if ctl else [])
    rows = []
    for i, row in enumerate(ORDER):
        if i in section_rows(len(head)):
            rows.append(section_rows(len(head))[i])
        fk = [acc(r, row, "fake_acc") for r in saf]
        cells = [row, f1(statistics.mean(acc(r, row, "real_acc") for r in saf)), f1(statistics.mean(fk)),
                 f"{min(fk):.1f}–{max(fk):.1f}"]
        if ctl:
            cells += [f1(statistics.mean(acc(r, row, "real_acc") for r in ctl)),
                      f1(statistics.mean(acc(r, row, "fake_acc") for r in ctl))]
        rows.append(cells)
    cells = ["**Mean (19)**", f"**{real:.1f}**", f"**{fake:.1f}**", ""]
    if ctl:
        cells += [f1(statistics.mean(acc(r, row, "real_acc") for r in ctl for row in ORDER)),
                  f1(statistics.mean(acc(r, row, "fake_acc") for r in ctl for row in ORDER))]
    rows.append(cells)
    L += table(head, rows) + [""]
    per_run = ", ".join(f"run {i + 1}: {avg(r, key='real_acc'):.1f} / {avg(r, key='fake_acc'):.1f}"
                        for i, r in enumerate(saf))
    L += [f"Per SAFAID run (real / fake): {per_run}.", ""]

    # ---- 8. checks
    L += ["## 8. Consistency checks", ""] + checks + [""]

    md = "\n".join(L)
    if a.out:
        with open(a.out, "w") as f:
            f.write(md)
    else:
        print(md, end="")


if __name__ == "__main__":
    main()
