#!/usr/bin/env python3
"""Post-cutoff fakes for the SAFAID contamination test (see contamination/README.md).

Qwen3-VL-4B-Instruct was released on 2025-10-15. Its pretraining data cannot contain images from
generators, or from decoders (VAEs), released after that date. Three generators, reported as two
groups that are NEVER averaged together:

  group PostCutoff                   (post-cutoff model AND post-cutoff decoder)
      flux2_klein_4b     FLUX.2-klein-4B     (BFL, 2026-01-15)  FLUX.2 VAE (2025-11-25)
      ernie_image_turbo  ERNIE-Image-Turbo   (Baidu, 2026-04-02) FLUX.2 VAE, the same file as klein's
  group PostCutoffModelPreCutoffVAE  (post-cutoff model, pre-cutoff decoder)
      z_image_turbo      Z-Image-Turbo       (Tongyi-MAI, 2025-11-25) FLUX.1-family VAE (2024)

N images per generator (default 1000) at the native 1024x1024, from a fixed seeded prompt plan: image i
uses prompt order[i % 200] (a shuffle of prompts_postcutoff.PROMPTS with --prompt_seed) and its own CPU
torch.Generator seeded base_seed+i, identical across generators (paired design). Each 1024 image goes
through postcutoff_common.to_eval_png(): one 2x Lanczos downscale to 512, PNG without metadata -- the
same function (and the same 2x factor) the Wikimedia reals go through.

    <out>/<generator>/1_fake/<generator>_<idx:05d>.png    512x512 PNG
    <out>/<generator>/metadata.jsonl                      one record per image (prompt, seed, src size, scale)
    <out>/<generator>/run_info.json                       config, group, versions, GPU, VAE sha256, calibration

Run in a separate venv (the training env pins transformers 4.57.3; these pipelines need diffusers 0.38 /
transformers 5.x). extra_steps.sh builds that venv and calls this script once per generator:
    genv/bin/python gen_fakes.py --check_env                           # configs + tokenizers only
    genv/bin/python gen_fakes.py --out DATA --generators flux2_klein_4b --n 1000 --purge_cache
Resumable: finished images are skipped. After the first --calib_batches batches it prints a CALIBRATION
line (s/img, projected minutes, peak VRAM, pixel statistics) and aborts that generator (exit 3) if the
images are degenerate (flat / black / NaN).
"""
import argparse
import json
import os
import platform
import random
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from postcutoff_common import CROP, OUT_RES, file_digests, to_eval_png  # noqa: E402
from prompts_postcutoff import PROMPTS, PROMPTS_SHA256  # noqa: E402

GEN_RES = CROP  # generate at exactly the crop size, so the only resample is the shared 2x Lanczos downscale
FLUX2_VAE_SHA256 = "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04"

# Revisions are HF commit SHAs (checked 2026-09-29: none gated, all Apache-2.0).
GENERATORS = {
    "flux2_klein_4b": {
        "display": "FLUX.2-klein-4B", "org": "Black Forest Labs", "released": "2026-01-15",
        "group": "PostCutoff", "decoder_post_cutoff": True,
        "repo": "black-forest-labs/FLUX.2-klein-4B", "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "license": "apache-2.0", "pipeline": "Flux2KleinPipeline", "min_diffusers": "0.37.0",
        "params": "4B rectified-flow transformer + Qwen3-4B text encoder",
        "vae": "FLUX.2 VAE (AutoencoderKLFlux2, released 2025-11-25); same file as ERNIE-Image-Turbo's",
        "vae_sha256_expected": FLUX2_VAE_SHA256,
        "call": {"num_inference_steps": 4, "guidance_scale": 1.0},  # model card: step-distilled, 4 steps
        "load": {},
        "ignore_patterns": ["flux-2-klein-4b.safetensors", "*.jpg", "*.png", "*.md"],  # skip the single-file ckpt
    },
    "ernie_image_turbo": {
        "display": "ERNIE-Image-Turbo", "org": "Baidu", "released": "2026-04-02",
        "group": "PostCutoff", "decoder_post_cutoff": True,
        "repo": "baidu/ERNIE-Image-Turbo", "revision": "bc68c81e2a1730a394d5fc9fae70713dee940140",
        "license": "apache-2.0", "pipeline": "ErnieImagePipeline", "min_diffusers": "0.38.0",
        "params": "8B single-stream DiT + Mistral3 (3B) text encoder; prompt enhancer not loaded",
        "vae": "FLUX.2 VAE (AutoencoderKLFlux2); same file as FLUX.2-klein-4B's",
        "vae_sha256_expected": FLUX2_VAE_SHA256,
        # model card: 8 steps, guidance 1.0. The prompt enhancer (PE) samples a rewrite at T=0.6, which would
        # make the prompts differ across generators and runs, so it is switched off (use_pe=False, not loaded).
        "call": {"num_inference_steps": 8, "guidance_scale": 1.0, "use_pe": False},
        "load": {"pe": None, "pe_tokenizer": None},
        "ignore_patterns": ["pe/*", "pe_tokenizer/*", "*.jpg", "*.png", "*.md"],
    },
    "z_image_turbo": {
        "display": "Z-Image-Turbo", "org": "Alibaba Tongyi-MAI", "released": "2025-11-25",
        "group": "PostCutoffModelPreCutoffVAE", "decoder_post_cutoff": False,
        "repo": "Tongyi-MAI/Z-Image-Turbo", "revision": "f332072aa78be7aecdf3ee76d5c247082da564a6",
        "license": "apache-2.0", "pipeline": "ZImagePipeline", "min_diffusers": "0.36.0",
        "params": "6B S3-DiT + Qwen3-4B text encoder",
        "vae": ("FLUX.1-family VAE (AutoencoderKL, 16 latent ch, scaling 0.3611, shift 0.1159; same config and "
                "byte size as FLUX.1-schnell's 2024 VAE file) -- a pre-cutoff decoder"),
        "vae_sha256_expected": None,
        "call": {"num_inference_steps": 9, "guidance_scale": 0.0},  # model card: 9 steps = 8 DiT forwards, CFG 0
        "load": {},
        "ignore_patterns": ["assets/*", "*.jpg", "*.png", "*.md"],
    },
}
DEFAULT_GENERATORS = ["flux2_klein_4b", "ernie_image_turbo", "z_image_turbo"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", help="output root (the <out> of the eval layout)")
    p.add_argument("--generators", default=",".join(DEFAULT_GENERATORS),
                   help=f"comma list from {list(GENERATORS)} (default: %(default)s)")
    p.add_argument("--n", type=int, default=1000, help="images per generator")
    p.add_argument("--base_seed", type=int, default=20251015, help="image i uses seed base_seed+i")
    p.add_argument("--prompt_seed", type=int, default=0, help="seed of the one-time prompt shuffle")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--keep_native", action="store_true", help="also save the 1024 PNG in <gen>/native/")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gen_device", default="cpu", help="device of the torch.Generator (cpu = portable seeds)")
    p.add_argument("--offload", action="store_true", help="enable_model_cpu_offload (GPUs < 48 GB)")
    p.add_argument("--calib_batches", type=int, default=2, help="batches before the CALIBRATION line (>=2)")
    p.add_argument("--min_pixel_std", type=float, default=2.0, help="calibration: abort if an image is flatter")
    p.add_argument("--purge_cache", action="store_true",
                   help="delete this generator's HF cache folder once all its images exist (saves disk)")
    p.add_argument("--local_path", action="append", default=[], metavar="GEN=DIR",
                   help="load GEN from a local diffusers folder instead of the Hub (tests / offline)")
    p.add_argument("--test_res", type=int, default=0,
                   help="TESTS ONLY (tiny random pipelines on CPU): generate/crop at this size, output test_res//2")
    p.add_argument("--check_env", action="store_true", help="download configs+tokenizers only and resolve all classes")
    p.add_argument("--plan_only", action="store_true", help="print the prompt/seed plan and exit (no torch needed)")
    return p.parse_args()


def plan(n, prompt_seed, base_seed):
    order = list(range(len(PROMPTS)))
    random.Random(prompt_seed).shuffle(order)
    out = []
    for i in range(n):
        pid = order[i % len(order)]
        out.append({"idx": i, "prompt_id": pid, "category": PROMPTS[pid][0], "prompt": PROMPTS[pid][1],
                    "seed": base_seed + i})
    return out


def versions():
    v = {"python": platform.python_version()}
    for m in ("torch", "diffusers", "transformers", "accelerate", "huggingface_hub", "safetensors", "PIL", "numpy"):
        try:
            v[m] = __import__(m).__version__
        except Exception:
            v[m] = None
    return v


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def check_env(names):
    import importlib
    import diffusers
    import transformers
    from huggingface_hub import snapshot_download
    ok = True
    print("versions:", versions())
    for name in names:
        g = GENERATORS[name]
        path = snapshot_download(g["repo"], revision=g["revision"], allow_patterns=[
            "model_index.json", "*/config.json", "scheduler/*", "tokenizer/*", "*/generation_config.json",
            "*/chat_template.jinja"])
        mi = json.load(open(os.path.join(path, "model_index.json")))
        cls_ok = hasattr(diffusers, g["pipeline"]) and mi["_class_name"] == g["pipeline"]
        print(f"== {name}: {g['repo']}@{g['revision'][:8]} -> {mi['_class_name']} ({'OK' if cls_ok else 'MISMATCH'})")
        ok &= cls_ok
        for comp, spec in mi.items():
            if comp.startswith("_") or not isinstance(spec, list) or spec[0] is None or comp in g["load"]:
                continue
            lib, cls = spec
            found = getattr(importlib.import_module(lib), cls, None) is not None
            msg = "class OK" if found else "CLASS MISSING"
            try:
                if comp.startswith("tokenizer"):
                    tok = transformers.AutoTokenizer.from_pretrained(path, subfolder=comp)
                    msg += f", tokenizer {type(tok).__name__} loads"
                elif lib == "transformers":
                    msg += f", config {type(transformers.AutoConfig.from_pretrained(path, subfolder=comp)).__name__}"
            except Exception as e:
                found = False
                msg += f", LOAD FAIL {type(e).__name__}: {str(e)[:120]}"
            ok &= found
            print(f"   {comp:13s} {lib}.{cls}: {msg}")
    print("ENV OK" if ok else "ENV PROBLEM")
    return ok


def load_pipe(name, g, args, local):
    import diffusers
    import torch
    if name in local:
        path = local[name]
    else:
        from huggingface_hub import snapshot_download
        path = snapshot_download(g["repo"], revision=g["revision"], ignore_patterns=g["ignore_patterns"])
    cls = getattr(diffusers, g["pipeline"])
    pipe = cls.from_pretrained(path, torch_dtype=torch.bfloat16, **g["load"])
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, path


def purge_hf_cache(g):
    """Remove models--<org>--<name> from the active HF hub cache. Returns bytes freed."""
    from huggingface_hub import constants
    d = os.path.join(constants.HF_HUB_CACHE, "models--" + g["repo"].replace("/", "--"))
    if not os.path.isdir(d):
        return 0
    n = 0
    for root, _, files in os.walk(d):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                n += os.path.getsize(fp)
    shutil.rmtree(d, ignore_errors=True)
    return n


def pixel_stats(img):
    import numpy as np
    a = np.asarray(img, dtype=np.float32)
    return float(a.mean()), float(a.std()), bool(np.isfinite(a).all())


def run_generator(name, args, local):
    import torch
    g = GENERATORS[name]
    res = args.test_res or GEN_RES
    out_res = args.test_res // 2 if args.test_res else OUT_RES
    gdir = os.path.join(args.out, name)
    fdir = os.path.join(gdir, "1_fake")
    ndir = os.path.join(gdir, "native")
    os.makedirs(fdir, exist_ok=True)
    if args.keep_native:
        os.makedirs(ndir, exist_ok=True)
    meta_path = os.path.join(gdir, "metadata.jsonl")
    done = {}
    if os.path.exists(meta_path):
        for line in open(meta_path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a killed run can leave a partial last line
            if os.path.exists(os.path.join(fdir, r["file"])):
                done[r["idx"]] = r
    items = plan(args.n, args.prompt_seed, args.base_seed)
    todo = [it for it in items if it["idx"] not in done]
    print(f"[{name}] group={g['group']}: {len(done)} done, {len(todo)} to generate at {res}x{res} -> {out_res}",
          flush=True)
    info_path = os.path.join(gdir, "run_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    info.update({"generator": name, "benchmark": g["group"], **{k: v for k, v in g.items() if k != "ignore_patterns"},
                 "gen_res": res, "crop": res, "out_res": out_res, "scale_factor": res / out_res, "n": args.n,
                 "base_seed": args.base_seed, "prompt_seed": args.prompt_seed, "prompts_sha256": PROMPTS_SHA256,
                 "batch_size": args.batch_size, "gen_device": args.gen_device, "dtype": "bfloat16",
                 "offload": args.offload, "test_res": args.test_res or None,
                 "watermark": "none (diffusers pipelines add none)",
                 "preprocess": f"generate {res}x{res} -> postcutoff_common.to_eval_png: Lanczos {res}->{out_res} "
                               f"(one 2x downscale), PNG without metadata",
                 "versions": versions(), "host": platform.node()})
    if not todo:
        info["n_images"] = len(done)
        json.dump(info, open(info_path, "w"), indent=1, ensure_ascii=False)
        if args.purge_cache and name not in local:
            print(f"[{name}] complete; purged {purge_hf_cache(g) / 1e9:.1f} GB of HF cache", flush=True)
        return 0
    t_load = time.time()
    pipe, path = load_pipe(name, g, args, local)
    info["load_seconds"] = round(time.time() - t_load, 1)
    info["weights_path"] = path
    vae_file = os.path.join(path, "vae", "diffusion_pytorch_model.safetensors")
    if os.path.exists(vae_file):
        info["vae_sha256"] = sha256_file(vae_file)
        exp = g.get("vae_sha256_expected")
        if exp and name not in local:
            info["vae_sha256_matches_expected"] = info["vae_sha256"] == exp
            print(f"[{name}] VAE sha256 {info['vae_sha256'][:12]} "
                  f"{'== FLUX.2 VAE (shared with the other PostCutoff generator)' if info['vae_sha256'] == exp else '!= expected ' + exp[:12]}",
                  flush=True)
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
    timings, calib_stats = [], []
    with open(meta_path, "a") as mf:
        for bi, b in enumerate(range(0, len(todo), args.batch_size)):
            batch = todo[b:b + args.batch_size]
            kw = dict(g["call"])
            kw.update(prompt=[it["prompt"] for it in batch], height=res, width=res,
                      generator=[torch.Generator(device=args.gen_device).manual_seed(it["seed"]) for it in batch])
            t0 = time.time()
            with torch.inference_mode():
                images = pipe(**kw).images
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.time() - t0
            timings.append((len(batch), dt))
            for it, img in zip(batch, images):
                if img.size != (res, res):
                    raise RuntimeError(f"{name} returned {img.size}, expected {(res, res)}")
                if bi < args.calib_batches:
                    calib_stats.append(pixel_stats(img))
                fname = f"{name}_{it['idx']:05d}.png"
                if args.keep_native:
                    img.save(os.path.join(ndir, fname))
                prep = to_eval_png(img, os.path.join(fdir, fname), crop=res, size=out_res)
                size, crc, sha = file_digests(os.path.join(fdir, fname))
                rec = {**it, "file": fname, "member": f"{name}/1_fake/{fname}", "gen_w": img.size[0],
                       "gen_h": img.size[1], **prep, "size": size, "crc32": crc, "sha256": sha,
                       "batch_seconds": round(dt, 3), "batch_len": len(batch)}
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mf.flush()
            n_done = sum(k for k, _ in timings)
            steady = timings[1:] or timings  # the first batch includes CUDA warm-up
            spi = sum(t for _, t in steady) / max(1, sum(k for k, _ in steady))
            eta = spi * (len(todo) - n_done) / 60
            if bi + 1 == min(args.calib_batches, -(-len(todo) // args.batch_size)):
                peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                means = [m for m, _, _ in calib_stats]
                stds = [s for _, s, _ in calib_stats]
                finite = all(f for _, _, f in calib_stats)
                cal = {"first_batch_seconds": round(timings[0][1], 2), "steady_sec_per_img": round(spi, 3),
                       "projected_minutes_total": round(spi * len(todo) / 60, 1), "peak_vram_gb": round(peak, 1),
                       "gen_size": [res, res], "out_size": [out_res, out_res],
                       "pixel_mean_range": [round(min(means), 1), round(max(means), 1)],
                       "pixel_std_min": round(min(stds), 2), "finite": finite, "n_images": len(calib_stats)}
                info["calibration"] = cal
                print(f"CALIBRATION [{name}] {json.dumps(cal)}", flush=True)
                if not finite or min(stds) < args.min_pixel_std:
                    info["calibration"]["status"] = "FAILED: degenerate images"
                    json.dump(info, open(info_path, "w"), indent=1, ensure_ascii=False)
                    print(f"CALIBRATION FAILED [{name}]: degenerate images (std {min(stds):.2f} < "
                          f"{args.min_pixel_std} or non-finite); aborting this generator", flush=True)
                    return 3
                info["calibration"]["status"] = "ok"
            print(f"[{name}] {n_done}/{len(todo)}  {spi:.2f} s/img  ETA {eta:.1f} min", flush=True)
            info.update({"sec_per_img_steady": round(spi, 3), "first_batch_seconds": round(timings[0][1], 2),
                         "images_this_run": n_done, "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
            json.dump(info, open(info_path, "w"), indent=1, ensure_ascii=False)
    del pipe
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    n_files = len([f for f in os.listdir(fdir) if f.endswith(".png")])
    info["n_images"] = n_files
    json.dump(info, open(info_path, "w"), indent=1, ensure_ascii=False)
    if n_files >= args.n and args.purge_cache and name not in local:
        print(f"[{name}] complete; purged {purge_hf_cache(g) / 1e9:.1f} GB of HF cache", flush=True)
    return 0


def main():
    args = parse_args()
    names = [s.strip() for s in args.generators.split(",") if s.strip()]
    bad = [s for s in names if s not in GENERATORS]
    if bad:
        sys.exit(f"unknown generator(s) {bad}; choose from {list(GENERATORS)}")
    if args.plan_only:
        pl = plan(args.n, args.prompt_seed, args.base_seed)
        for it in pl[:10]:
            print(it)
        from collections import Counter
        print("categories:", dict(Counter(it["category"] for it in pl)))
        print("prompts_sha256:", PROMPTS_SHA256)
        return
    if args.check_env:
        sys.exit(0 if check_env(names) else 1)
    if not args.out:
        sys.exit("--out is required")
    if args.calib_batches < 1:
        sys.exit("--calib_batches must be >= 1")
    local = dict(kv.split("=", 1) for kv in args.local_path)
    rc = 0
    for name in names:
        rc = max(rc, run_generator(name, args, local))
    print("done:", ", ".join(names), "rc", rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()
