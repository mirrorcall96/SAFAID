# Post-cutoff contamination test (pending)

Qwen3-VL was pretrained on a large, undisclosed web corpus, so some benchmark images (or near-duplicates) may have been seen during pretraining. This test evaluates the trained detectors on images that **cannot** be in that corpus: fakes from generators released after the backbone (Qwen3-VL-4B-Instruct, 2025-10-15) and real photographs taken after 2025-11-01.

**Status:** the scripts are complete and tested; results will be added here and to `results/` once the run is done.

## Design

| Side | Source | Count | Notes |
|---|---|---:|---|
| Fake, post-cutoff model and decoder | [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), [ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo) | 1,000 each | both share the FLUX.2 VAE; reported per generator and as one group |
| Fake, post-cutoff model, older decoder | [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | 1,000 | FLUX.1 VAE (2024); reported separately, never averaged with the group above |
| Real | Wikimedia Commons, "Photographs taken on YYYY-MM-DD", 2025-11-01 onwards | 1,000 | free licences only, camera EXIF (Make/Model) required, AI/composite source types and C2PA manifests rejected, one photo per uploader per day; attribution recorded for every file |

- **Prompts:** a fixed, seeded list covering the 20 LSUN/VOC categories plus scenes, faces and landscapes (`prompts_postcutoff.py`); every generator uses the same prompt/seed plan.
- **Matched processing:** fakes are generated at 1024 × 1024; reals are centre-cropped to 1024 × 1024 at native resolution. Both then get exactly one Lanczos 1024 → 512 resize and are saved as metadata-free PNGs, so resolution and JPEG history cannot act as a shortcut.
- **Metrics:** real accuracy, fake accuracy and **balanced accuracy** ((real + fake) / 2) per generator and per decoder group, because the 1,000 reals are shared by three generators.
- **Inference:** `eval_post.py` copies the inference code of `safaid/eval.py` verbatim (`--check_against safaid/eval.py` verifies this).

## Files

| File | Purpose |
|---|---|
| `gen_fakes.py` | generates the fakes, one generator at a time, pinned model revisions; prints a calibration line (s/img, VRAM) |
| `fetch_recent_reals.py` | downloads and filters the Wikimedia photos, writes `attribution.csv` |
| `postcutoff_common.py`, `prompts_postcutoff.py` | shared image processing and the prompt list |
| `eval_manifest_builder.py` | builds a manifest in the format of `manifests/eval_members_manifest.csv` |
| `eval_post.py` | evaluates one model on the manifest |
| `score_postcutoff.py` | aggregates all models into `RESULTS_postcutoff.md` |
| `extra_steps.sh` | end-to-end driver for the GPU machine (separate `diffusers` venv, never touches the training environment) |

## Usage

```bash
# real photos (CPU only; Wikimedia asks for a contact in the User-Agent)
python contamination/fetch_recent_reals.py --out data/postcutoff --n 1000 --contact <your email or URL>

# fakes (GPU; separate venv with diffusers 0.38.0 and transformers 5.7.0)
python contamination/gen_fakes.py --out data/postcutoff --generators flux2_klein_4b,ernie_image_turbo,z_image_turbo --n 1000

# manifest, evaluation, scoring
python contamination/eval_manifest_builder.py --root data/postcutoff --out data/postcutoff/postcutoff_manifest.csv --verify
python contamination/eval_post.py --manifest data/postcutoff/postcutoff_manifest.csv \
       --model_path <run>/model/checkpoint-1600 --run_dir runs/postcutoff/<run> --test_prompt investigative --tag investigative
python contamination/score_postcutoff.py --runs_root runs/postcutoff --out_md runs/postcutoff/RESULTS_postcutoff.md
```

Estimated GPU time on one RTX PRO 6000: ~2 h to generate the 3,000 fakes (0.7 s, 3.3 s and 2.4 s per image) and a few minutes per evaluated model.
