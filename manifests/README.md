# Manifests

These files pin every image used for training and evaluation. They hold file names, byte offsets and checksums only. The images stay under their datasets' licenses (see `../LICENSE`).

| File | Rows | What it is |
|---|---|---|
| `safaid_train_subset_v2_seed0.csv` | 4,000 | SAFAID training subset: 100 real + 100 fake ProGAN images for each of the 20 LSUN categories |
| `safaid_train_subset_v2_seed0_sha256.csv` | 4,000 | Same rows plus the SHA256 of each decompressed image. `safaid/fetch_data.py` reads this file. |
| `safaid_train_subset_v2_seed0_sha256.txt` | 4,000 | `sha256sum -c` format, with the paths `train/<category>/<label>/<file>` |
| `safaid_val_subset_v2_seed0*.csv/.txt` | 800 | Seeded validation subset, 20 real + 20 fake per category. The paper runs did not use it (see below). |
| `eval_members_manifest.csv` | 100,329 | Every member of the two test zips, mapped to its Table 1 row |
| `diffusion_datasets_zip.sha256`, `diffusion_zip_bytes.txt` | 1 | SHA256 and byte size of the UniverDiffu `diffusion_datasets.zip` (917,979,875 bytes) |

## Columns

**Train/val subsets**
`split, source, category, label, name, size, crc, offset, csize, method[, sha256]`

- `source`: `progan_train` (training zip), `progan_val` (official validation zip), or `progan_train_heldout`.
- `name`: the member path inside the zip, e.g. `airplane/0_real/00190.png`.
- `size`, `csize`, `crc`, `method`, `offset`: the uncompressed size, compressed size, CRC32, compression method (8 = deflate) and local-header offset, all from the zip central directory.
- `sha256`: the SHA256 of the decompressed file.

**Eval manifest**
`benchmark, table1_row, folder, subclass, label, zip, member, size, crc32`

- `table1_row`: one of the 19 Table 1 rows. Two further values mark the UniverDiffu real pools: `real pool for ADM only` (ImageNet) and `real pool for all LDM/Glide/DALL-E rows` (LAION).
- `(not in Table 1)` marks members that are listed but not evaluated: the `stylegan2` and `whichfaceisreal` folders, 17,976 images in total.
- `subclass`: set for StyleGAN only (`bedroom`, `car`, `cat`).
- `size`, `crc32`: from the zip central directory. `python scripts/verify_data.py` checks the extracted files against these values.

## How the training subset was selected (`scripts/make_subset_v2.py`)

1. **Candidate pool.** Take the PNGs in the ProGAN training zip whose (CRC32, size) pair is unique within that zip and appears in neither `progan_val.zip` nor `CNN_synth_testset.zip`. This removes duplicates and any train/test overlap.
2. **Sampling.** For each (category, label) cell, draw 100 files with `random.Random(f"safaid-train-seed0-{category}-{label}")`.
3. **Validation set.** Take 20 files per cell from `progan_val.zip` after the same de-duplication. Two cells, `cat/0_real` and `horse/0_real`, have no clean validation files. They are filled from the clean training pool, without touching the training picks, and are marked `progan_train_heldout`.

The selection is deterministic. Running the script again on the zip listings reproduces both CSVs byte for byte:

```bash
# 1. Build the zip listings: this reads only the central directories, via HTTP range requests
python manifests/scripts/multipart.py      # -> manifests/progan_train_listing.json (training set inside the 7z volumes)
python manifests/scripts/httpzip.py https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip \
       manifests/progan_val_listing.json
python manifests/scripts/httpzip.py https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/CNN_synth_testset.zip \
       manifests/cnn_synth_testset_listing.json
# 2. Select the subsets. This rewrites safaid_{train,val}_subset_v2_seed0.csv, so check `git diff` afterwards.
python manifests/scripts/make_subset_v2.py
# 3. Optional: download the selected members by range request and recompute the *_sha256.csv/.txt files
python manifests/scripts/hash_subsets.py
```

If you already have the full `progan_train.zip`, `scripts/extract_subset_from_local_zip.py` extracts only the listed members and checks their sizes and SHA256. It writes them to `<out>/train/<category>/<label>/`, so pass `--data_dir <out>/train` to `safaid/train.py`.

The scripts resolve `manifests/` from their own location. Set `SAFAID_MANIFESTS` to use another directory.
