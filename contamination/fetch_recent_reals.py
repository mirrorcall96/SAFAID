#!/usr/bin/env python3
"""Post-cutoff REAL photographs from Wikimedia Commons for the SAFAID contamination test.

Qwen3-VL-4B-Instruct was released on 2025-10-15. A photo *taken* on or after 2025-11-01 cannot be in its
pretraining data. This script collects about 1000 of them, spread over the whole window, and writes them
in the same layout and with the same processing as the fakes:

    <out>/wikimedia_2026/0_real/wm_<pageid>.png   512x512 PNG (postcutoff_common.to_eval_png)
    <out>/wikimedia_2026/attribution.csv          URL, license, author, capture date, camera, src size, crop, ...
    <out>/wikimedia_2026/run_info.json            arguments, filter counts, query time
    <out>/wikimedia_2026/_cache/                  per-day candidate lists (resumable, auditable)

Source: "Category:Photographs taken on YYYY-MM-DD" (direct file members). A file is kept only if ALL hold.
Checked on the Commons API metadata first, and again on the downloaded original bytes (authoritative):
  * JPEG, shortest side >= --min_side (default 1024 = the native crop), <= --max_mpix, aspect <= --max_aspect
  * camera EXIF: Make or Model present, AND each of ExposureTime / FNumber / ISO / FocalLength that is
    present is a positive number (zeros, NaN and 0/0 rationals are rejected)
  * EXIF DateTimeOriginal >= --start (2025-11-01), within +-1 day of the category date, not after upload
  * no IPTC DigitalSourceType for AI / composite / non-photographic media (trainedAlgorithmicMedia,
    compositeWithTrainedAlgorithmicMedia, algorithmicMedia, and also compositeSynthetic, composite,
    algorithmicallyEnhanced, dataDrivenMedia, digitalArt, virtualRecording, screenCapture, softwareImage,
    digitalCreation), searched in XMP / IPTC / EXIF / C2PA segments
  * no C2PA (JUMBF) manifest (--c2pa reject_any, default) or none that declares AI (--c2pa reject_ai), and
    no "Made with AI" / generator-name marker in the metadata segments or the EXIF Software field
  * a free license from --licenses (CC0, public domain, CC BY, CC BY-SA); Commons "Restrictions" empty
  * no category, title or description matching an AI / render / screenshot / collage / nudity pattern
  * at most one file per uploader per day and at most --per_user_cap per uploader in total
Selection goes round-robin over days. The original file is downloaded (a Commons thumbnail would add a
resample and a second JPEG encode) and its SHA-1 is checked against the API value.

Processing (identical function to the fakes): EXIF orientation -> sRGB -> center crop 1024x1024 at native
resolution -> Lanczos 1024 -> 512 (one 2x downscale) -> PNG without metadata. The source size, crop box and
scale factor are written to attribution.csv.

Usage (the contact string goes into the User-Agent, as the Wikimedia User-Agent policy asks):
    python fetch_recent_reals.py --out DATA --contact "<email or URL>" --metadata_only 5 --dry_download
    python fetch_recent_reals.py --out DATA --contact "<email or URL>" --n 1000
Dependencies: requests, Pillow. Resumable.
"""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import html
import io
import json
import math
import os
import random
import re
import struct
import sys
import time

import requests
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from postcutoff_common import CROP, OUT_RES, file_digests, to_eval_png  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
EXT_FIELDS = ("ObjectName|ImageDescription|Artist|Credit|LicenseShortName|License|LicenseUrl|UsageTerms|"
              "AttributionRequired|Copyrighted|Restrictions|DateTimeOriginal")
DEFAULT_LICENSES = r"^(cc0|pd|cc-by-[1-4]\.[05]|cc-by-sa-[1-4]\.[05])$"
BAD_TEXT = re.compile(
    r"(\bAI[- ]generated|artificial intelligence|generative (AI|art)|\bgenAI\b|stable diffusion|midjourney|dall[-· ]?e|"
    r"\bflux\.?[12]\b|adobe firefly|google imagen|chatgpt|gemini ai|google gemini|computer[- ]generated|\bCGI\b|"
    r"\brender(s|ed|ing)?\b|made with ai|"
    r"3D model|screenshot|screen ?shot|collage|photomontage|montage|composite|digitally (altered|manipulated)|"
    r"\bnud(e|es|ity)\b|naked|topless|erotic|pornograph)", re.I)
BAD_SOFTWARE = re.compile(r"(firefly|midjourney|stable ?diffusion|dall|generative|comfyui|automatic1111|flux|"
                          r"imagen|novelai|leonardo|ideogram|chatgpt|openai)", re.I)
# markers searched in the raw metadata segments (XMP, IPTC-IIM, EXIF, COM, C2PA/JUMBF), lower-cased latin-1
AI_MARKERS = re.compile(
    r"(made with ai|ai[- ]generated|generated (?:with|by|using) (?:an? )?ai\b|generative (?:ai|fill|expand)|"
    r"adobe firefly|midjourney|dall[-· ]?e\b|stable ?diffusion|comfyui|automatic1111|novelai|leonardo\.ai|ideogram|"
    r"google imagen|gemini ai|chatgpt|openai|synthid|c2pa\.ai_|\bgenai\b)")
DST_RE = re.compile(rb"digitalsourcetype/([A-Za-z]+)", re.I)
# IPTC NewsCodes digital source types that are AI-made, composites or not camera captures
REJECT_DST = {c.lower() for c in (
    "trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia", "algorithmicMedia",  # AI-made media
    "compositeSynthetic", "composite", "algorithmicallyEnhanced", "dataDrivenMedia", "digitalArt",
    "virtualRecording", "screenCapture", "softwareImage", "digitalCreation")}
EXPOSURE_API = ("ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength")
EXPOSURE_TAGS = {"ExposureTime": 0x829A, "FNumber": 0x829D, "ISOSpeedRatings": 0x8827, "FocalLength": 0x920A}
FIELDS = ["file", "pageid", "title", "page_url", "file_url", "sha1", "width", "height", "bytes", "mime",
          "license_short", "license_code", "license_url", "attribution_required", "author", "author_html", "credit",
          "uploader", "upload_timestamp", "exif_datetime_original", "category_date", "camera_make", "camera_model",
          "exposure_time", "f_number", "iso", "focal_length", "software", "digital_source_type", "c2pa",
          "icc", "src_w", "src_h", "oriented_w", "oriented_h", "crop_box", "scale_factor",
          "categories", "out_size", "out_crc32", "out_sha256"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output root (same <out> as gen_fakes.py)")
    p.add_argument("--subdir", default="wikimedia_2026")
    p.add_argument("--contact", default=os.environ.get("WIKIMEDIA_CONTACT"),
                   help="email or URL for the User-Agent (Wikimedia policy); or env WIKIMEDIA_CONTACT")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--start", default="2025-11-01")
    p.add_argument("--end", default=None, help="last category date (default: today - 3 days)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cand_per_day", type=int, default=0, help="0 = auto (3x the per-day quota)")
    p.add_argument("--per_user_cap", type=int, default=10)
    p.add_argument("--min_side", type=int, default=CROP, help=f"shortest side (must be >= the {CROP} crop)")
    p.add_argument("--max_mpix", type=float, default=60.0)
    p.add_argument("--max_aspect", type=float, default=2.0)
    p.add_argument("--max_mb", type=float, default=40.0)
    p.add_argument("--licenses", default=DEFAULT_LICENSES, help="regex on the Commons machine license code")
    p.add_argument("--c2pa", choices=["reject_any", "reject_ai"], default="reject_any",
                   help="reject every file with a C2PA manifest (default), or only manifests that declare AI")
    p.add_argument("--api_sleep", type=float, default=0.25)
    p.add_argument("--dl_sleep", type=float, default=1.0, help="pause after each original-file download")
    p.add_argument("--metadata_only", type=int, default=0,
                   help="dry run: accept this many records from spread-out days, print them, no full run")
    p.add_argument("--dry_download", action="store_true",
                   help="with --metadata_only: also download, check and process them into <subdir>_dryrun/")
    a = p.parse_args()
    if a.min_side < CROP:
        p.error(f"--min_side must be >= {CROP} (the native center crop)")
    return a


class Commons:
    def __init__(self, contact, api_sleep):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = (f"SAFAID-postcutoff-reals/1.1 (research data collection; {contact}) "
                                        f"python-requests/{requests.__version__}")
        self.sleep = api_sleep
        self.calls = 0
        self.records_fetched = 0

    def get(self, url, stream=False, **params):
        for attempt in range(10):
            try:
                r = self.s.get(url, params=params or None, timeout=120, stream=stream)
            except requests.RequestException as e:
                wait = min(120, 2 ** attempt)
                print(f"  network error {type(e).__name__}; retry in {wait}s", flush=True)
                time.sleep(wait)
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                try:
                    wait = int(r.headers.get("Retry-After", 0) or 0)
                except ValueError:
                    wait = 0
                wait = min(300, wait or 2 ** attempt * 2)
                print(f"  HTTP {r.status_code}; retry in {wait}s", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        raise RuntimeError(f"giving up on {url} {params}")

    def api(self, **params):
        params.update(format="json", formatversion="2", maxlag="5")
        while True:
            self.calls += 1
            j = self.get(API, **params).json()
            time.sleep(self.sleep)
            if j.get("error", {}).get("code") == "maxlag":
                time.sleep(5)
                continue
            if "error" in j:
                raise RuntimeError(j["error"])
            return j

    def day_members(self, day, limit=None):
        """(pageid, title) of the files directly in Category:Photographs taken on <day>."""
        out, cont = [], {}
        while True:
            j = self.api(action="query", list="categorymembers", cmtitle=f"Category:Photographs taken on {day}",
                         cmtype="file", cmlimit=str(min(500, limit or 500)), cmprop="ids|title", **cont)
            out += [(m["pageid"], m["title"]) for m in j["query"]["categorymembers"]]
            if limit and len(out) >= limit:
                return out[:limit]
            if "continue" not in j:
                return out
            cont = {"cmcontinue": j["continue"]["cmcontinue"]}

    def file_info(self, pageids):
        """imageinfo + all categories for up to 50 pageids (follows clcontinue)."""
        pages, cont = {}, {}
        while True:
            j = self.api(action="query", pageids="|".join(map(str, pageids)), prop="imageinfo|categories",
                         iiprop="url|size|mime|sha1|timestamp|user|extmetadata|commonmetadata",
                         iiextmetadatafilter=EXT_FIELDS, iiextmetadatalanguage="en", cllimit="max", **cont)
            for p in j["query"]["pages"]:
                q = pages.setdefault(p["pageid"], {"pageid": p["pageid"], "title": p.get("title"), "categories": []})
                if "imageinfo" in p:
                    q["imageinfo"] = p["imageinfo"][0]
                q["categories"] += [c["title"] for c in p.get("categories", [])]
            if "continue" not in j:
                break
            cont = {k: v for k, v in j["continue"].items()}
        self.records_fetched += len(pageids)
        return [pages[i] for i in pageids if i in pages]


def strip_html(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def exif_date(s):
    m = re.match(r"\s*(\d{4})[:\-](\d{2})[:\-](\d{2})", str(s or ""))
    if not m:
        return None
    try:
        return dt.date(int(m[1]), int(m[2]), int(m[3]))
    except ValueError:
        return None


def num(v):
    """EXIF number from the API ('1/250', 5.6, [{'value': 100}]) or Pillow (IFDRational, tuple). None if absent."""
    if v is None or v == "":
        return None
    if isinstance(v, (list, tuple)):
        if not v:
            return None
        v = v[0]
        if isinstance(v, dict):
            v = v.get("value")
        return num(v)
    if isinstance(v, dict):
        return num(v.get("value"))
    if isinstance(v, str):
        s = v.strip()
        m = re.fullmatch(r"(-?[\d.]+)\s*/\s*(-?[\d.]+)", s)
        try:
            if m:
                d = float(m[2])
                return float(m[1]) / d if d else float("nan")
            return float(s)
        except ValueError:
            return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")


def camera_check(make, model, exposure):
    """Make or Model required; every exposure field that is present must be a positive finite number."""
    if not (str(make or "").strip() or str(model or "").strip()):
        return "no_make_model"
    for k, v in exposure.items():
        if v is not None and not (v == v and 0 < v < float("inf")):
            return f"zero_{k}"
    return None


def evaluate(page, day, args, lic_re):
    """API-level filter. Return (record, None) if the file passes, else (None, reason)."""
    ii = page.get("imageinfo")
    if not ii:
        return None, "no_imageinfo"
    if ii.get("mime") != "image/jpeg":
        return None, "not_jpeg"
    w, h, nbytes = ii.get("width", 0), ii.get("height", 0), ii.get("size", 0)
    if min(w, h) < args.min_side:
        return None, "too_small"
    if w * h > args.max_mpix * 1e6 or nbytes > args.max_mb * 1e6:
        return None, "too_large"
    if max(w, h) / max(1, min(w, h)) > args.max_aspect:
        return None, "aspect"
    ext = {k: (v.get("value") if isinstance(v, dict) else v) for k, v in (ii.get("extmetadata") or {}).items()}
    cm = {m["name"]: m["value"] for m in ii.get("commonmetadata") or [] if "name" in m}
    code = (ext.get("License") or "").strip().lower()
    if not lic_re.match(code):
        return None, "license"
    if (ext.get("Restrictions") or "").strip():
        return None, "restrictions"
    text = " ".join([page.get("title") or "", " ".join(page["categories"]), strip_html(ext.get("ImageDescription"))[:2000]])
    if BAD_TEXT.search(text):
        return None, "ai_or_nonphoto_text"
    if BAD_SOFTWARE.search(str(cm.get("Software", ""))):
        return None, "software"
    make, model = cm.get("Make"), cm.get("Model")
    make = make if isinstance(make, str) else ""
    model = model if isinstance(model, str) else ""
    expo = {k: num(cm.get(k)) for k in EXPOSURE_API}
    why = camera_check(make, model, expo)
    if why:
        return None, why
    d_exif = exif_date(cm.get("DateTimeOriginal"))
    start = dt.date.fromisoformat(args.start)
    cat_day = dt.date.fromisoformat(day)
    if d_exif is None:
        return None, "no_exif_date"
    if d_exif < start:
        return None, "exif_before_start"
    if abs((d_exif - cat_day).days) > 1:
        return None, "exif_category_mismatch"
    up = dt.date.fromisoformat(ii["timestamp"][:10])
    if d_exif > up:
        return None, "exif_after_upload"
    rec = {
        "pageid": page["pageid"], "title": page["title"], "page_url": ii.get("descriptionurl"),
        "file_url": ii.get("url"), "sha1": ii.get("sha1"), "width": w, "height": h, "bytes": nbytes,
        "mime": ii.get("mime"), "license_short": ext.get("LicenseShortName"), "license_code": code,
        "license_url": ext.get("LicenseUrl") or "", "attribution_required": ext.get("AttributionRequired"),
        "author": strip_html(ext.get("Artist")), "author_html": ext.get("Artist") or "",
        "credit": strip_html(ext.get("Credit"))[:500], "uploader": ii.get("user"),
        "upload_timestamp": ii.get("timestamp"), "exif_datetime_original": str(cm.get("DateTimeOriginal")),
        "category_date": day, "camera_make": make.strip(), "camera_model": model.strip(),
        "software": str(cm.get("Software") or ""), "categories": "|".join(c.replace("Category:", "") for c in page["categories"]),
    }
    return rec, None


def jpeg_metadata_segments(data):
    """[(marker, payload)] of the APPn / COM segments before the first scan (SOS)."""
    if data[:2] != b"\xff\xd8":
        return []
    i, out = 2, []
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            break
        m = data[i + 1]
        if m == 0xFF:  # fill byte
            i += 1
            continue
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:  # markers without a length
            i += 2
            continue
        if m in (0xDA, 0xD9):  # start of scan / end of image: no metadata after this point
            break
        (n,) = struct.unpack(">H", data[i + 2:i + 4])
        if 0xE0 <= m <= 0xEF or m == 0xFE:
            out.append((m, data[i + 4:i + 2 + n]))
        i += 2 + n
    return out


def provenance_check(data, c2pa_policy):
    """Inspect the metadata segments of the original JPEG. Returns (reason or None, info dict)."""
    segs = jpeg_metadata_segments(data)
    blob = b"\n".join(p for _, p in segs)
    c2pa = any(m == 0xEB and (b"jumb" in p or b"c2pa" in p) for m, p in segs) or b"c2pa" in blob.lower()
    dst = sorted({x.decode("latin-1") for x in DST_RE.findall(blob)})
    info = {"digital_source_type": "|".join(dst), "c2pa": int(c2pa)}
    bad = [d for d in dst if d.lower() in REJECT_DST]
    if bad or b"algorithmicmedia" in blob.lower():
        return "iptc_ai_or_composite_source", info
    text = blob.decode("latin-1").lower()
    if AI_MARKERS.search(text):
        return "ai_marker_in_metadata", info
    if c2pa and c2pa_policy == "reject_any":
        return "c2pa_manifest", info
    return None, info


def file_exif_check(img, args):
    """Camera and date checks on the EXIF stored in the downloaded file. Returns (reason or None, info)."""
    ex = img.getexif()
    sub = ex.get_ifd(0x8769)  # Exif IFD
    make, model, software = (str(ex.get(t) or "").strip("\x00 ") for t in (0x010F, 0x0110, 0x0131))
    expo = {k: num(sub.get(t)) for k, t in EXPOSURE_TAGS.items()}
    info = {"camera_make_file": make, "camera_model_file": model, "exposure_time": expo["ExposureTime"],
            "f_number": expo["FNumber"], "iso": expo["ISOSpeedRatings"], "focal_length": expo["FocalLength"],
            "software_file": software}
    why = camera_check(make, model, expo)
    if why:
        return "file_" + why, info
    if BAD_SOFTWARE.search(software):
        return "file_software", info
    d_file = exif_date(sub.get(0x9003))
    if d_file is None or d_file < dt.date.fromisoformat(args.start):
        return "file_exif_date_before_start", info
    return None, info


def days_between(a, b):
    a, b = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
    return [(a + dt.timedelta(d)).isoformat() for d in range((b - a).days + 1)]


def collect_day(cm, day, want, args, lic_re, stats, batch=20):
    """Shuffle a day's members with a per-day seed and evaluate them in batches until `want` accepted."""
    members = cm.day_members(day)
    rng = random.Random(f"{args.seed}-{day}")
    rng.shuffle(members)
    acc, users_today = [], set()
    for i in range(0, len(members), batch):
        for page in cm.file_info([pid for pid, _ in members[i:i + batch]]):
            rec, why = evaluate(page, day, args, lic_re)
            if rec and rec["uploader"] in users_today:
                rec, why = None, "same_uploader_same_day"
            stats[why or "accepted"] = stats.get(why or "accepted", 0) + 1
            if rec:
                users_today.add(rec["uploader"])
                acc.append(rec)
                if len(acc) >= want:
                    return acc, len(members)
    return acc, len(members)


class Rejected(Exception):
    pass


def download_and_process(cm, rec, dst, args):
    """Download the original, verify it, run the file-level filters, write the eval PNG. Returns extra fields."""
    r = cm.get(rec["file_url"], stream=True)
    data = r.content
    time.sleep(args.dl_sleep)
    if rec.get("sha1") and hashlib.sha1(data).hexdigest() != rec["sha1"]:
        raise ValueError("sha1 mismatch")
    why, pinfo = provenance_check(data, args.c2pa)
    if why:
        raise Rejected(why)
    img = Image.open(io.BytesIO(data))
    if img.format not in ("JPEG", "MPO"):  # MPO = JPEG + extra frames (gain map / depth); frame 0 is the photo
        raise Rejected("file_not_jpeg")
    img.load()  # full decode at native resolution (no draft/DCT scaling)
    why, einfo = file_exif_check(img, args)
    if why:
        raise Rejected(why)
    prep = to_eval_png(img, dst)
    return {**pinfo, **einfo, **prep, "crop_box": " ".join(map(str, prep["crop_box"]))}


def write_row(w, r, fname, extra, dst):
    size, crc, sha = file_digests(dst)
    row = {**r, **{k: v for k, v in extra.items() if k in FIELDS}, "file": fname,
           "out_size": size, "out_crc32": crc, "out_sha256": sha}
    for k in ("camera_make", "camera_model"):  # prefer what the file itself says
        if extra.get(k + "_file"):
            row[k] = extra[k + "_file"]
    w.writerow(row)


def main():
    args = parse_args()
    if not args.contact:
        args.contact = "no contact configured"
        print("WARNING: no --contact / WIKIMEDIA_CONTACT; the Wikimedia User-Agent policy asks for one", flush=True)
    end = args.end or (dt.date.today() - dt.timedelta(days=3)).isoformat()
    days = days_between(args.start, end)
    lic_re = re.compile(args.licenses, re.I)
    root = os.path.join(args.out, args.subdir)
    cm = Commons(args.contact, args.api_sleep)
    stats, post = {}, {}

    if args.metadata_only:  # ---- dry run: a few spread-out days, small batches ----
        k = args.metadata_only
        probe = [days[round(i * (len(days) - 1) / max(1, k - 1))] for i in range(k)]
        droot = root + "_dryrun"
        os.makedirs(os.path.join(droot, "0_real"), exist_ok=True)
        recs, rows = [], []
        for day in probe:
            got, n_members = collect_day(cm, day, 3 if args.dry_download else 1, args, lic_re, stats, batch=5)
            print(f"{day}: {n_members} files in category, {len(got)} passed the API filters", flush=True)
            for r in got:
                if not args.dry_download:
                    recs.append(r)
                    break
                fname = f"wm_{r['pageid']}.png"
                dst = os.path.join(droot, "0_real", fname)
                try:
                    extra = download_and_process(cm, r, dst, args)
                except Exception as e:
                    key = str(e) if isinstance(e, Rejected) else type(e).__name__
                    post[key] = post.get(key, 0) + 1
                    print(f"   rejected after download: {r['title']}: {key}", flush=True)
                    continue
                recs.append(r)
                rows.append((r, fname, extra, dst))
                break
        with open(os.path.join(droot, "dryrun_metadata.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if rows:
            with open(os.path.join(droot, "attribution.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                for r, fname, extra, dst in rows:
                    write_row(w, r, fname, extra, dst)
        for r in recs:
            print(json.dumps({k2: r[k2] for k2 in ("title", "license_short", "author", "exif_datetime_original",
                                                   "category_date", "camera_make", "camera_model", "width", "height")},
                             ensure_ascii=False))
        for r, fname, extra, dst in rows:
            print(f"   {fname}: src {extra['src_w']}x{extra['src_h']} crop {extra['crop_box']} -> 512 "
                  f"(x{1 / extra['scale_factor']:.2f}), DST={extra['digital_source_type'] or '-'} c2pa={extra['c2pa']} "
                  f"exp={extra['exposure_time']} f={extra['f_number']} iso={extra['iso']} fl={extra['focal_length']}")
        print(f"dry run: {len(recs)} accepted of {cm.records_fetched} metadata records ({cm.calls} API calls); "
              f"API filter counts {stats}; post-download rejections {post}; wrote {droot}")
        return

    os.makedirs(os.path.join(root, "0_real"), exist_ok=True)
    cache = os.path.join(root, "_cache")
    os.makedirs(cache, exist_ok=True)
    quota = math.ceil(args.n / len(days))
    want = args.cand_per_day or 3 * quota
    print(f"{len(days)} days {days[0]}..{days[-1]}, quota {quota}/day, collecting up to {want} candidates/day", flush=True)

    # ---- phase 1: per-day candidates (cached; the cache key includes the filter version) ----
    per_day = {}
    for i, day in enumerate(days):
        cp = os.path.join(cache, f"{day}.v2.json")
        if os.path.exists(cp):
            per_day[day] = json.load(open(cp))["candidates"]
            continue
        for attempt in range(3):
            try:
                got, n_members = collect_day(cm, day, want, args, lic_re, stats)
                break
            except Exception as e:  # e.g. persistent 429s: wait, retry, and finally skip the day (not cached)
                print(f"  {day}: {type(e).__name__}: {str(e)[:100]}; retry {attempt + 1}/3 in 60s", flush=True)
                time.sleep(60)
        else:
            print(f"  {day}: skipped after 3 failures", flush=True)
            continue
        json.dump({"day": day, "n_members": n_members, "queried_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                   "candidates": got}, open(cp, "w"), ensure_ascii=False)
        per_day[day] = got
        if i % 10 == 0:
            print(f"  {day}: {n_members} files, {len(got)} candidates; total "
                  f"{sum(map(len, per_day.values()))}; API calls {cm.calls}", flush=True)

    # ---- phase 2: round-robin selection over days with a per-uploader cap ----
    rng = random.Random(args.seed)
    queues = {d: list(c) for d, c in per_day.items() if c}
    order, per_user, seen = [], {}, set()
    while queues:
        ds = list(queues)
        rng.shuffle(ds)
        for d in ds:
            while queues[d]:
                r = queues[d].pop(0)
                if r["pageid"] in seen or per_user.get(r["uploader"], 0) >= args.per_user_cap:
                    continue
                seen.add(r["pageid"])
                per_user[r["uploader"]] = per_user.get(r["uploader"], 0) + 1
                order.append(r)
                break
            if not queues[d]:
                del queues[d]
    print(f"{len(order)} eligible candidates after per-uploader cap; need {args.n}", flush=True)

    # ---- phase 3: download originals, verify, filter again on the file, preprocess (resumable) ----
    csv_path = os.path.join(root, "attribution.csv")
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            rd = csv.DictReader(f)
            if rd.fieldnames != FIELDS:
                sys.exit(f"{csv_path} has an old header; move it away (and 0_real/) to restart")
            done = {int(r["pageid"]) for r in rd if os.path.exists(os.path.join(root, "0_real", r["file"]))}
    rej_path = os.path.join(cache, "rejected_post_download.json")  # so a resume does not re-download them
    rejected = json.load(open(rej_path)) if os.path.exists(rej_path) else {}
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        n_ok = len(done)
        for r in order:
            if n_ok >= args.n:
                break
            if r["pageid"] in done or str(r["pageid"]) in rejected:
                continue
            fname = f"wm_{r['pageid']}.png"
            dst = os.path.join(root, "0_real", fname)
            try:
                extra = download_and_process(cm, r, dst, args)
            except Exception as e:
                key = str(e) if isinstance(e, Rejected) else type(e).__name__ + ": " + str(e)[:60]
                post[key] = post.get(key, 0) + 1
                if isinstance(e, Rejected):
                    rejected[str(r["pageid"])] = key
                    json.dump(rejected, open(rej_path, "w"))
                if os.path.exists(dst):
                    os.remove(dst)
                continue
            write_row(w, r, fname, extra, dst)
            f.flush()
            n_ok += 1
            if n_ok % 50 == 0:
                print(f"  {n_ok}/{args.n} downloaded; post-download rejections so far {post}", flush=True)
    info = {"args": {k: v for k, v in vars(args).items() if k != "contact"}, "days": [days[0], days[-1]],
            "n_days": len(days), "api_filter_counts_this_run": stats, "post_download_rejections": post,
            "post_download_rejections_all_runs": dict(collections.Counter(rejected.values())),
            "n_images": n_ok, "api_calls": cm.calls,
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "preprocess": f"postcutoff_common.to_eval_png: center crop {CROP}x{CROP} at native resolution, "
                          f"Lanczos {CROP}->{OUT_RES} (one 2x downscale), PNG without metadata",
            "source": "Category:Photographs taken on YYYY-MM-DD", "pillow": Image.__version__,
            "reject_digital_source_types": sorted(REJECT_DST), "c2pa_policy": args.c2pa}
    json.dump(info, open(os.path.join(root, "run_info.json"), "w"), indent=1, ensure_ascii=False)
    print(f"done: {n_ok} images in {root}/0_real; attribution in {csv_path}; post-download rejections {post}")
    if n_ok < args.n:
        print(f"WARNING: only {n_ok}/{args.n}; raise --cand_per_day or --per_user_cap, or widen --end")


if __name__ == "__main__":
    main()
