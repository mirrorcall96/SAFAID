"""Shared image processing for the post-cutoff contamination test (matched processing of reals and fakes).

gen_fakes.py and fetch_recent_reals.py both call `to_eval_png()`, so real and fake images go through the
same steps and each class gets exactly ONE 2x Lanczos downscale:

    reals: EXIF orientation -> sRGB -> center crop CROP x CROP at native resolution (no resampling;
           the shortest side must be >= CROP) -> Lanczos resize CROP -> OUT_RES (factor 2.0)
    fakes: generated at exactly CROP x CROP -> (orientation/sRGB/crop are no-ops) -> the same
           Lanczos resize CROP -> OUT_RES (factor 2.0)

Output: RGB PNG with no ancillary chunks (no iCCP/eXIf/tEXt/pHYs/gAMA), written atomically.
Each call returns a dict with the source size, crop box and scale factor; both scripts log it per image.

Standard library + Pillow only, so every venv can import it.
"""
import hashlib
import io
import os
import struct
import zlib

from PIL import Image, ImageCms, ImageOps

CROP = 1024  # native-resolution square: reals are center-cropped to this, fakes are generated at this size
OUT_RES = 512  # final size; CROP / OUT_RES = 2.0 for both classes
RESAMPLE = Image.Resampling.LANCZOS
Image.MAX_IMAGE_PIXELS = 200_000_000  # large camera files are allowed; the fetcher filters by pixel count first

_SRGB = ImageCms.createProfile("sRGB")


def is_srgb_description(desc):
    """True for real sRGB profiles ("sRGB IEC61966-2.1", "sRGB built-in"), False for wide-gamut profiles that only
    borrow the sRGB transfer curve (e.g. Samsung "DCI-P3 D65 Gamut with sRGB Transfer")."""
    d = desc.lower()
    return d.startswith("srgb") and not any(k in d for k in ("p3", "dci", "adobe", "prophoto", "2020"))


def to_srgb(img):
    """Convert an embedded non-sRGB ICC profile (e.g. Display P3, Adobe RGB) to sRGB. Returns (image, note).
    Generated images carry no profile and pass through unchanged. This is a colour transform, not a resample."""
    icc = img.info.get("icc_profile")
    if not icc:
        return img, "none"
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        desc = (ImageCms.getProfileDescription(src) or "").strip()
        if is_srgb_description(desc):
            return img, f"sRGB ({desc})"
        inp = img if img.mode in ("RGB", "CMYK") else img.convert("RGB")
        out = ImageCms.profileToProfile(inp, src, _SRGB, outputMode="RGB")
        return out, f"converted {desc} -> sRGB"
    except Exception as e:  # broken profile: keep the pixels, but record it
        return img, f"icc error ({type(e).__name__}); left as-is"


def to_eval_png(img, path, crop=CROP, size=OUT_RES):
    """The full shared pipeline (see module docstring). Raises ValueError if the image is smaller than crop."""
    src_w, src_h = img.size
    img = ImageOps.exif_transpose(img)  # lossless transpose; a no-op for generated images
    img, icc_note = to_srgb(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if min(w, h) < crop:
        raise ValueError(f"shortest side {min(w, h)} < {crop}")
    left, top = (w - crop) // 2, (h - crop) // 2
    box = (left, top, left + crop, top + crop)
    sq = img.crop(box) if (w, h) != (crop, crop) else img  # integer crop: no resampling
    out = sq.resize((size, size), RESAMPLE, reducing_gap=None)  # the one Lanczos downscale
    out.info = {}  # Pillow would otherwise carry icc_profile/exif/dpi into the PNG
    tmp = path + ".part"
    out.save(tmp, format="PNG")
    os.replace(tmp, path)
    return {"src_w": src_w, "src_h": src_h, "oriented_w": w, "oriented_h": h, "crop_box": list(box), "crop": crop,
            "out_w": size, "out_h": size, "scale_factor": round(crop / size, 6), "resample": "LANCZOS",
            "icc": icc_note}


def file_digests(path):
    """(size_bytes, crc32 hex as in eval_members_manifest.csv, sha256 hex)."""
    crc, sha, n = 0, hashlib.sha256(), 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            crc = zlib.crc32(chunk, crc)
            sha.update(chunk)
            n += len(chunk)
    return n, f"{crc & 0xFFFFFFFF:08x}", sha.hexdigest()


def png_info(path):
    """(width, height, [chunk types]) read from the PNG container itself."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    i, chunks, wh = 8, [], (0, 0)
    while i + 8 <= len(data):
        n, typ = struct.unpack(">I4s", data[i:i + 8])
        t = typ.decode("latin-1")
        if t == "IHDR":
            wh = struct.unpack(">II", data[i + 8:i + 16])
        if not chunks or chunks[-1] != t:
            chunks.append(t)
        i += 12 + n
        if t == "IEND":
            break
    return wh[0], wh[1], chunks
