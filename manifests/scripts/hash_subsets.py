# Fetch each selected member by HTTP range (no full download), verify CRC32, record SHA256.
import csv, struct, zlib, hashlib, time, os
MAN = os.environ.get("SAFAID_MANIFESTS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # release: manifests/ dir
from concurrent.futures import ThreadPoolExecutor
from multipart import MultiRange, ZIP_START
from httpzip import HTTPRangeFile
mr = MultiRange(); [mr._url(i) for i in range(7)]
vf = HTTPRangeFile("https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip")
def rd(src, a, b): return mr._raw(ZIP_START+a, ZIP_START+b) if src != "progan_val" else vf._get(a, b)
def get(r):
    src = r.get("source", "progan_train"); off = int(r["offset"]); cs = int(r["csize"])
    h = rd(src, off, off+29); n, x = struct.unpack("<HH", h[26:30])
    raw = rd(src, off+30+n+x, off+30+n+x+cs-1)
    data = zlib.decompress(raw, -15) if int(r["method"]) == 8 else raw
    assert format(zlib.crc32(data), "08x") == r["crc"] and len(data) == int(r["size"]), r["name"]
    return dict(r, sha256=hashlib.sha256(data).hexdigest())
for f in [os.path.join(MAN, "safaid_train_subset_v2_seed0.csv"), os.path.join(MAN, "safaid_val_subset_v2_seed0.csv")]:
    rows = list(csv.DictReader(open(f))); t = time.time()
    with ThreadPoolExecutor(48) as ex: out = list(ex.map(get, rows))
    fo = f.replace(".csv", "_sha256.csv")
    with open(fo, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    with open(fo.replace(".csv", ".txt"), "w") as fh:   # sha256sum-compatible
        for o in out: fh.write(f"{o['sha256']}  {o['split']}/{o['category']}/{o['label']}/{o['name'].split('/')[-1]}\n")
    print(fo, len(out), f"{time.time()-t:.0f}s", "unique sha:", len({o['sha256'] for o in out}))
