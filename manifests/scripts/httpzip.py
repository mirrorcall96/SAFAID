import io, urllib.request, zipfile, sys, json, time

class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests (with small block cache)."""
    def __init__(self, url, block=1<<20):
        self.url = url; self.pos = 0; self.block = block; self.cache = {}
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers["Content-Range"]; self.size = int(cr.split("/")[1])
            self.final_url = r.geturl()
        self.nreq = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        if whence == 0: self.pos = off
        elif whence == 1: self.pos += off
        else: self.pos = self.size + off
        return self.pos
    def _get(self, start, end):
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.final_url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    self.nreq += 1
                    return r.read()
            except Exception as e:
                time.sleep(2); err = e
        raise err
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0: return b""
        if n > 4*self.block:
            data = self._get(self.pos, self.pos + n - 1); self.pos += n; return data
        out = bytearray(); p = self.pos; end = p + n
        while p < end:
            b = p // self.block
            if b not in self.cache:
                s = b*self.block; e = min(self.size, s+self.block) - 1
                self.cache[b] = self._get(s, e)
                if len(self.cache) > 64: self.cache.pop(next(iter(self.cache)))
            blk = self.cache[b]; o = p - b*self.block
            take = min(end - p, len(blk) - o); out += blk[o:o+take]; p += take
        self.pos = end; return bytes(out)
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

def list_zip(url, out_json):
    f = HTTPRangeFile(url)
    z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1<<20))
    rows = [{"name": i.filename, "size": i.file_size, "csize": i.compress_size,
             "crc": format(i.CRC, "08x"), "method": i.compress_type, "offset": i.header_offset}
            for i in z.infolist()]
    json.dump({"url": url, "zip_bytes": f.size, "entries": rows}, open(out_json, "w"))
    print(url, "zip bytes", f.size, "entries", len(rows), "http requests", f.nreq)
    return rows

if __name__ == "__main__":
    list_zip(sys.argv[1], sys.argv[2])
