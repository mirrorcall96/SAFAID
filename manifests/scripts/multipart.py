import io, os, urllib.request, zipfile, json, time, sys
MAN = os.environ.get("SAFAID_MANIFESTS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # release: manifests/ dir
BASE = "https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.00%d"
VOL = [10737418240]*6 + [10499493436]
ZIP_START = 32                      # 7z signature header is 32 bytes; coder = Copy
ZIP_LEN = 74924002746               # NextHeaderOffset from the 7z start header
class MultiRange(io.RawIOBase):
    def __init__(self, block=1<<20):
        self.pos=0; self.size=ZIP_LEN; self.block=block; self.cache={}; self.nreq=0; self.urls={}
        self.starts=[]; s=0
        for v in VOL: self.starts.append(s); s+=v
    def _url(self,i):
        if i not in self.urls:
            r=urllib.request.urlopen(urllib.request.Request(BASE%(i+1),headers={"Range":"bytes=0-0"}),timeout=60)
            self.urls[i]=r.geturl()
        return self.urls[i]
    def _raw(self, gs, ge):  # global (concatenated) inclusive range
        out=bytearray()
        for i,(st,ln) in enumerate(zip(self.starts,VOL)):
            a=max(gs,st); b=min(ge,st+ln-1)
            if a<=b:
                for k in range(5):
                    try:
                        r=urllib.request.urlopen(urllib.request.Request(self._url(i),headers={"Range":f"bytes={a-st}-{b-st}"}),timeout=120)
                        out+=r.read(); self.nreq+=1; break
                    except Exception as e:
                        time.sleep(2); 
                        if k==4: raise
        return bytes(out)
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self,o,w=0):
        self.pos = o if w==0 else (self.pos+o if w==1 else self.size+o); return self.pos
    def read(self,n=-1):
        if n is None or n<0: n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0: return b""
        if n>4*self.block:
            d=self._raw(ZIP_START+self.pos, ZIP_START+self.pos+n-1); self.pos+=n; return d
        out=bytearray(); p=self.pos; end=p+n
        while p<end:
            b=p//self.block
            if b not in self.cache:
                s=b*self.block; e=min(self.size,s+self.block)-1
                self.cache[b]=self._raw(ZIP_START+s, ZIP_START+e)
                if len(self.cache)>64: self.cache.pop(next(iter(self.cache)))
            blk=self.cache[b]; o=p-b*self.block; t=min(end-p,len(blk)-o); out+=blk[o:o+t]; p+=t
        self.pos=end; return bytes(out)
    def readinto(self,b):
        d=self.read(len(b)); b[:len(d)]=d; return len(d)
if __name__=="__main__":
    f=MultiRange()
    z=zipfile.ZipFile(io.BufferedReader(f,buffer_size=1<<20))
    rows=[{"name":i.filename,"size":i.file_size,"csize":i.compress_size,"crc":format(i.CRC,"08x"),"method":i.compress_type,"offset":i.header_offset} for i in z.infolist()]
    json.dump({"url":"progan_train.7z.001-007 (inner zip, store-mode 7z)","zip_bytes":ZIP_LEN,"entries":rows},open(os.path.join(MAN,"progan_train_listing.json"),"w"))
    print("entries",len(rows),"requests",f.nreq)
