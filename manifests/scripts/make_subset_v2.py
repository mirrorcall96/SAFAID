# Deterministic SAFAID train/val subsets from the official CNNDetection ProGAN zips (HF sywang/CNNDetection).
# Train pool: PNGs whose (crc32,size) is unique within train AND absent from progan_val.zip and CNN_synth_testset.zip.
# Val: official progan_val.zip minus files duplicated in train/test; a cell with < n clean files is topped up
#      from the clean train pool (disjoint from the training picks). Needed for val cat/0_real (all 200 are in the test set).
import json, random, collections, csv, os
MAN = os.environ.get("SAFAID_MANIFESTS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # release: manifests/ dir
CATS=["airplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
L=lambda f:[e for e in json.load(open(f))["entries"] if e["name"].lower().endswith(".png")]
tr,va,te=L(os.path.join(MAN,"progan_train_listing.json")),L(os.path.join(MAN,"progan_val_listing.json")),L(os.path.join(MAN,"cnn_synth_testset_listing.json"))
key=lambda e:(e["crc"],e["size"])
trc=collections.Counter(map(key,tr)); vak=set(map(key,va)); tek=set(map(key,te))
def cells(E, banned):
    by=collections.defaultdict(list)
    for e in E:
        c,lab,_=e["name"].split("/")
        if key(e) not in banned: by[(c,lab)].append(e)
    for k in by: by[k].sort(key=lambda e:e["name"])
    return by
def sample(pool, n, tag): return sorted(random.Random(tag).sample(pool, n), key=lambda e:e["name"])
trpool=cells(tr, {k for k,v in trc.items() if v>1} | vak | tek)
vapool=cells(va, set(trc) | tek)
R,V=[],[]
for c in CATS:
    for lab in ("0_real","1_fake"):
        picks=sample(trpool[(c,lab)],100,f"safaid-train-seed0-{c}-{lab}")
        R+=[dict(split="train",source="progan_train",category=c,label=lab,**e) for e in picks]
        vp=vapool[(c,lab)]
        if len(vp)>=20: vs=[("progan_val",e) for e in sample(vp,20,f"safaid-val-seed0-{c}-{lab}")]
        else:
            chosen={e["name"] for e in picks}
            rest=[e for e in trpool[(c,lab)] if e["name"] not in chosen]
            vs=[("progan_val",e) for e in vp]+[("progan_train_heldout",e) for e in sample(rest,20-len(vp),f"safaid-valtopup-seed0-{c}-{lab}")]
            print("val top-up:",c,lab,"official clean =",len(vp))
        V+=[dict(split="val",source=s,category=c,label=lab,**e) for s,e in vs]
F=["split","source","category","label","name","size","crc","offset","csize","method"]
for rows,f in [(R,os.path.join(MAN,"safaid_train_subset_v2_seed0.csv")),(V,os.path.join(MAN,"safaid_val_subset_v2_seed0.csv"))]:
    with open(f,"w",newline="") as fh: w=csv.DictWriter(fh,fieldnames=F); w.writeheader(); w.writerows(rows)
    print(f,len(rows),"files",sum(r["size"] for r in rows),"bytes")
assert not ({r["name"] for r in R} & {r["name"] for r in V if r["source"]!="progan_val"})
