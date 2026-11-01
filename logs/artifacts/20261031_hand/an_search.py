import json,sys,collections
rows=[];refs=[]
for l in open(sys.argv[1]):
    if l.startswith('{"r"'):
        d=json.loads(l); rows.append((d["r"]["ms"],d["r"]["mhz"],d["kw"]))
    elif l.startswith('{"ref"'):
        refs.append(json.loads(l)["ref"]["ms"])
rows.sort(key=lambda x:x[0])
print(len(rows),"refs",refs)
for r in rows[:7]: print(r)
for key in ("BM","opt","NB","G","gap","cps","bpos"):
    c=collections.defaultdict(list)
    for r in rows: c[r[2][key]].append(r[0])
    print(key,{k:round(sorted(v)[len(v)//2],2) for k,v in sorted(c.items())})
