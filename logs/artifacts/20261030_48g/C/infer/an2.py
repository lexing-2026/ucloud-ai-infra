import gzip, json, glob, collections
d = json.load(gzip.open(glob.glob('/root/infer/prof/*.trace.json.gz')[0]))
k = sorted([e for e in d['traceEvents'] if e.get('cat') == 'kernel' and e.get('ph') == 'X' and 'fp8_matmul' in e['name']], key=lambda e: e['ts'])
ds = [round(e['dur'] / 1e3, 1) for e in k]
print('first 24 durations ms', ds[:24])
c = collections.Counter(round(x) for x in ds)
print(sorted(c.items()))
print(json.dumps({x: k[0].get('args', {}).get(x) for x in list(k[0].get('args', {}))[:12]})[:400])
