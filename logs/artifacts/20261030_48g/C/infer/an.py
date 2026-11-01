import gzip, json, glob, re, collections
f = glob.glob('/root/infer/prof/*.trace.json.gz')[0]
d = json.load(gzip.open(f))
ev = d['traceEvents']
cats = collections.Counter(e.get('cat') for e in ev)
print(cats.most_common(12))
k = [e for e in ev if e.get('cat') == 'kernel' and e.get('ph') == 'X']
if k:
    t0 = min(e['ts'] for e in k); t1 = max(e['ts'] + e['dur'] for e in k)
    busy = sum(e['dur'] for e in k)
    print('gpu kernels', len(k), 'span_ms', round((t1 - t0) / 1e3, 1), 'busy_ms', round(busy / 1e3, 1))
    a = collections.defaultdict(lambda: [0, 0.0])
    for e in k:
        n = e['name']
        n = re.sub(r'^_Z\w*?\d+', '', n)[:70]
        a[n][0] += 1; a[n][1] += e['dur']
    for n, (c, s) in sorted(a.items(), key=lambda x: -x[1][1])[:24]:
        print(f'{s/1e3:9.1f}ms {100*s/busy:5.1f}% n={c:5d} {n}')
gaps = [e for e in ev if e.get('cat') in ('gpu_memcpy', 'gpu_memset') and e.get('ph') == 'X']
print('memcpy_ms', round(sum(e['dur'] for e in gaps) / 1e3, 1), len(gaps))
