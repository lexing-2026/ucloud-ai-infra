import json, time, torch, triton, triton.language as tl
@triton.jit
def _peak(A, B, Out, ITERS, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BM); offs_n = tl.arange(0, BN); offs_k = tl.arange(0, BK)
    a = tl.load(A + offs_m[:, None] * BK + offs_k[None, :])
    b = tl.load(B + offs_k[:, None] * BN + offs_n[None, :])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, ITERS):
        acc = tl.dot(a, b, acc)
    tl.store(Out + pid * BM * BN + offs_m[:, None] * BN + offs_n[None, :], acc)
d = 'cuda'
torch.manual_seed(0)
A = (torch.randn(512, 128, device=d)).to(torch.float8_e4m3fn).contiguous()
B = (torch.randn(128, 512, device=d)).to(torch.float8_e4m3fn).contiguous()
for BM, BN, BK, w, grid in [(64,128,128,4,16384)]:
    Out = torch.empty(grid * BM * BN, device=d, dtype=torch.float32)
    iters = 40
    f = lambda: _peak[(grid,)](A, B, Out, iters, BM=BM, BN=BN, BK=BK, num_warps=w, num_stages=1)
    f(); torch.cuda.synchronize()
    ts = []
    for _ in range(9):
        t = time.perf_counter(); f(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    ts.sort(); t = ts[4]
    print(json.dumps(dict(tile=[BM, BN, BK], warps=w, ms=round(t * 1e3, 3), TMACs=round(grid * iters * BM * BN * BK / t / 1e12, 1))), flush=True)
    del Out
print('PEAK_END')
