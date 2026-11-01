import json, torch
import sglang.kernels.ops.gemm.fp8_kernel as fk

dev = "cuda"; BLOCK = [128, 128]; torch.manual_seed(0)

def bench(N, K, M, cfgs, iters=40):
    B = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn)
    Bs = (torch.rand(N // 128, K // 128, device=dev) * 0.02 + 0.01).float()
    A = torch.randn(M, K, device=dev).to(torch.float8_e4m3fn)
    As = (torch.rand(M, K // 128, device=dev) * 0.02 + 0.01).float()
    orig = fk.get_w8a8_block_fp8_configs
    best = None; rows = []
    for cfg in cfgs:
        fk.get_w8a8_block_fp8_configs = lambda *a, _c=cfg: {M: _c}
        try:
            out = fk.w8a8_block_fp8_matmul(A, B, As, Bs, BLOCK, torch.bfloat16)
            torch.cuda.synchronize()
            for _ in range(4): fk.w8a8_block_fp8_matmul(A, B, As, Bs, BLOCK, torch.bfloat16)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(iters): fk.w8a8_block_fp8_matmul(A, B, As, Bs, BLOCK, torch.bfloat16)
            e.record(); torch.cuda.synchronize()
            ms = s.elapsed_time(e) / iters
            rows.append((ms, cfg))
        except Exception:
            pass
    fk.get_w8a8_block_fp8_configs = orig
    rows.sort(key=lambda r: r[0])
    # baseline: what the installed JSON picks for this M
    real = fk.w8a8_block_fp8_matmul(A, B, As, Bs, BLOCK, torch.bfloat16)
    torch.cuda.synchronize()
    for _ in range(4): fk.w8a8_block_fp8_matmul(A, B, As, Bs, BLOCK, torch.bfloat16)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fk.w8a8_block_fp8_matmul(A, B, As, Bs, BLOCK, torch.bfloat16)
    e.record(); torch.cuda.synchronize()
    base_ms = s.elapsed_time(e) / iters
    print(json.dumps(dict(N=N, K=K, M=M, tuned_ms=round(base_ms, 4),
                          best_ms=round(rows[0][0], 4),
                          gain=round(base_ms / rows[0][0] - 1, 3),
                          best_cfg={k: rows[0][1].get(k) for k in ('BLOCK_SIZE_M','BLOCK_SIZE_N','BLOCK_SIZE_K','num_warps','num_stages','SPLIT_K','SWAP_AB')},
                          gbs=round(N*K/rows[0][0]/1e6, 1))), flush=True)
    del B, Bs, A, As, out

cfgs = []
for bm in (16, 32, 64):
    for bn in (64, 128):
        for st in (3, 4):
            cfgs.append(dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=128,
                             GROUP_SIZE_M=32, num_warps=4, num_stages=st))
for extra in (dict(SPLIT_K=2), dict(SPLIT_K=4)):
    for bm in (32, 64):
        for bn in (64, 128):
            cfgs.append({**dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=128,
                                GROUP_SIZE_M=32, num_warps=4, num_stages=3), **extra})

for N, K in [(34816,5120),(16384,5120),(14336,5120),(5120,17408),(5120,6144)]:
    for M in (9, 72):
        bench(N, K, M, cfgs)
print("SWEEP2_END")
