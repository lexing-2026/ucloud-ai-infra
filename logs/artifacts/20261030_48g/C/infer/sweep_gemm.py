import json, itertools, torch
import sglang.kernels.ops.gemm.fp8_kernel as fk

dev = "cuda"
BLOCK = [128, 128]
torch.manual_seed(0)

def bench(N, K, M, cfgs, iters=50):
    B = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn)
    Bs = (torch.rand(N // 128, K // 128, device=dev) * 0.02 + 0.01).float()
    A = torch.randn(M, K, device=dev).to(torch.float8_e4m3fn)
    As = (torch.rand(M, K // 128, device=dev) * 0.02 + 0.01).float()
    ref = None
    out_rows = []
    orig = fk.get_w8a8_block_fp8_configs
    for cfg in cfgs:
        fk.get_w8a8_block_fp8_configs = lambda *a, _c=cfg: {M: _c}
        try:
            out = fk.w8a8_block_fp8_matmul(A, B, As, Bs, BLOCK, torch.bfloat16)
            torch.cuda.synchronize()
            if ref is None:
                Ad = (A.float().view(M, K // 128, 128) * As[..., None]).reshape(M, K)
                Bd = (B.float().view(N // 128, 128, K // 128, 128) * Bs[:, None, :, None]).reshape(N, K)
                ref = Ad @ Bd.T
            err = ((out.float() - ref).abs().mean() / ref.abs().mean()).item()
            for _ in range(5): fk.w8a8_block_fp8_matmul(A, B, As, Bs, BLOCK, torch.bfloat16)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(iters): fk.w8a8_block_fp8_matmul(A, B, As, Bs, BLOCK, torch.bfloat16)
            e.record(); torch.cuda.synchronize()
            ms = s.elapsed_time(e) / iters
            out_rows.append(dict(cfg=cfg, ms=round(ms, 4), weight_gbs=round(N * K / ms / 1e6, 1),
                                 tflops=round(2 * M * N * K / ms / 1e9, 1), err=round(err, 5)))
        except Exception as ex:
            out_rows.append(dict(cfg=cfg, error=str(ex)[:80]))
    fk.get_w8a8_block_fp8_configs = orig
    out_rows.sort(key=lambda r: r.get("ms", 9e9))
    for r in out_rows[:6]:
        print(json.dumps(dict(N=N, K=K, M=M, **r)), flush=True)
    del B, Bs, A, As, ref

base = lambda bm, bn, w, st: dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=128,
                                  GROUP_SIZE_M=32, num_warps=w, num_stages=st)
cfgs = []
for bm in (16, 32, 64):
    for bn in (32, 64, 128):
        for w in (4, 8):
            for st in (3, 4):
                cfgs.append(base(bm, bn, w, st))
for sk in (2, 4):
    for bn in (32, 64, 128):
        cfgs.append({**base(32, bn, 4, 3), "SPLIT_K": sk})

print("=== N=5120 K=6144 M=72 ===", flush=True)
bench(5120, 6144, 72, cfgs)
print("=== N=5120 K=17408 M=72 ===", flush=True)
bench(5120, 17408, 72, cfgs)
print("=== N=5120 K=6144 M=5 ===", flush=True)
bench(5120, 6144, 5, cfgs)
print("SWEEP_END")
