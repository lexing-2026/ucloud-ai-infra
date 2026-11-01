"""Corrected sm89 FP8 mma peak test.

The earlier mma_hand.py never stored its accumulators, so ptxas deleted the whole
mma loop and the '2e4 TMAC/s' numbers were launch overhead of an empty kernel.
This version stores every accumulator to global memory, calibrates launch
overhead with an empty kernel, and checks that time scales with ITERS.
"""
import ctypes, json, subprocess, tempfile, time, torch

PTXAS = "/root/infer/sg22/lib/python3.12/site-packages/nvidia/cu13/bin/ptxas"
torch.cuda.init(); torch.zeros(1, device="cuda")
cu = ctypes.CDLL("libcuda.so.1")
ctx = ctypes.c_void_p(); dev = ctypes.c_int()
assert cu.cuDeviceGet(ctypes.byref(dev), 0) == 0
assert cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev) == 0
assert cu.cuCtxSetCurrent(ctx) == 0
print("gpu", torch.cuda.get_device_name(0), flush=True)


def build(src, name):
    d = tempfile.mkdtemp()
    p, o = f"{d}/k.ptx", f"{d}/k.cubin"
    open(p, "w").write(src)
    subprocess.run([PTXAS, "-arch", "sm_89", "-m64", p, "-o", o], check=True, capture_output=True, text=True)
    img = open(o, "rb").read()
    mod = ctypes.c_void_p(); fn = ctypes.c_void_p()
    rc = cu.cuModuleLoadData(ctypes.byref(mod), img); assert rc == 0, rc
    rc = cu.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()); assert rc == 0, rc
    return fn, img


def launch(fn, grid, block, shmem, args):
    vals, ptrs = [], (ctypes.c_void_p * len(args))()
    for i, a in enumerate(args):
        v = ctypes.c_uint64(a) if a > 0xffffffff else ctypes.c_uint32(a)
        vals.append(v); ptrs[i] = ctypes.cast(ctypes.pointer(v), ctypes.c_void_p)
    s = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = cu.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, shmem, s, ptrs, None); assert rc == 0, rc


HEAD = """.version 8.5
.target sm_89
.address_size 64
"""
MMA = "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"


def empty_kernel(nt):
    return f"""{HEAD}.visible .entry empty(.param .u64 pA, .param .u64 pOut, .param .u32 iters)
.reqntid {nt}
{{
  ret;
}}
"""


def reg_peak(nacc, inner, nt):
    """nacc accumulator tiles, inner mma per iteration, accs stored at the end."""
    nr = 16 + 4 * nacc + 4
    L = [HEAD, f".visible .entry peak(.param .u64 pA, .param .u64 pOut, .param .u32 iters)\n.reqntid {nt}\n{{"]
    L += [f"  .reg .b32 %r<{nr}>;", "  .reg .b64 %d<8>;", "  .reg .pred %p<2>;",
          "  .reg .b32 %t, %i, %n, %u;"]
    L += ["  ld.param.u64 %d0, [pA];", "  ld.param.u64 %d2, [pOut];", "  ld.param.u32 %n, [iters];"]
    L += ["  mov.u32 %t, %tid.x;", "  and.b32 %u, %t, 31;", "  shl.b32 %u, %u, 4;",
          "  cvt.u64.u32 %d1, %u;", "  add.u64 %d0, %d0, %d1;"]
    L += ["  ld.global.v4.b32 {%r0, %r1, %r2, %r3}, [%d0];",
          "  ld.global.v2.b32 {%r4, %r5}, [%d0+1024];"]
    for j in range(4 * nacc):
        L.append(f"  mov.b32 %r{16 + j}, 0;")
    L += ["  mov.u32 %i, 0;", "$L0:"]
    for _ in range(inner):
        for a in range(nacc):
            b = 16 + 4 * a
            q = f"%r{b}, %r{b+1}, %r{b+2}, %r{b+3}"
            L.append(f"  {MMA} {{ {q} }}, {{ %r0, %r1, %r2, %r3 }}, {{ %r4, %r5 }}, {{ {q} }};")
    L += ["  add.s32 %i, %i, 1;", "  setp.lt.s32 %p0, %i, %n;", "  @%p0 bra $L0;"]
    # out offset = (ctaid.x * nt + tid) * (4*nacc) * 4B
    L += ["  mov.u32 %u, %ctaid.x;", "  cvt.u64.u32 %d3, %u;",
          f"  mul.lo.u64 %d3, %d3, {nt};", "  cvt.u64.u32 %d4, %t;", "  add.u64 %d3, %d3, %d4;",
          f"  mul.lo.u64 %d3, %d3, {16 * nacc};", "  add.u64 %d2, %d2, %d3;"]
    for a in range(nacc):
        b = 16 + 4 * a
        L.append(f"  st.global.v4.b32 [%d2+{16 * a}], {{%r{b}, %r{b+1}, %r{b+2}, %r{b+3}}};")
    L += ["  ret;", "}"]
    return "\n".join(L)


def timeit(f, reps=9):
    f(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2]


Abuf = torch.randn(1 << 16, device="cuda").to(torch.float8_e4m3fn).contiguous()
rows = []
for nacc, inner, nt, grid, iters in [(16, 4, 128, 16384, 40), (16, 4, 256, 8192, 40),
                                     (4, 16, 128, 16384, 40), (1, 64, 128, 16384, 40),
                                     (16, 4, 64, 32768, 40), (16, 4, 32, 65536, 40),
                                     (8, 8, 128, 16384, 40)]:
    out = torch.empty(grid * nt * 4 * nacc, device="cuda", dtype=torch.float32)
    fn, img = build(reg_peak(nacc, inner, nt), "peak")
    ms = timeit(lambda: launch(fn, grid, nt, 0, [Abuf.data_ptr(), out.data_ptr(), iters]))
    macs = grid * (nt // 32) * nacc * inner * iters * 4096
    rows.append(dict(variant=f"reg_persistent nacc={nacc}", inner=inner, threads=nt, grid=grid,
                     iters=iters, ms=round(ms * 1e3, 3), TMACs=round(macs / ms / 1e12, 1)))
    print(json.dumps(rows[-1]), flush=True)
    del out

# empty kernel calibration at the largest grid used
for nt, grid in [(128, 16384), (256, 8192)]:
    out = torch.empty(16, device="cuda", dtype=torch.float32)
    fn, img = build(empty_kernel(nt), "empty")
    ms = timeit(lambda: launch(fn, grid, nt, 0, [Abuf.data_ptr(), out.data_ptr(), 0]))
    print(json.dumps(dict(variant="empty", threads=nt, grid=grid, ms=round(ms * 1e3, 3))), flush=True)

# scaling check: same kernel, 1 vs 40 iterations
out = torch.empty(16384 * 128 * 64, device="cuda", dtype=torch.float32)
fn, img = build(reg_peak(16, 4, 128), "peak")
for iters in (1, 10, 40, 80):
    ms = timeit(lambda: launch(fn, 16384, 128, 0, [Abuf.data_ptr(), out.data_ptr(), iters]))
    print(json.dumps(dict(variant="scaling", iters=iters, ms=round(ms * 1e3, 3))), flush=True)
print("PEAK2_END")
