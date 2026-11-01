"""Hand-written PTX on sm_89: raw mma.sync rate with operands in registers, then with ldmatrix per mma."""
import ctypes, json, os, subprocess, tempfile, time, torch
PTXAS = "/root/infer/sg22/lib/python3.12/site-packages/nvidia/cu13/bin/ptxas"
torch.cuda.init(); torch.zeros(1, device="cuda")
cu = ctypes.CDLL("libcuda.so.1")
ctx = ctypes.c_void_p(); dev = ctypes.c_int()
assert cu.cuDeviceGet(ctypes.byref(dev), 0) == 0
assert cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev) == 0
assert cu.cuCtxSetCurrent(ctx) == 0

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
    vals = []
    ptrs = (ctypes.c_void_p * len(args))()
    for i, a in enumerate(args):
        v = ctypes.c_uint64(a) if a > 0xffffffff else ctypes.c_uint32(a)
        vals.append(v); ptrs[i] = ctypes.cast(ctypes.pointer(v), ctypes.c_void_p)
    s = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = cu.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, shmem, s, ptrs, None); assert rc == 0, rc

HEAD = """.version 8.5
.target sm_89
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
"""
MMA = "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"

def reg_peak(nacc, inner, nt):
    """registers only: nacc accumulator tiles cycled, operands fixed."""
    L = [HEAD, f".visible .entry peak(.param .u64 pA, .param .u32 iters)\n.reqntid {nt}\n{{"]
    L += [f"  .reg .b32 %r<{8 + 4 * nacc + 8}>;", "  .reg .b64 %d<4>;", "  .reg .pred %p<2>;", "  .reg .b32 %t, %i, %n;"]
    L += ["  ld.param.u64 %d0, [pA];", "  ld.param.u32 %n, [iters];", "  mov.u32 %t, %tid.x;", "  and.b32 %t, %t, 31;", "  shl.b32 %t, %t, 4;", "  cvt.u64.u32 %d1, %t;", "  add.u64 %d0, %d0, %d1;"]
    L += ["  ld.global.v4.b32 {%r0, %r1, %r2, %r3}, [%d0];", "  ld.global.v2.b32 {%r4, %r5}, [%d0+1024];"]
    for j in range(4 * nacc):
        L.append(f"  mov.b32 %r{8 + j}, 0;")
    L += ["  mov.u32 %i, 0;", "$L0:"]
    for _ in range(inner):
        for a in range(nacc):
            b = 8 + 4 * a
            q = f"%r{b}, %r{b+1}, %r{b+2}, %r{b+3}"
            L.append(f"  {MMA} {{ {q} }}, {{ %r0, %r1, %r2, %r3 }}, {{ %r4, %r5 }}, {{ {q} }};")
    L += ["  add.s32 %i, %i, 1;", "  setp.lt.s32 %p0, %i, %n;", "  @%p0 bra $L0;", "  ret;", "}"]
    return "\n".join(L)

d = "cuda"
Abuf = torch.randn(1 << 16, device=d).to(torch.float8_e4m3fn).contiguous()
res = []
for nacc, inner, nt, grid, iters in [(16, 4, 128, 16384, 40), (16, 4, 256, 8192, 40), (4, 16, 128, 16384, 40), (1, 64, 128, 16384, 40), (16, 4, 64, 32768, 40), (16, 4, 32, 65536, 40)]:
    fn, img = build(reg_peak(nacc, inner, nt), "peak")
    f = lambda: launch(fn, grid, nt, 0, [Abuf.data_ptr(), iters])
    f(); torch.cuda.synchronize()
    ts = []
    for _ in range(9):
        t = time.perf_counter(); f(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    ts.sort(); t = ts[4]
    mma_per_warp = nacc * inner * iters
    total = grid * (nt // 32) * mma_per_warp * 4096
    print(json.dumps(dict(variant="reg", nacc=nacc, inner=inner, threads=nt, ms=round(t * 1e3, 3), TMACs=round(total / t / 1e12, 1))), flush=True)
print("HAND_END")
