import ctypes, json, subprocess, tempfile, time, torch
PTXAS = "/root/infer/sg22/lib/python3.12/site-packages/nvidia/cu13/bin/ptxas"
torch.cuda.init(); torch.zeros(1, device="cuda")
cu = ctypes.CDLL("libcuda.so.1")
ctx = ctypes.c_void_p(); dev = ctypes.c_int()
cu.cuDeviceGet(ctypes.byref(dev), 0); cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev); cu.cuCtxSetCurrent(ctx)

def build(src, name):
    d = tempfile.mkdtemp(); p, o = f"{d}/k.ptx", f"{d}/k.cubin"
    open(p, "w").write(src)
    r = subprocess.run([PTXAS, "-arch", "sm_89", "-m64", p, "-o", o], capture_output=True, text=True)
    if r.returncode:
        print("PTXAS_ERROR", r.stderr.strip().splitlines()[-1][:160], flush=True); return None, None
    img = open(o, "rb").read(); mod = ctypes.c_void_p(); fn = ctypes.c_void_p()
    assert cu.cuModuleLoadData(ctypes.byref(mod), img) == 0
    assert cu.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()) == 0
    return fn, img

def launch(fn, grid, block, args):
    vals, ptrs = [], (ctypes.c_void_p * len(args))()
    for i, a in enumerate(args):
        v = ctypes.c_uint64(a) if a > 0xffffffff else ctypes.c_uint32(a)
        vals.append(v); ptrs[i] = ctypes.cast(ctypes.pointer(v), ctypes.c_void_p)
    s = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    assert cu.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, s, ptrs, None) == 0

HEAD = ".version 8.5\n.target sm_89\n.address_size 64\n"

def reg_peak(mma, nreg, nacc, inner, nt):
    nr = 16 + nreg * nacc + 4
    L = [HEAD, f".visible .entry peak(.param .u64 pA, .param .u64 pOut, .param .u32 iters)\n.reqntid {nt}\n{{"]
    L += [f"  .reg .b32 %r<{nr}>;", "  .reg .b64 %d<8>;", "  .reg .pred %p<2>;", "  .reg .b32 %t, %i, %n, %u;"]
    L += ["  ld.param.u64 %d0, [pA];", "  ld.param.u64 %d2, [pOut];", "  ld.param.u32 %n, [iters];",
          "  mov.u32 %t, %tid.x;", "  and.b32 %u, %t, 31;", "  shl.b32 %u, %u, 4;",
          "  cvt.u64.u32 %d1, %u;", "  add.u64 %d0, %d0, %d1;",
          "  ld.global.v4.b32 {%r0, %r1, %r2, %r3}, [%d0];",
          "  ld.global.v2.b32 {%r4, %r5}, [%d0+1024];"]
    for j in range(nreg * nacc):
        L.append(f"  mov.b32 %r{16 + j}, 0;")
    L += ["  mov.u32 %i, 0;", "$L0:"]
    for _ in range(inner):
        for a in range(nacc):
            b = 16 + nreg * a
            q = ", ".join(f"%r{b + j}" for j in range(nreg))
            L.append(f"  {mma} {{ {q} }}, {{ %r0, %r1, %r2, %r3 }}, {{ %r4, %r5 }}, {{ {q} }};")
    L += ["  add.s32 %i, %i, 1;", "  setp.lt.s32 %p0, %i, %n;", "  @%p0 bra $L0;",
          "  mov.u32 %u, %ctaid.x;", "  cvt.u64.u32 %d3, %u;", f"  mul.lo.u64 %d3, %d3, {nt};",
          "  cvt.u64.u32 %d4, %t;", "  add.u64 %d3, %d3, %d4;",
          f"  mul.lo.u64 %d3, %d3, {4 * nreg * nacc};", "  add.u64 %d2, %d2, %d3;"]
    for a in range(nacc):
        b = 16 + nreg * a
        v = "v4" if nreg == 4 else "v2"
        regs = ", ".join(f"%r{b + j}" for j in range(nreg))
        L.append(f"  st.global.{v}.b32 [%d2+{4 * nreg * a}], {{{regs}}};")
    L += ["  ret;", "}"]
    return "\n".join(L)

def timeit(f, reps=8):
    f(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2]

Abuf = torch.randn(1 << 16, device="cuda").to(torch.float8_e4m3fn).contiguous()
tests = {
  "fp8_f32_k32": ("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32", 4, 4096),
  "fp16_f32_k16": ("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", 4, 2048),
  "fp16_f32_k32_bf16": ("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", 4, 2048),
  "fp8_f16_k32": ("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16", 2, 4096),
}
for name, (mma, nreg, macs) in tests.items():
    fn, img = build(reg_peak(mma, nreg, 8, 8, 128), "peak")
    if fn is None: continue
    grid, iters = 16384, 100
    out = torch.empty(grid * 128 * 4 * nreg * 8, device="cuda", dtype=torch.float32)
    ms = timeit(lambda: launch(fn, grid, 128, [Abuf.data_ptr(), out.data_ptr(), iters]))
    total = grid * 4 * 8 * 8 * iters * macs
    print(json.dumps(dict(variant=name, ms=round(ms*1e3, 3), TMACs=round(total/ms/1e12, 1),
                          TFLOPS=round(2*total/ms/1e12, 1))), flush=True)
print("PEAK4_END")
