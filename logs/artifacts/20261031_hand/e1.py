"""E1: can FP32 issue while the tensor core is busy? mma stream with N fillers per mma."""
import ctypes, json, subprocess, tempfile, time, threading, torch, sys
PTXAS = "/root/infer/sg22/lib/python3.12/site-packages/nvidia/cu13/bin/ptxas"
torch.cuda.init(); torch.zeros(1, device="cuda")
cu = ctypes.CDLL("libcuda.so.1")
ctx = ctypes.c_void_p(); dev = ctypes.c_int()
cu.cuDeviceGet(ctypes.byref(dev), 0); cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev); cu.cuCtxSetCurrent(ctx)
def build(src, name):
    d = tempfile.mkdtemp(); p, o = f"{d}/k.ptx", f"{d}/k.cubin"
    open(p, "w").write(src)
    r = subprocess.run([PTXAS, "-arch", "sm_89", "-m64", p, "-o", o], capture_output=True, text=True)
    if r.returncode: raise SystemExit(r.stderr[-400:])
    img = open(o, "rb").read(); mod = ctypes.c_void_p(); fn = ctypes.c_void_p()
    assert cu.cuModuleLoadData(ctypes.byref(mod), img) == 0
    assert cu.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()) == 0
    return fn, o
def launch(fn, grid, block, args, smem=0):
    vals, ptrs = [], (ctypes.c_void_p * len(args))()
    for i, a in enumerate(args):
        v = ctypes.c_uint64(a) if a > 0xffffffff else ctypes.c_uint32(a)
        vals.append(v); ptrs[i] = ctypes.cast(ctypes.pointer(v), ctypes.c_void_p)
    s = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    r = cu.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, smem, s, ptrs, None); assert r == 0, r
HEAD = ".version 8.5\n.target sm_89\n.address_size 64\n.extern .shared .align 16 .b8 smem[];\n"
MMA = "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"
def src(nacc, fill, nt, kind):
    nr = 8 + 4 * nacc + max(fill, 4)
    L = [HEAD, f".visible .entry peak(.param .u64 pA, .param .u64 pOut, .param .u32 iters)\n.reqntid {nt}\n{{",
         f"  .reg .b32 %r<{nr + 8}>;", "  .reg .b64 %d<8>;", "  .reg .pred %p<2>;", "  .reg .b32 %t, %i, %n, %u;",
         "  ld.param.u64 %d0, [pA];", "  ld.param.u64 %d2, [pOut];", "  ld.param.u32 %n, [iters];",
         "  mov.u32 %t, %tid.x;", "  and.b32 %u, %t, 31;", "  shl.b32 %u, %u, 4;", "  cvt.u64.u32 %d1, %u;",
         "  add.u64 %d0, %d0, %d1;", "  ld.global.v4.b32 {%r0, %r1, %r2, %r3}, [%d0];",
         "  ld.global.v2.b32 {%r4, %r5}, [%d0+1024];",
         "  mov.u64 %d5, smem;", "  st.shared.b32 [%d5], %t;", "  ld.shared.b32 %r6, [%d5];"]
    fb = 8 + 4 * nacc
    for j in range(4 * nacc): L.append(f"  mov.b32 %r{8 + j}, 0;")
    nf = max(fill, 4)
    for j in range(nf): L.append(f"  mov.b32 %r{fb + j}, 0x3f800000;")
    L += ["  mov.u32 %i, 0;", "$L0:"]
    fi = 0
    for rep in range(8):
        for a in range(nacc):
            b = 8 + 4 * a
            q = ", ".join(f"%r{b + j}" for j in range(4))
            L.append(f"  {MMA} {{ {q} }}, {{ %r0, %r1, %r2, %r3 }}, {{ %r4, %r5 }}, {{ {q} }};")
            for j in range(fill):
                x = fb + (fi % nf); fi += 1
                if kind == "fma":
                    L.append(f"  fma.rn.f32 %r{x}, %r{x}, %r1, %r{x};")
                elif kind == "fmul":
                    L.append(f"  mul.rn.f32 %r{x}, %r{x}, 0x3f800001;")
                else:
                    L.append(f"  add.u32 %r{x}, %r{x}, %t;")
    L += ["  add.s32 %i, %i, 1;", "  setp.lt.s32 %p0, %i, %n;", "  @%p0 bra $L0;",
          "  mov.u32 %u, %ctaid.x;", "  cvt.u64.u32 %d3, %u;", f"  mul.lo.u64 %d3, %d3, {nt};",
          "  cvt.u64.u32 %d4, %t;", "  add.u64 %d3, %d3, %d4;", f"  mul.lo.u64 %d3, %d3, {16 * nacc + 16};",
          "  add.u64 %d2, %d2, %d3;"]
    for a in range(nacc):
        b = 8 + 4 * a
        L.append(f"  st.global.v4.b32 [%d2+{16 * a}], {{%r{b}, %r{b+1}, %r{b+2}, %r{b+3}}};")
    for j in range(4, nf):
        L.append(f"  xor.b32 %r{fb + (j & 3)}, %r{fb + (j & 3)}, %r{fb + j};")
    x = [fb + j for j in range(4)]
    L.append(f"  st.global.v4.b32 [%d2+{16 * nacc}], {{%r{x[0]}, %r{x[1]}, %r{x[2]}, %r{x[3]}}};")
    L += ["  ret;", "}"]
    return "\n".join(L)
def timeit(f, reps=7):
    f(); torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2]
SMEM_ATTR = 8  # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
mx = ctypes.c_int(); cu.cuDeviceGetAttribute(ctypes.byref(mx), 81, dev); print("max_smem_optin", mx.value, flush=True)
sm = ctypes.c_int(); cu.cuDeviceGetAttribute(ctypes.byref(sm), 4, dev); print("num_sms", sm.value, flush=True)
Abuf = torch.randn(1 << 16, device="cuda").to(torch.float8_e4m3fn).contiguous()
for nt, smem, label in [(128, 0, "free"), (128, 48 * 1024, "smem48k"), (256, 48 * 1024, "8w_smem48k")]:
    for kind in ("fma", "int"):
        for fill in (0, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64):
            nacc = 8
            fn, o = build(src(nacc, fill, nt, kind), "peak")
            if smem:
                cu.cuFuncSetCacheConfig(fn, 1)
                rc = cu.cuFuncSetAttribute(fn, SMEM_ATTR, ctypes.c_int(smem))
                if rc: print("attr_rc", rc, flush=True)
            grid = 2048; iters = max(40, 120 // (1 + fill // 16))
            out = torch.empty(grid * nt * (16 * nacc + 16) // 4 + 64, device="cuda", dtype=torch.float32)
            ms = timeit(lambda: launch(fn, grid, nt, [Abuf.data_ptr(), out.data_ptr(), iters], smem))
            warps = nt // 32
            macs = grid * warps * 8 * nacc * iters * 4096
            n = subprocess.run(f"/usr/local/cuda/bin/cuobjdump -sass {o} | grep -cE ' (FFMA|FMUL|IADD3|IMAD) '", shell=True, capture_output=True, text=True).stdout.strip()
            print(json.dumps(dict(label=label, kind=kind, fill=fill, ms=round(ms * 1e3, 3), TMACs=round(macs / ms / 1e12, 1), sass_ops=n)), flush=True)
            del out
print("E1_END")
