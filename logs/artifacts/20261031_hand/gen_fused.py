"""PTX generator: fused gate_up GEMM + SiLU * up + per-token-group FP8 quant for sm_89.

One kernel replaces W8A8 GEMM (N gate + N up columns), silu_and_mul and the FP8 quant of the
down_proj input. Each warp owns 4 A fragments x 4 B fragments, 2 gate + 2 up, so g and u land in
the same lane and the activation runs on the accumulators. The quant group (128) equals the tile
width, so a row's scale is known at the end of the block: absmax is reduced through SMEM in three
steps on the A buffers, which are idle by then. A layer's MLP drops from 1.86 GB of activation
traffic through HBM to 0.14 GB.
"""
BM = 64
BN2 = 128
NT = 256
A_STAGE, B_STAGE, S_STAGE = 0x2000, 0x8000, 0x200
LOG2E = "0fBFB8AA3B"      # -log2(e)
ONE = "0f3F800000"
FP8_MAX = "0f43E00000"    # 448
EPS = "0f2EDBE6FF"        # 1e-10


def smem_bytes(stages):
    return stages * (A_STAGE + B_STAGE + S_STAGE)


def gen_fused(K, stages=2, gap=5, NB=4, cps=8, bpos=5, name="fp8_gemm_silu_quant"):
    S = stages
    SB0 = S * A_STAGE
    SS0 = SB0 + S * B_STAGE
    R1, R2, R3 = 0, A_STAGE, A_STAGE + 0x800   # scratch inside the A buffers
    L = []
    e = L.append
    e(".version 8.5\n.target sm_89\n.address_size 64\n.extern .shared .align 16 .b8 smem[];\n")
    e(f".visible .entry {name}(.param .u64 pA, .param .u64 pB, .param .u64 pQ, .param .u64 pAs, .param .u64 pBs, .param .u64 pQs,")
    e("  .param .u32 pM, .param .u32 pN, .param .u32 pG)")
    e(f".reqntid {NT}\n{{")
    e("  .reg .b32 %T<64>;\n  .reg .b32 %C<64>;\n  .reg .b32 %A<16>;\n  .reg .b32 %B<32>;\n  .reg .b32 %S<16>;")
    e("  .reg .b32 %x<16>;\n  .reg .b32 %y<4>;\n  .reg .b32 %I<8>;\n  .reg .b16 %h0;\n  .reg .b64 %d<10>;\n  .reg .pred %p<8>;\n  .reg .pred %pr<2>;\n  .reg .pred %pq<2>;\n  .reg .pred %pbu;")
    e("  .reg .b32 %la<4>;\n  .reg .b32 %lb<2>;\n  .reg .b32 %ra<4>;\n  .reg .b32 %rb<2>;\n  .reg .b32 %da<2>;\n  .reg .b32 %db<2>;")
    e("  .reg .b32 %daz<2>;\n  .reg .b32 %dbz<2>;\n  .reg .b32 %rs, %ds, %ds0, %k, %kp, %st, %g, %ln, %li, %t4, %slot;")
    e("  .reg .b32 %t, %lane, %w, %M, %N, %N2, %KT, %G, %pid, %pidend, %pidm, %pidn, %base, %row, %smem, %rowa, %colb, %npm, %npn, %colt;")
    e("  .reg .b64 %pa, %pb, %pu, %ps;\n  .reg .b32 %bg, %bu;")
    e("  ld.param.u32 %M, [pM];\n  ld.param.u32 %N, [pN];\n  ld.param.u32 %G, [pG];\n  shr.u32 %N2, %N, 1;")
    e("  ld.param.u64 %d0, [pA];\n  ld.param.u64 %d1, [pB];\n  ld.param.u64 %d2, [pQ];")
    e("  ld.param.u64 %d3, [pAs];\n  ld.param.u64 %d4, [pBs];\n  ld.param.u64 %d9, [pQs];")
    e("  mov.b32 %smem, smem;")
    e("  mov.u32 %t, %tid.x;\n  and.b32 %lane, %t, 31;\n  shr.u32 %w, %t, 5;\n  and.b32 %t4, %lane, 3;")
    e(f"  mov.u32 %KT, {K // 128};")
    e(f"  add.u32 %npm, %M, {BM - 1};\n  shr.u32 %npm, %npm, 6;\n  shr.u32 %npn, %N, 8;")     # gate blocks
    e("  mul.lo.u32 %x0, %npm, %npn;")
    e(f"  mov.u32 %pid, %ctaid.x;\n  mul.lo.u32 %pid, %pid, {NB};\n  add.u32 %pidend, %pid, {NB};\n  min.u32 %pidend, %pidend, %x0;")
    e("  and.b32 %ln, %lane, 7;\n  shr.u32 %li, %lane, 3;")
    # K_SW128 write side
    e("  xor.b32 %x3, %ln, %li;\n  shl.b32 %x3, %x3, 4;\n  add.u32 %x4, %li, 4;\n  xor.b32 %x4, %ln, %x4;\n  shl.b32 %x4, %x4, 4;")
    e("  shl.b32 %rowa, %w, 3;\n  add.u32 %rowa, %rowa, %li;")
    e("  shl.b32 %x1, %rowa, 7;\n  add.u32 %x1, %x1, %smem;\n  add.u32 %daz0, %x1, %x3;\n  add.u32 %daz1, %x1, %x4;")
    e("  shl.b32 %colb, %w, 4;\n  add.u32 %colb, %colb, %li;")
    e(f"  shl.b32 %x1, %colb, 7;\n  add.u32 %x1, %x1, {SB0};\n  add.u32 %x1, %x1, %smem;\n  add.u32 %dbz0, %x1, %x3;\n  add.u32 %dbz1, %x1, %x4;")
    # scale slots: threads 0..63 fetch a_s of row t, 64 and 65 fetch b_s of gate and up
    e("  and.b32 %x5, %t, 7;\n  shl.b32 %x5, %x5, 3;\n  shr.u32 %x6, %t, 4;\n  and.b32 %x6, %x6, 3;\n  shl.b32 %x6, %x6, 1;")
    e("  shr.u32 %x7, %t, 3;\n  and.b32 %x7, %x7, 1;\n  add.u32 %x5, %x5, %x6;\n  add.u32 %x5, %x5, %x7;\n  shl.b32 %slot, %x5, 2;")
    e(f"  setp.eq.u32 %p1, %t, {BM};\n  setp.lt.u32 %p7, %t, {BM};\n  setp.eq.u32 %pbu, %t, {BM + 1};")
    e(f"  selp.b32 %x5, {BM * 4}, %slot, %p1;\n  selp.b32 %x5, {BM * 4 + 4}, %x5, %pbu;\n  add.u32 %x5, %x5, {SS0};\n  add.u32 %ds0, %smem, %x5;")
    # ldmatrix lanes
    e("  shr.u32 %x1, %lane, 3;\n  and.b32 %x1, %x1, 1;\n  shl.b32 %x1, %x1, 3;\n  add.u32 %x1, %x1, %ln;\n  shl.b32 %x1, %x1, 7;")
    e("  shr.u32 %x2, %lane, 4;")
    for j in range(4):
        e(f"  add.u32 %x3, %x2, {2 * j};\n  xor.b32 %x3, %x3, %ln;\n  shl.b32 %x3, %x3, 4;")
        e(f"  add.u32 %la{j}, %x1, %x3;\n  add.u32 %la{j}, %la{j}, %smem;")
    e("  shl.b32 %x1, %ln, 7;\n  shl.b32 %x4, %w, 11;\n  add.u32 %x1, %x1, %x4;")
    e(f"  add.u32 %x1, %x1, {SB0};\n  add.u32 %x1, %x1, %smem;")
    for p in range(2):
        e(f"  add.u32 %x3, %li, {4 * p};\n  xor.b32 %x3, %x3, %ln;\n  shl.b32 %x3, %x3, 4;\n  add.u32 %lb{p}, %x1, %x3;")
    e("  shr.u32 %g, %lane, 2;\n  shl.b32 %x1, %g, 5;")
    e(f"  add.u32 %rs, %x1, {SS0};\n  add.u32 %rs, %rs, %smem;")
    e("  shl.b32 %x2, %w, 4;\n  shl.b32 %x3, %t4, 1;\n  add.u32 %colt, %x2, %x3;")

    def copy(pred):
        out = []
        if pred:
            for c in range(2):
                out.append(f"  and.pred %pq{c}, %pr{c}, {pred};")
        for c in range(2):
            q = f"@%pq{c}" if pred else f"@%pr{c}"
            out.append(f"  {q} cp.async.cg.shared.global [%da{c & 1} + {c * 512}], [%pa + {4 * c * K}], 16;")
        q = f"@{pred} " if pred else ""
        for c in range(4):
            out.append(f"  {q}cp.async.cg.shared.global [%db{c & 1} + {c * 512}], [%pb + {4 * c * K}], 16;")
        for c in range(4):
            out.append(f"  {q}cp.async.cg.shared.global [%db{c & 1} + {16384 + c * 512}], [%pu + {4 * c * K}], 16;")
        p2 = "%p3" if pred else "%p2"
        out.append(f"  @{p2} ld.global.b32 %x12, [%ps];")
        out.append("  cp.async.commit_group;")
        return out

    def ptr_advance():
        return ["  add.u64 %pa, %pa, 128;", "  add.u64 %pb, %pb, 128;", "  add.u64 %pu, %pu, 128;", "  add.u64 %ps, %ps, 4;"]

    def set_dst(st):
        return [f"  mul.lo.u32 %x8, {st}, {A_STAGE};", "  add.u32 %da0, %daz0, %x8;", "  add.u32 %da1, %daz1, %x8;",
                f"  mul.lo.u32 %x8, {st}, {B_STAGE};", "  add.u32 %db0, %dbz0, %x8;", "  add.u32 %db1, %dbz1, %x8;",
                f"  mul.lo.u32 %x8, {st}, {S_STAGE};", "  add.u32 %ds, %ds0, %x8;"]

    def set_src(st):
        return [f"  mul.lo.u32 %x8, {st}, {A_STAGE};"] + [f"  add.u32 %ra{j}, %la{j}, %x8;" for j in range(4)] + [
                f"  mul.lo.u32 %x8, {st}, {B_STAGE};", "  add.u32 %rb0, %lb0, %x8;", "  add.u32 %rb1, %lb1, %x8;"]

    def ld_a(F, j):
        q = ", ".join(f"%A{F * 4 + i}" for i in range(4))
        return f"  ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{q}}}, [%ra{j} + {F * 2048}];"

    def ld_b(bset, nb):
        q = ", ".join(f"%B{bset * 16 + nb * 4 + i}" for i in range(4))
        return f"  ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{q}}}, [%rb{bset} + {(nb & 1) * 1024 + (nb >> 1) * 16384}];"

    def mma(f, F, bset, nb, off, first):
        t = ", ".join(f"%T{f * 4 + i}" for i in range(4))
        a = ", ".join(f"%A{F * 4 + i}" for i in range(4))
        b = f"%B{bset * 16 + nb * 4 + off}, %B{bset * 16 + nb * 4 + off + 1}"
        out = [f"  mov.b32 %T{f * 4 + i}, 0;" for i in range(4)] if first else []
        out.append(f"  mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {{{t}}}, {{{a}}}, {{{b}}}, {{{t}}};")
        return out

    def fma(f):
        F, nb = f >> 2, f & 3
        off = 8 if nb >= 2 else 0
        return [f"  fma.rn.f32 %C{f * 4 + i}, %T{f * 4 + i}, %S{off + F * 2 + (i >> 1)}, %C{f * 4 + i};" for i in range(4)]

    e("$BLOCK:")
    e("  setp.ge.u32 %p0, %pid, %pidend;\n  @%p0 bra $END;")
    e("  mul.lo.u32 %x2, %G, %npn;")
    e("  div.u32 %x3, %pid, %x2;\n  mul.lo.u32 %x3, %x3, %G;")
    e("  sub.u32 %x4, %npm, %x3;\n  min.u32 %x4, %x4, %G;")
    e("  rem.u32 %x5, %pid, %x4;\n  add.u32 %pidm, %x3, %x5;")
    e("  rem.u32 %x5, %pid, %x2;\n  div.u32 %pidn, %x5, %x4;")
    e("  shl.b32 %base, %pidm, 6;")
    e(f"  add.u32 %row, %base, %rowa;\n  mul.wide.u32 %d5, %row, {K};")
    e("  add.u64 %pa, %d0, %d5;\n  shl.b32 %x1, %ln, 4;\n  cvt.u64.u32 %d6, %x1;\n  add.u64 %pa, %pa, %d6;")
    for c in range(2):
        e(f"  add.u32 %x1, %row, {4 * c};\n  setp.lt.u32 %pr{c}, %x1, %M;")
    e(f"  shl.b32 %x1, %pidn, 7;\n  add.u32 %x1, %x1, %colb;\n  mul.wide.u32 %d5, %x1, {K};\n  add.u64 %pb, %d1, %d5;\n  add.u64 %pb, %pb, %d6;")
    e(f"  mul.wide.u32 %d5, %N2, {K};\n  add.u64 %pu, %pb, %d5;")
    e("  add.u32 %row, %base, %t;\n  setp.lt.u32 %p2, %row, %M;\n  and.pred %p2, %p2, %p7;\n  or.pred %p2, %p2, %p1;\n  or.pred %p2, %p2, %pbu;")
    e("  shl.b32 %x4, %KT, 2;\n  mul.wide.u32 %d5, %row, %x4;\n  add.u64 %ps, %d3, %d5;")
    e("  mul.wide.u32 %d6, %pidn, %x4;\n  add.u64 %d6, %d4, %d6;\n  selp.b64 %ps, %d6, %ps, %p1;")
    e("  add.u32 %x1, %pidn, %npn;\n  mul.wide.u32 %d6, %x1, %x4;\n  add.u64 %d6, %d4, %d6;\n  selp.b64 %ps, %d6, %ps, %pbu;")
    for i in range(64):
        e(f"  mov.b32 %C{i}, 0;")
    for f in range(16 - gap, 16):
        for i in range(4):
            e(f"  mov.b32 %T{f * 4 + i}, 0;")
    for i in range(16):
        e(f"  mov.b32 %S{i}, 0;")
    e("  mov.u32 %k, 0;")
    for i in range(S - 1):
        e(f"  mov.u32 %st, {i};")
        L.extend(set_dst("%st"))
        L.extend(copy(None))
        e("  @%p2 st.shared.b32 [%ds], %x12;")
        L.extend(ptr_advance())
    e(f"  cp.async.wait_group {S - 2};\n  bar.sync 0;")
    e("  mov.u32 %st, 0;")
    L.extend(set_src("%st"))
    for F in range(4):
        e(ld_a(F, 0))
    for nb in range(4):
        e(ld_b(0, nb))

    e("$LOOP:")
    e(f"  add.u32 %kp, %k, {S - 1};\n  setp.lt.u32 %p4, %kp, %KT;\n  and.pred %p3, %p4, %p2;")
    e(f"  rem.u32 %st, %kp, {S};")
    L.extend(set_dst("%st"))
    cp = copy("%p4") + ptr_advance()

    for j in range(4):
        bset, off = j >> 1, (j & 1) * 2
        idx = 0
        if j == 3:
            e(f"  cp.async.wait_group {S - 2};\n  bar.sync 0;")
            e(f"  add.u32 %kp, %k, 1;\n  rem.u32 %st, %kp, {S};")
            L.extend(set_src("%st"))
        for F in range(4):
            for nb in range(4):
                f = F * 4 + nb
                L.extend(mma(f, F, bset, nb, off, j == 0))
                if j == 0:
                    if idx < gap:
                        L.extend(fma(16 - gap + idx))
                    if idx >= gap:
                        take = cp[:cps]
                        del cp[:cps]
                        L.extend(take)
                    if idx == 15:
                        L.extend(cp)
                        cp = []
                elif j == 1:
                    if bpos <= idx < bpos + 4:
                        e(ld_b(1, idx - bpos))
                    if idx == 4:
                        e("  ld.shared.v4.b32 {%S0, %S1, %S2, %S3}, [%x9];")
                        e("  ld.shared.v4.b32 {%S4, %S5, %S6, %S7}, [%x9 + 16];")
                        e("  ld.shared.v2.b32 {%bg, %bu}, [%x10];")
                    if idx >= 8:
                        i = idx - 8
                        e(f"  mul.rn.f32 %S{8 + i}, %S{i}, %bu;")
                        e(f"  mul.rn.f32 %S{i}, %S{i}, %bg;")
                elif j == 2:
                    if idx == 0:
                        e("  @%p3 st.shared.b32 [%ds], %x12;")
                elif j == 3:
                    if bpos <= idx < bpos + 4:
                        e(ld_b(0, idx - bpos))
                    if idx >= gap:
                        L.extend(fma(idx - gap))
                idx += 1
            e(ld_a(F, (j + 1) & 3))
        if j == 0:
            e(f"  rem.u32 %x8, %k, {S};\n  mul.lo.u32 %x8, %x8, {S_STAGE};")
            e(f"  add.u32 %x9, %rs, %x8;\n  add.u32 %x10, %smem, {SS0 + BM * 4};\n  add.u32 %x10, %x10, %x8;")
    e("  add.u32 %k, %k, 1;\n  setp.lt.u32 %p5, %k, %KT;\n  @%p5 bra $LOOP;")
    for f in range(16 - gap, 16):
        L.extend(fma(f))

    # silu(g) * up on the accumulators; the result stays in the gate registers
    for F in range(4):
        for nb in range(2):
            for i in range(4):
                g = f"%C{(F * 4 + nb) * 4 + i}"
                u = f"%C{(F * 4 + nb + 2) * 4 + i}"
                e(f"  mul.rn.f32 %y0, {g}, {LOG2E};\n  ex2.approx.ftz.f32 %y0, %y0;\n  add.rn.f32 %y0, %y0, {ONE};")
                e(f"  rcp.approx.ftz.f32 %y0, %y0;\n  mul.rn.f32 {g}, {g}, %y0;\n  mul.rn.f32 {g}, {g}, {u};")
    # absmax of each row of this lane, written next to the 32 lanes that share the row
    e(f"  shl.b32 %x1, %g, 7;\n  shl.b32 %x2, %w, 2;\n  add.u32 %x2, %x2, %t4;\n  shl.b32 %x2, %x2, 2;\n  add.u32 %x1, %x1, %x2;\n  add.u32 %x1, %x1, %smem;")
    for F in range(4):
        for h in range(2):
            vals = [f"%C{(F * 4 + nb) * 4 + 2 * h + q}" for nb in range(2) for q in range(2)]
            e(f"  abs.f32 %y1, {vals[0]};")
            for v in vals[1:]:
                e(f"  abs.f32 %y2, {v};\n  max.f32 %y1, %y1, %y2;")
            e(f"  st.shared.b32 [%x1 + {(16 * F + 8 * h) * 128 + R1}], %y1;")
    e("  bar.sync 0;")
    e("  shr.u32 %x1, %t, 2;\n  shl.b32 %x2, %x1, 7;\n  and.b32 %x3, %t, 3;\n  shl.b32 %x4, %x3, 5;\n  add.u32 %x2, %x2, %x4;\n  add.u32 %x2, %x2, %smem;")
    e("  ld.shared.v4.b32 {%I0, %I1, %I2, %I3}, [%x2];\n  ld.shared.v4.b32 {%I4, %I5, %I6, %I7}, [%x2 + 16];")
    for i in range(1, 8):
        e(f"  max.f32 %I0, %I0, %I{i};")
    e(f"  shl.b32 %x2, %x1, 4;\n  shl.b32 %x4, %x3, 2;\n  add.u32 %x2, %x2, %x4;\n  add.u32 %x2, %x2, %smem;\n  st.shared.b32 [%x2 + {R2}], %I0;")
    e("  bar.sync 0;")
    # threads 0..63: one row each
    e(f"  shl.b32 %x2, %t, 4;\n  add.u32 %x2, %x2, %smem;\n  add.u32 %x2, %x2, {R2};")
    e("  @%p7 ld.shared.v4.b32 {%I0, %I1, %I2, %I3}, [%x2];")
    for i in range(1, 4):
        e(f"  max.f32 %I0, %I0, %I{i};")
    e(f"  max.f32 %I0, %I0, {EPS};\n  div.rn.f32 %y1, %I0, {FP8_MAX};\n  div.rn.f32 %y2, {ONE}, %y1;")
    e(f"  add.u32 %x2, %smem, {R3};\n  add.u32 %x2, %x2, %slot;\n  @%p7 st.shared.b32 [%x2], %y2;")
    e("  add.u32 %row, %base, %t;\n  setp.lt.u32 %p6, %row, %M;\n  and.pred %p6, %p6, %p7;")
    e("  shr.u32 %x3, %N, 8;\n  mul.wide.u32 %d5, %row, %x3;\n  cvt.u64.u32 %d6, %pidn;\n  add.u64 %d5, %d5, %d6;\n  shl.b64 %d5, %d5, 2;\n  add.u64 %d5, %d9, %d5;")
    e("  @%p6 st.global.b32 [%d5], %y1;")
    e("  bar.sync 0;")
    # every lane: inverse scales of its 8 rows, then the FP8 store
    e(f"  shl.b32 %x2, %g, 5;\n  add.u32 %x2, %x2, %smem;\n  add.u32 %x2, %x2, {R3};")
    e("  ld.shared.v4.b32 {%I0, %I1, %I2, %I3}, [%x2];\n  ld.shared.v4.b32 {%I4, %I5, %I6, %I7}, [%x2 + 16];")
    e("  shl.b32 %x1, %pidn, 7;\n  add.u32 %x1, %x1, %colt;\n  cvt.u64.u32 %d7, %x1;\n  add.u64 %d7, %d2, %d7;")
    for F in range(4):
        for h in range(2):
            e(f"  add.u32 %row, %base, {16 * F + 8 * h};\n  add.u32 %row, %row, %g;\n  setp.lt.u32 %p6, %row, %M;")
            e("  mul.wide.u32 %d8, %row, %N2;\n  add.u64 %d8, %d7, %d8;")
            for nb in range(2):
                f = F * 4 + nb
                e(f"  mul.rn.f32 %y1, %C{f * 4 + 2 * h}, %I{F * 2 + h};\n  mul.rn.f32 %y2, %C{f * 4 + 2 * h + 1}, %I{F * 2 + h};")
                e("  cvt.rn.satfinite.e4m3x2.f32 %h0, %y2, %y1;")
                e(f"  @%p6 st.global.b16 [%d8 + {8 * nb}], %h0;")
    e("  cp.async.wait_group 0;\n  bar.sync 0;")
    e("  add.u32 %pid, %pid, 1;\n  bra.uni $BLOCK;")
    e("$END:\n  ret;\n}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    print(gen_fused(5120))
