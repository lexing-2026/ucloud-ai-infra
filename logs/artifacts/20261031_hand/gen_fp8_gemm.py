"""PTX generator for a hand-scheduled block-FP8 GEMM on sm_89 (RTX 4090).

Tile: M=BM (64 or 128), N=128, K=128; BM/16 warps, each owning 4 A x 4 B m16n8k32
fragments. Measured reasons it beats the Triton kernel at the 450 W power cap:

  * Persistent CTAs: one CTA walks NB blocks, so a single-CTA SM does not idle for dispatch.
  * BM=128: B tile is shared by twice the rows, 33% less on-chip traffic per MAC, the power
    controller answers with +120 MHz.
  * FP32 scaling is software-pipelined into the mma stream of the last K32 step and needs
    72 FP ops per warp and tile instead of 128 (s[i] = a_s[i] * b_s is computed once).
  * Scales do not need a second barrier or an SMEM broadcast per warp.

Lane mapping rules that are not optional (each one cost a factor when ignored):
  * one cp.async covers 4 cache lines, 8 lanes each, so lanes coalesce into 4 L2 requests;
  * SMEM uses K_SW128: chunk ^= row & 7, otherwise every ldmatrix phase is an 8-way bank conflict.
"""

BN, BK = 128, 128
B_STAGE = 0x4000


def stage_sizes(BM):
    return BM * 128, B_STAGE, 0x200 * (BM // 64)


def smem_bytes(stages, BM=128):
    return stages * sum(stage_sizes(BM))


def gen(K, stages=2, gap=5, BM=128, NB=8, interleave=True, name="fp8_gemm_hand", dbg=False, cps=3, bpos=0):
    S = stages
    NT = BM * 2
    LOGM = {64: 6, 128: 7}[BM]
    A_STAGE, _, S_STAGE = stage_sizes(BM)
    SB0 = S * A_STAGE
    SS0 = SB0 + S * B_STAGE
    L = []
    e = L.append
    e(".version 8.5\n.target sm_89\n.address_size 64\n.extern .shared .align 16 .b8 smem[];\n")
    e(f".visible .entry {name}(.param .u64 pA, .param .u64 pB, .param .u64 pC, .param .u64 pAs, .param .u64 pBs,")
    e("  .param .u32 pM, .param .u32 pN, .param .u32 pK, .param .u32 pG" + (", .param .u64 pD)" if dbg else ")"))
    e(f".reqntid {NT}\n{{")
    e("  .reg .b32 %T<64>;\n  .reg .b32 %C<64>;\n  .reg .b32 %A<16>;\n  .reg .b32 %B<32>;\n  .reg .b32 %S<8>;")
    e("  .reg .b32 %x<16>;\n  .reg .b64 %d<10>;\n  .reg .pred %p<8>;\n  .reg .pred %pr<4>;\n  .reg .pred %pq<4>;\n  .reg .pred %pw, %pz;")
    e("  .reg .b32 %la<4>;\n  .reg .b32 %lb<2>;\n  .reg .b32 %ra<4>;\n  .reg .b32 %rb<2>;\n  .reg .b32 %da<2>;\n  .reg .b32 %db<2>;")
    e("  .reg .b32 %daz<2>;\n  .reg .b32 %dbz<2>;\n  .reg .b32 %rs, %ds, %ds0, %k, %kp, %st, %g, %wm, %wn, %ln, %li;")
    e("  .reg .b32 %t, %lane, %w, %M, %N, %KT, %G, %pid, %pidend, %pidm, %pidn, %base, %row, %smem, %rowa, %colb, %npm, %npn;")
    e("  .reg .b64 %pa, %pb, %ps;\n  .reg .b32 %bscale, %colt;\n  .reg .b32 %c<8>;\n  .reg .b64 %dd;")
    clk = (lambda r: e(f"  mov.u32 %c{r}, %clock;")) if dbg else (lambda r: None)
    clk(0)
    if dbg:
        e("  mov.u32 %c5, 0;\n  mov.u32 %c6, 0;")
    e("  ld.param.u32 %M, [pM];\n  ld.param.u32 %N, [pN];\n  ld.param.u32 %G, [pG];")
    e("  ld.param.u64 %d0, [pA];\n  ld.param.u64 %d1, [pB];\n  ld.param.u64 %d2, [pC];")
    e("  ld.param.u64 %d3, [pAs];\n  ld.param.u64 %d4, [pBs];")
    e("  mov.b32 %smem, smem;")
    e("  mov.u32 %t, %tid.x;\n  and.b32 %lane, %t, 31;\n  shr.u32 %w, %t, 5;")
    e("  shr.u32 %wm, %w, 2;\n  and.b32 %wn, %w, 3;")                      # warp -> M half, N quarter
    e(f"  mov.u32 %KT, {K // BK};")
    e(f"  add.u32 %npm, %M, {BM - 1};\n  shr.u32 %npm, %npm, {LOGM};\n  shr.u32 %npn, %N, 7;")
    e("  mul.lo.u32 %x0, %npm, %npn;")                                     # blocks in the matrix
    e(f"  mov.u32 %pid, %ctaid.x;\n  mul.lo.u32 %pid, %pid, {NB};\n  add.u32 %pidend, %pid, {NB};\n  min.u32 %pidend, %pidend, %x0;")
    # lane constants
    e("  and.b32 %ln, %lane, 7;\n  shr.u32 %li, %lane, 3;")                # lane in cache line, line in instruction
    # K_SW128 on the write side: row & 7 is li for even copies, li + 4 for odd ones
    e("  xor.b32 %x3, %ln, %li;\n  shl.b32 %x3, %x3, 4;\n  add.u32 %x4, %li, 4;\n  xor.b32 %x4, %ln, %x4;\n  shl.b32 %x4, %x4, 4;")
    e("  shl.b32 %rowa, %wn, 4;\n  add.u32 %rowa, %rowa, %li;\n  shl.b32 %x1, %wm, 6;\n  add.u32 %rowa, %rowa, %x1;")
    e("  shl.b32 %x1, %rowa, 7;\n  add.u32 %x1, %x1, %smem;\n  add.u32 %daz0, %x1, %x3;\n  add.u32 %daz1, %x1, %x4;")
    e("  shl.b32 %colb, %wn, 5;\n  add.u32 %colb, %colb, %li;\n  setp.lt.u32 %pw, %w, 4;")
    e(f"  shl.b32 %x1, %colb, 7;\n  add.u32 %x1, %x1, {SB0};\n  add.u32 %x1, %x1, %smem;\n  add.u32 %dbz0, %x1, %x3;\n  add.u32 %dbz1, %x1, %x4;")
    # scale slots: threads 0..BM-1 fetch a_s of row t, thread BM fetches b_s
    e("  and.b32 %x5, %t, 7;\n  shl.b32 %x5, %x5, 3;\n  shr.u32 %x6, %t, 4;\n  and.b32 %x6, %x6, 3;\n  shl.b32 %x6, %x6, 1;")
    e("  shr.u32 %x7, %t, 3;\n  and.b32 %x7, %x7, 1;\n  add.u32 %x5, %x5, %x6;\n  add.u32 %x5, %x5, %x7;")
    e("  shr.u32 %x6, %t, 6;\n  shl.b32 %x6, %x6, 6;\n  add.u32 %x5, %x5, %x6;\n  shl.b32 %x5, %x5, 2;")
    e(f"  setp.eq.u32 %p1, %t, {BM};\n  setp.lt.u32 %p7, %t, {BM};")
    e(f"  selp.b32 %x5, {BM * 4}, %x5, %p1;\n  add.u32 %x5, %x5, {SS0};\n  add.u32 %ds0, %smem, %x5;")
    # ldmatrix lane offsets, one register per K32 step because the swizzle is not additive
    e("  shr.u32 %x1, %lane, 3;\n  and.b32 %x1, %x1, 1;\n  shl.b32 %x1, %x1, 3;\n  add.u32 %x1, %x1, %ln;\n  shl.b32 %x1, %x1, 7;")
    e("  shl.b32 %x4, %wm, 13;\n  add.u32 %x1, %x1, %x4;\n  shr.u32 %x2, %lane, 4;")
    for j in range(4):
        e(f"  add.u32 %x3, %x2, {2 * j};\n  xor.b32 %x3, %x3, %ln;\n  shl.b32 %x3, %x3, 4;")
        e(f"  add.u32 %la{j}, %x1, %x3;\n  add.u32 %la{j}, %la{j}, %smem;")
    e("  shl.b32 %x1, %ln, 7;\n  shl.b32 %x4, %wn, 12;\n  add.u32 %x1, %x1, %x4;")
    e(f"  add.u32 %x1, %x1, {SB0};\n  add.u32 %x1, %x1, %smem;")
    for p in range(2):
        e(f"  add.u32 %x3, %li, {4 * p};\n  xor.b32 %x3, %x3, %ln;\n  shl.b32 %x3, %x3, 4;\n  add.u32 %lb{p}, %x1, %x3;")
    # shared offset of this lane's 8 scales
    e("  shr.u32 %g, %lane, 2;\n  shl.b32 %x1, %g, 5;\n  shl.b32 %x2, %wm, 8;\n  add.u32 %x1, %x1, %x2;")
    e(f"  add.u32 %rs, %x1, {SS0};\n  add.u32 %rs, %rs, %smem;")
    # column byte offset of this lane inside the output tile
    e("  shl.b32 %x2, %wn, 6;\n  and.b32 %x3, %lane, 3;\n  shl.b32 %x3, %x3, 2;\n  add.u32 %colt, %x2, %x3;")

    def copy(pred):
        out = []
        if pred:
            for c in range(4):
                out.append(f"  and.pred %pq{c}, %pr{c}, {pred};")
        for c in range(4):
            q = f"@%pq{c}" if pred else f"@%pr{c}"
            out.append(f"  {q} cp.async.cg.shared.global [%da{c & 1} + {c * 512}], [%pa + {4 * c * K}], 16;")
        out.append(f"  and.pred %pz, %pw, {pred};" if pred else "  mov.pred %pz, %pw;")
        for c in range(8):
            out.append(f"  @%pz cp.async.cg.shared.global [%db{c & 1} + {c * 512}], [%pb + {4 * c * K}], 16;")
        p2 = "%p3" if pred else "%p2"
        out.append(f"  @{p2} ld.global.b32 %x12, [%ps];")
        out.append("  cp.async.commit_group;")
        return out

    def ptr_advance():
        return ["  add.u64 %pa, %pa, 128;", "  add.u64 %pb, %pb, 128;", "  add.u64 %ps, %ps, 4;"]

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
        return f"  ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{q}}}, [%rb{bset} + {nb * 1024}];"

    def mma(f, F, bset, nb, off, first):
        t = ", ".join(f"%T{f * 4 + i}" for i in range(4))
        a = ", ".join(f"%A{F * 4 + i}" for i in range(4))
        b = f"%B{bset * 16 + nb * 4 + off}, %B{bset * 16 + nb * 4 + off + 1}"
        out = [f"  mov.b32 %T{f * 4 + i}, 0;" for i in range(4)] if first else []
        out.append(f"  mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {{{t}}}, {{{a}}}, {{{b}}}, {{{t}}};")
        return out

    def fma(f):
        F = f >> 2
        return [f"  fma.rn.f32 %C{f * 4 + i}, %T{f * 4 + i}, %S{F * 2 + (i >> 1)}, %C{f * 4 + i};" for i in range(4)]

    e("$BLOCK:")
    e("  setp.ge.u32 %p0, %pid, %pidend;\n  @%p0 bra $END;")
    clk(1)
    # same block walk as the Triton kernel: rows of a group share one N column
    e("  mul.lo.u32 %x2, %G, %npn;")                                       # pids in group
    e("  div.u32 %x3, %pid, %x2;\n  mul.lo.u32 %x3, %x3, %G;")             # first_pid_m
    e("  sub.u32 %x4, %npm, %x3;\n  min.u32 %x4, %x4, %G;")                # group_size_m
    e("  rem.u32 %x5, %pid, %x4;\n  add.u32 %pidm, %x3, %x5;")
    e("  rem.u32 %x5, %pid, %x2;\n  div.u32 %pidn, %x5, %x4;")
    e(f"  shl.b32 %base, %pidm, {LOGM};")
    e(f"  add.u32 %row, %base, %rowa;\n  mul.wide.u32 %d5, %row, {K};")
    e("  add.u64 %pa, %d0, %d5;\n  shl.b32 %x1, %ln, 4;\n  cvt.u64.u32 %d6, %x1;\n  add.u64 %pa, %pa, %d6;")
    for c in range(4):
        e(f"  add.u32 %x1, %row, {4 * c};\n  setp.lt.u32 %pr{c}, %x1, %M;")
    e(f"  shl.b32 %x1, %pidn, 7;\n  add.u32 %x1, %x1, %colb;\n  mul.wide.u32 %d5, %x1, {K};\n  add.u64 %pb, %d1, %d5;\n  add.u64 %pb, %pb, %d6;")
    e("  add.u32 %row, %base, %t;\n  setp.lt.u32 %p2, %row, %M;\n  and.pred %p2, %p2, %p7;\n  or.pred %p2, %p2, %p1;")
    e("  shl.b32 %x4, %KT, 2;\n  mul.wide.u32 %d5, %row, %x4;\n  add.u64 %ps, %d3, %d5;")
    e("  mul.wide.u32 %d6, %pidn, %x4;\n  add.u64 %d6, %d4, %d6;\n  selp.b64 %ps, %d6, %ps, %p1;")
    for i in range(64):
        e(f"  mov.b32 %C{i}, 0;")
    for f in range(16 - gap, 16):
        for i in range(4):
            e(f"  mov.b32 %T{f * 4 + i}, 0;")
    for i in range(8):
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
            clk(3)
            e(f"  cp.async.wait_group {S - 2};\n  bar.sync 0;")
            clk(4)
            if dbg:
                e("  sub.u32 %c7, %c4, %c3;\n  add.u32 %c5, %c5, %c7;")
            e(f"  add.u32 %kp, %k, 1;\n  rem.u32 %st, %kp, {S};")
            L.extend(set_src("%st"))
        for F in range(4):
            for nb in range(4):
                f = F * 4 + nb
                L.extend(mma(f, F, bset, nb, off, j == 0))
                if j == 0:
                    if idx < gap and interleave:
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
                        e("  ld.shared.b32 %bscale, [%x10];")
                    if idx >= 8:
                        e(f"  mul.rn.f32 %S{idx - 8}, %S{idx - 8}, %bscale;")
                elif j == 2:
                    if idx == 0:
                        e("  @%p3 st.shared.b32 [%ds], %x12;")
                elif j == 3:
                    if bpos <= idx < bpos + 4:
                        e(ld_b(0, idx - bpos))
                    if interleave and idx >= gap:
                        L.extend(fma(idx - gap))
                idx += 1
            e(ld_a(F, (j + 1) & 3))
        if j == 0:
            # scale slots of this tile
            e(f"  rem.u32 %x8, %k, {S};\n  mul.lo.u32 %x8, %x8, {S_STAGE};")
            e(f"  add.u32 %x9, %rs, %x8;\n  add.u32 %x10, %smem, {SS0 + BM * 4};\n  add.u32 %x10, %x10, %x8;")
        if j == 3 and not interleave:
            for f in range(16 - gap):
                L.extend(fma(f))
    if not interleave:
        for f in range(16 - gap, 16):
            L.extend(fma(f))
    e("  add.u32 %k, %k, 1;\n  setp.lt.u32 %p5, %k, %KT;\n  @%p5 bra $LOOP;")
    if interleave:
        for f in range(16 - gap, 16):
            L.extend(fma(f))
    clk(2)

    # store: bf16x2 keeps one instruction per column pair
    e("  shl.b32 %x0, %N, 1;\n  shl.b32 %x1, %pidn, 8;\n  add.u32 %x1, %x1, %colt;\n  cvt.u64.u32 %d7, %x1;")
    e("  add.u64 %d7, %d2, %d7;\n  shl.b32 %x3, %wm, 6;\n  add.u32 %x3, %x3, %base;")
    for F in range(4):
        for h in range(2):
            e(f"  add.u32 %row, %x3, {16 * F + 8 * h};\n  add.u32 %row, %row, %g;\n  setp.lt.u32 %p6, %row, %M;")
            e("  mul.wide.u32 %d8, %row, %x0;\n  add.u64 %d8, %d7, %d8;")
            for nb in range(4):
                f = F * 4 + nb
                e(f"  cvt.rn.bf16x2.f32 %x11, %C{f * 4 + 2 * h + 1}, %C{f * 4 + 2 * h};")
                e(f"  @%p6 st.global.b32 [%d8 + {16 * nb}], %x11;")
    e("  cp.async.wait_group 0;\n  bar.sync 0;")
    if dbg:
        e("  mov.u32 %c3, %clock;\n  sub.u32 %c4, %c3, %c2;\n  add.u32 %c6, %c6, %c4;")
    e("  add.u32 %pid, %pid, 1;\n  bra.uni $BLOCK;")
    e("$END:")
    if dbg:
        e("  mov.u32 %c3, %clock;\n  sub.u32 %c4, %c3, %c0;")
        e("  ld.param.u64 %dd, [pD];\n  mov.u32 %x0, %ctaid.x;\n  shl.b32 %x0, %x0, 4;\n  cvt.u64.u32 %d5, %x0;\n  add.u64 %dd, %dd, %d5;")
        e("  setp.eq.u32 %p6, %t, 0;\n  @%p6 st.global.v4.b32 [%dd], {%c4, %c5, %c6, %pid};")
    e("  ret;\n}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    print(gen(5120))
