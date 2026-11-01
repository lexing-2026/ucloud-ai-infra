import torch
from sglang.kernels.ops.gemm.fp8_kernel import w8a8_block_fp8_matmul
N, K, M = 34816, 5120, 8192
B = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
Bs = (torch.rand(N // 128, K // 128, device="cuda") * 0.02 + 0.01).float()
A = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
As = (torch.rand(M, K // 128, device="cuda") * 0.02 + 0.01).float()
out = w8a8_block_fp8_matmul(A, B, As, Bs, [128, 128], torch.bfloat16)
torch.cuda.synchronize()
print("ok", out.shape, out.dtype)
