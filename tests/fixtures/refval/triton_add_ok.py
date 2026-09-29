import triton
import triton.language as tl


@triton.jit
def _add_kernel(a, b, c, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(c + offs, tl.load(a + offs, mask=mask) + tl.load(b + offs, mask=mask), mask=mask)


def launch_add(a, b, c, n):
    if not (a.is_cuda and b.is_cuda and c.is_cuda):
        raise RuntimeError("CUDA tensors required")
    _add_kernel[(triton.cdiv(n, 128),)](a, b, c, n, BLOCK=128)
