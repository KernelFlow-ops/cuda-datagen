import triton
import triton.language as tl


@triton.jit
def _scale_kernel(x, alpha, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out + offs, tl.load(x + offs, mask=mask) * alpha, mask=mask)


def launch_scale(x, alpha, out, n):
    _scale_kernel[(triton.cdiv(n, 128),)](x, alpha, out, n, BLOCK=128)
