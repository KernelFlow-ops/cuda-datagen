# @fake compile=pass refval=pass cases=12
import triton
import triton.language as tl


@triton.jit
def _add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)


def launch_vector_add(a, b, c, n):
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _add_kernel[grid](a, b, c, n, BLOCK=BLOCK)
