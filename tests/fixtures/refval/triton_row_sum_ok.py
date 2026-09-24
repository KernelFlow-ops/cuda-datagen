import triton
import triton.language as tl


@triton.jit
def _row_sum_kernel(x, out, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    values = tl.load(x + row * cols + offs, mask=mask, other=0.0)
    tl.store(out + row, tl.sum(values, axis=0))


def launch_row_sum(x, out, rows, cols):
    _row_sum_kernel[(rows,)](x, out, cols, BLOCK=max(1, triton.next_power_of_2(cols)))
