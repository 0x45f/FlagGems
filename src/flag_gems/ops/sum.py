import logging
import math

import torch
import triton
import triton.language as tl

from .. import runtime
from ..runtime import torch_device_fn
from ..utils import dim_compress, libentry, libtuner
from ..utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def sum_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M

    inp_val = tl.load(inp_ptrs, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def sum_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    if tl.constexpr(mid.dtype.element_ty == tl.float16) or tl.constexpr(
        mid.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = mid.dtype.element_ty

    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=0).to(cdtype)
    sum_val = tl.sum(mid_val)
    tl.store(out, sum_val)


@libentry()
# @libtuner(
#     configs=runtime.get_tuned_config("naive_reduction"),
#     key=["M", "N"],
#     share="naive_reduction",
# )
@triton.heuristics(
    values={
        "BLOCK_M": lambda args: 256,
        "BLOCK_N": lambda args: 8,
        "num_warps": lambda args: 4,
    }
)
@triton.jit
def sum_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(
        inp.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    # Map the program id to the row of inp it should compute.
    pid = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + pid * N
    out = out + pid
    row_mask = pid < M

    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=0).to(cdtype)
        _sum += a
    sum = tl.sum(_sum, axis=1)[:, None]
    tl.store(out, sum, row_mask)


@triton.jit
def sum_dim1_kernel(
    input_ptr,
    output_ptr,
    # 张量元数据
    seq_len, num_heads, feat_dim,
    # 内存布局信息
    input_stride_seq, input_stride_head, input_stride_feat,
    output_stride_seq, output_stride_feat,
    # 内核配置
    BLOCK_SEQ: tl.constexpr,
    BLOCK_FEAT: tl.constexpr,
    DTYPE: tl.constexpr
):
    # 计算程序ID
    pid_seq = tl.program_id(0)
    pid_feat = tl.program_id(1)
    
    # 计算序列和特征维度的起始索引
    seq_start = pid_seq * BLOCK_SEQ
    feat_start = pid_feat * BLOCK_FEAT
    
    # 创建序列和特征维度的偏移量
    seq_offsets = seq_start + tl.arange(0, BLOCK_SEQ)
    feat_offsets = feat_start + tl.arange(0, BLOCK_FEAT)
    
    # 创建掩码，处理边界
    seq_mask = seq_offsets < seq_len
    feat_mask = feat_offsets < feat_dim
    
    # 初始化累加器 - 为每个(序列,特征)位置创建
    accumulator = tl.zeros((BLOCK_SEQ, BLOCK_FEAT), dtype=DTYPE)
    
    # 遍历注意力头维度(维度1)
    for head_idx in range(0, num_heads):
        # 计算输入内存偏移量
        input_offsets = (
            seq_offsets[:, None] * input_stride_seq + 
            head_idx * input_stride_head + 
            feat_offsets[None, :] * input_stride_feat
        )
        
        # 加载数据块
        block = tl.load(
            input_ptr + input_offsets,
            mask=seq_mask[:, None] & feat_mask[None, :],
            other=0.0
        )
        
        # 累加到累加器
        accumulator += block
    
    # 计算输出内存偏移量
    output_offsets = (
        seq_offsets[:, None] * output_stride_seq + 
        feat_offsets[None, :] * output_stride_feat
    )
    
    # 存储结果
    tl.store(
        output_ptr + output_offsets,
        accumulator,
        mask=seq_mask[:, None] & feat_mask[None, :]
    )


def sum_dim(
    input: torch.Tensor,
    dim = None,
    keepdim: bool = False, 
    *,
    dtype = None,
) -> torch.Tensor:
    """
    针对[seq_len, num_heads, feat_dim]形状优化的维度1求和
    
    参数:
        input: 输入张量，形状为[seq_len, num_heads, feat_dim]
        keepdim: 是否保留归约维度
        dtype: 输出数据类型(可选)
    
    返回:
        归约后的张量，形状为:
            keepdim=True: [seq_len, 1, feat_dim]
            keepdim=False: [seq_len, feat_dim]
    """
    if input.dim() != 3:
        return
    assert input.dim() == 3, "输入张量必须是3维"
    seq_len, num_heads, feat_dim = input.shape
    
    # 准备输出张量
    output_shape = (seq_len, feat_dim) if not keepdim else (seq_len, 1, feat_dim)
    output_dtype = dtype if dtype is not None else input.dtype
    output = torch.empty(output_shape, device=input.device, dtype=output_dtype)
    
    # 确定Triton数据类型
    if input.dtype == torch.float16:
        tl_dtype = tl.float16
    elif input.dtype == torch.float32:
        tl_dtype = tl.float32
    elif input.dtype == torch.bfloat16:
        tl_dtype = tl.bfloat16
    else:
        raise ValueError(f"不支持的数据类型: {input.dtype}")
    
    # 获取内存布局信息
    if input.is_contiguous():
        input_stride_seq = input.stride(0)
        input_stride_head = input.stride(1)
        input_stride_feat = input.stride(2)
    else:
        # 对于非连续输入，创建连续视图
        input = input.contiguous()
        input_stride_seq = input.stride(0)
        input_stride_head = input.stride(1)
        input_stride_feat = input.stride(2)
    
    if output.is_contiguous():
        output_stride_seq = output.stride(0)
        if keepdim:
            output_stride_feat = output.stride(2)
        else:
            output_stride_feat = output.stride(1)
    else:
        # 对于非连续输出，创建连续视图
        output = output.contiguous()
        output_stride_seq = output.stride(0)
        if keepdim:
            output_stride_feat = output.stride(2)
        else:
            output_stride_feat = output.stride(1)
    
    # 配置内核参数
    # 针对A100/H100优化
    BLOCK_SEQ = 32  # 每个程序处理的序列长度块
    BLOCK_FEAT = 128  # 每个程序处理的特征维度块
    
    # 计算网格大小
    grid_seq = triton.cdiv(seq_len, BLOCK_SEQ)
    grid_feat = triton.cdiv(feat_dim, BLOCK_FEAT)
    grid = (grid_seq, grid_feat)
    
    # 启动内核
    sum_dim1_kernel[grid](
        input, output,
        # 张量元数据
        seq_len, num_heads, feat_dim,
        # 内存布局
        input_stride_seq, input_stride_head, input_stride_feat,
        output_stride_seq, output_stride_feat,
        # 内核配置
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_FEAT=BLOCK_FEAT,
        DTYPE=tl_dtype
    )
    
    return output







def sum(inp, *, dtype=None):
    logger.debug("GEMS SUM")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int64)
            dtype = torch.int64
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        sum_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        sum_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


def sum_out(inp, *, dtype=None, out):
    logger.debug("GEMS SUM_OUT")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int64)
            dtype = torch.int64
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        sum_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        sum_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


def sum_out(inp, *, dtype=None, out):
    logger.debug("GEMS SUM_OUT")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int64)
            dtype = torch.int64
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        sum_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        sum_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


# def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
#     logger.debug("GEMS SUM DIM")
#     if dtype is None:
#         dtype = inp.dtype
#         if dtype is torch.bool:
#             dtype = torch.int64

#     if dim == []:
#         if not keepdim:
#             return sum(inp, dtype=dtype)
#         else:
#             dim_num = inp.ndim
#             return torch.reshape(sum(inp, dtype=dtype), [1] * dim_num)

#     shape = list(inp.shape)
#     dim = [d % inp.ndim for d in dim]
#     inp = dim_compress(inp, dim)
#     N = 1
#     for i in dim:
#         N *= shape[i]
#         shape[i] = 1
#     M = inp.numel() // N

#     out = torch.empty(shape, dtype=dtype, device=inp.device)

#     grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
#     with torch_device_fn.device(inp.device):
#         sum_kernel[grid](inp, out, M, N)
#     if not keepdim:
#         out = out.squeeze(dim=dim)
#     return out


def sum_dim_out(inp, dim=None, keepdim=False, *, dtype=None, out):
    logger.debug("GEMS SUM_DIM_OUT")
    # print(f"sum_dim_out: {inp.shape}, {inp.dtype}, {dim}, {keepdim}, {dtype}, {out.shape}")
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64

    if dim == []:
        if not keepdim:
            return sum_out(inp, dtype=dtype, out=out)
        else:
            dim_num = inp.ndim
            return torch.reshape(sum_out(inp, dtype=dtype, out=out), [1] * dim_num)

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out = torch.empty(shape, dtype=dtype, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        sum_kernel[grid](inp, out, M, N)
    if not keepdim:
        out.squeeze_(dim=dim)
    return out


def sum_dim_out(inp, dim=None, keepdim=False, *, dtype=None, out):
    logger.debug("GEMS SUM_DIM_OUT")
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64

    if dim == []:
        if not keepdim:
            return sum_out(inp, dtype=dtype, out=out)
        else:
            dim_num = inp.ndim
            return torch.reshape(sum_out(inp, dtype=dtype, out=out), [1] * dim_num)

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        sum_kernel[grid](inp, out, M, N)
    if not keepdim:
        out.squeeze_(dim=dim)
    return out
