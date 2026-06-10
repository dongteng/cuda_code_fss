import torch
from torch import Tensor


#这个__all__变量定义了当使用from module import *语句时，模块公开的名称列表。
#Python 会默认导入 ops.py 中所有不以下划线 _ 开头的名字，但如果文件里定义了 __all__，就只会导入 __all__ 列表里的名字。
__all__ = ["mymuladd", "myadd_out", "mysgemm"]


#python封装接口函数
#torch.ops.extension_cpp.mymuladd.default：
# 访问 C++/CUDA 扩展模块 _C 的 operator。
# default 是默认实现的命名（torch 2.x 新库 API 风格）。
# 本质是：
# 用户调用 Python 函数 → 调用 C++ 扩展 → GPU/CPU 执行 fused kernel。
def mymuladd(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.extension_cpp.mymuladd.default(a, b, c)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
#原理与作用：

# FakeTensor 又叫 “meta kernel” 或 “abstract impl”。
# 作用：
# 只描述输出 tensor 的 shape、dtype、device，不真正计算。
# PyTorch 编译器（torch.compile）在做 graph tracing 或优化时会用。
# 代码做了几个检查：
# 两个输入 shape 是否相等
# dtype 是否为 float
# device 是否相同
# 返回一个 空 tensor 作为输出的“占位符”。
@torch.library.register_fake("extension_cpp::mymuladd")
def _(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = torch.ops.extension_cpp.mymul.default(grad, b)
    if ctx.needs_input_grad[1]:
        grad_b = torch.ops.extension_cpp.mymul.default(grad, a)
    return grad_a, grad_b, None


def _setup_context(ctx, inputs, output):
    a, b, c = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "extension_cpp::mymuladd", _backward, setup_context=_setup_context)


@torch.library.register_fake("extension_cpp::mymul")
def _(a, b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


def myadd_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    """Writes a + b into out"""
    torch.ops.extension_cpp.myadd_out.default(a, b, out)


def mysgemm(a: Tensor, b: Tensor, alpha: float, beta: float) -> Tensor:
    """Performs a GEMM operation: alpha * (A @ B) + beta * C"""
    return torch.ops.extension_cpp.mysgemm.default(a, b, alpha, beta)