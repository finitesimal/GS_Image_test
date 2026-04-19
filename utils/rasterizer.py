import os
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension as cpp_extension

_sources = [os.path.join(os.path.dirname(__file__), "../cuda/main.cu")]
_includes = [os.path.join(os.path.dirname(__file__), "../glm/")]
_build_dir = os.path.join(os.path.dirname(__file__), "../compiled/")

os.makedirs(_build_dir, exist_ok=True)

_C = cpp_extension.load(
    name="_gaussian_image_cuda",
    sources=_sources,
    extra_cuda_cflags=["-allow-unsupported-compiler"],
    extra_include_paths=_includes,
    build_directory=_build_dir,
    verbose=True,
)

def check_input(t):
    assert t.dtype == torch.float
    assert t.is_cuda
    assert t.is_contiguous()

class CudaWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, num, param, center, weight, bound, size, origin_resolution):
        size_x, size_y = size
        origin_x, origin_y, resolution = origin_resolution
        result = torch.zeros((size_x, size_y, 4), dtype=torch.float, device="cuda")

        ctx.save_for_backward(param, center, weight, bound)
        ctx.saved_parameters = (num, size, origin_resolution)
        _C.forward(param, center, weight, bound, origin_x, origin_y, resolution, result)
        return result
    
    @staticmethod
    def backward(ctx, loss):
        param, center, weight, bound = ctx.saved_tensors
        num, size, origin_resolution = ctx.saved_parameters
        origin_x, origin_y, resolution = origin_resolution

        param_grad = torch.zeros_like(param)
        center_grad = torch.zeros_like(center)
        weight_grad = torch.zeros_like(weight)

        _C.backward(param, center, weight, bound, origin_x, origin_y, resolution, loss, param_grad, center_grad, weight_grad)
        return None, param_grad, center_grad, weight_grad, None, None, None

def rasterize(sigma, mu, weight, size, origin_resolution):
    check_input(sigma)
    check_input(mu)
    check_input(weight)
    num = sigma.size(0)
    assert sigma.shape == (num, 3)
    assert mu.shape == (num, 2)
    assert weight.shape == (num, 4)

    a = sigma[:, 0].unsqueeze(1)
    b = sigma[:, 1].unsqueeze(1)
    c = sigma[:, 2].unsqueeze(1)
    param = torch.cat([c, -b, a], dim=1) / (a * c - b * b)
    bound = 3.0 * torch.cat([a, c], dim=1).sqrt()

    return CudaWrapper.apply(num, param, mu, weight, bound, size, origin_resolution)

def rasterize_blend(sigma, mu, color, weight, size, origin_resolution):
    check_input(sigma)
    check_input(mu)
    check_input(color)
    check_input(weight)
    num = sigma.size(0)
    assert sigma.shape == (num, 3)
    assert mu.shape == (num, 2)
    assert color.shape == (num, 3)
    assert weight.shape == (num,)

    weight = weight.unsqueeze(1)
    weight = torch.cat([color * weight, weight], dim=1)
    result = rasterize(sigma, mu, weight, size, origin_resolution)
    return result[:, 0 : 3] / result[:, 3].unsqueeze(1)