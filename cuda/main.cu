#include <torch/extension.h>

#define GLM_FORCE_CUDA
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/geometric.hpp>

using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::ivec2;

__device__ void atomicAdd(vec4* ptr, vec4 val) {
    atomicAdd(&(ptr->x), val.x);
    atomicAdd(&(ptr->y), val.y);
    atomicAdd(&(ptr->z), val.z);
    atomicAdd(&(ptr->w), val.w);
}

__device__ __forceinline__ float gaussian_value(const vec3 param, const vec2 l){
    const float quadratic = l.x * l.x * param[0] + 2.0f * l.x * l.y * param[1] + l.y * l.y * param[2];
    return expf(-0.5f * quadratic);
}

__global__ void forward_kernel(
    const int num_GS,
    const vec3 *GS_param,
    const vec2 *GS_center,
    const vec4 *GS_weight,
    const vec2 *GS_bound,

    const ivec2 pixel_size,
    const vec2 pixel_origin,
    const float pixel_resolution,
    vec4 *pixel
){
    const int idx_GS = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx_GS >= num_GS){
        return;
    }

    const vec3 param = GS_param[idx_GS];
    const vec2 center = GS_center[idx_GS];
    const vec4 weight = GS_weight[idx_GS];
    const vec2 bound = GS_bound[idx_GS];

    const vec2 start = (center - pixel_origin - bound) / pixel_resolution;
    const vec2 end = (center - pixel_origin + bound) / pixel_resolution;

    const int sx = max(__float2int_rn(start.x), 0);
    const int ex = min(__float2int_rn(end.x), pixel_size.x);
    const int sy = max(__float2int_rn(start.y), 0);
    const int ey = min(__float2int_rn(end.y), pixel_size.y);

    for(int px = sx; px < ex; ++px){
        for(int py = sy; py < ey; ++py){
            const int idx_pixel = px * pixel_size.y + py;
            const vec2 pos = (vec2(px, py) + 0.5f) * pixel_resolution + pixel_origin;
            const vec2 l = center - pos;
            const float value = gaussian_value(param, l);
            const vec4 res = weight * value;
            atomicAdd(&pixel[idx_pixel], res);
        }
    }
}

__global__ void backward_kernel(
    const int num_GS,
    const vec3 *GS_param,
    const vec2 *GS_center,
    const vec4 *GS_weight,
    const vec2 *GS_bound,

    const ivec2 pixel_size,
    const vec2 pixel_origin,
    const float pixel_resolution,
    const vec4 *pixel_grad,

    vec3 *GS_param_grad,
    vec2 *GS_center_grad,
    vec4 *GS_weight_grad
){
    const int idx_GS = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx_GS >= num_GS){
        return;
    }

    const vec3 param = GS_param[idx_GS];
    const vec2 center = GS_center[idx_GS];
    const vec4 weight = GS_weight[idx_GS];
    const vec2 bound = GS_bound[idx_GS];

    vec3 param_grad{};
    vec2 center_grad{};
    vec4 weight_grad{};

    const vec2 start = (center - pixel_origin - bound) / pixel_resolution;
    const vec2 end = (center - pixel_origin + bound) / pixel_resolution;

    const int sx = max(__float2int_rn(start.x), 0);
    const int ex = min(__float2int_rn(end.x), pixel_size.x);
    const int sy = max(__float2int_rn(start.y), 0);
    const int ey = min(__float2int_rn(end.y), pixel_size.y);

    for(int px = sx; px < ex; ++px){
        for(int py = sy; py < ey; ++py){
            const int idx_pixel = px * pixel_size.y + py;
            const vec4 grad = pixel_grad[idx_pixel];

            const vec2 pos = (vec2(px, py) + 0.5f) * pixel_resolution + pixel_origin;
            const vec2 l = center - pos;
            const float value = gaussian_value(param, l);

            param_grad += glm::dot(grad, weight) * value * -vec3(0.5f * l.x * l.x, l.x * l.y, 0.5f * l.y * l.y);
            center_grad += glm::dot(grad, weight) * value * -vec2(param[0] * l.x + param[1] * l.y, param[1] * l.x + param[2] * l.y);
            weight_grad += grad * value;
        }
    }
    GS_param_grad[idx_GS] = param_grad;
    GS_center_grad[idx_GS] = center_grad;
    GS_weight_grad[idx_GS] = weight_grad;
}

