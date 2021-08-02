#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

PYBIND11_MAKE_OPAQUE(std::map<std::string, int>);

std::vector<torch::Tensor> sdf_renderer_cuda_forward(torch::Tensor& sdf,
                                                     torch::Tensor& position,
                                                     torch::Tensor& orientation,
                                                     torch::Tensor& inv_scale,
                                                     const unsigned int width,
                                                     const unsigned int height,
                                                     const float cx,
                                                     const float cy,
                                                     const float fx,
                                                     const float fy,
                                                     const float threshold);

std::vector<torch::Tensor> sdf_renderer_cuda_backward(torch::Tensor& grad_depth_image,
                                                      torch::Tensor& depth_image,
                                                      torch::Tensor& sdf,
                                                      torch::Tensor& position,
                                                      torch::Tensor& orientation,
                                                      torch::Tensor& inv_scale,
                                                      const int width,
                                                      const int height,
                                                      const float cx,
                                                      const float cy,
                                                      const float fx,
                                                      const float fy);

std::vector<torch::Tensor> sdf_renderer_forward(torch::Tensor& sdf,
                                                torch::Tensor& position,
                                                torch::Tensor& orientation,
                                                torch::Tensor& inv_scale,
                                                const int width,
                                                const int height,
                                                const float cx,
                                                const float cy,
                                                const float fx,
                                                const float fy,
                                                const float threshold) {
  CHECK_INPUT(sdf);
  CHECK_INPUT(position);
  CHECK_INPUT(orientation);
  CHECK_INPUT(inv_scale);
  // see: https://discuss.pytorch.org/t/c-cuda-extension-with-multiple-gpus/91241
  const at::cuda::OptionalCUDAGuard device_guard(device_of(sdf));
  return sdf_renderer_cuda_forward(sdf, position, orientation, inv_scale, width, height,
                                   cx, cy, fx, fy, threshold);
}

std::vector<torch::Tensor> sdf_renderer_backward(torch::Tensor& grad_depth_image,
                                                 torch::Tensor& depth_image,
                                                 torch::Tensor& sdf,
                                                 torch::Tensor& position,
                                                 torch::Tensor& orientation,
                                                 torch::Tensor& inv_scale,
                                                 const int width,
                                                 const int height,
                                                 const float cx,
                                                 const float cy,
                                                 const float fx,
                                                 const float fy) {
  CHECK_INPUT(grad_depth_image);
  CHECK_INPUT(depth_image);
  CHECK_INPUT(sdf);
  CHECK_INPUT(position);
  CHECK_INPUT(orientation);
  CHECK_INPUT(inv_scale);
  // assuming square pixels
  const at::cuda::OptionalCUDAGuard device_guard(device_of(sdf));
  return sdf_renderer_cuda_backward(grad_depth_image, depth_image, sdf, position,
                                    orientation, inv_scale, width, height, cx, cy, fx,
                                    fy);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sdf_renderer_forward, "SDF Renderer forward (CUDA)");
  m.def("backward", &sdf_renderer_backward, "SDF Renderer backward (CUDA)");
}
