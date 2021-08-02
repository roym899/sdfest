#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <vector>

namespace {

const unsigned int THREADS_PER_DIM = 16;

#define ATV3(S, V) (S[V.x][V.y][V.z])

template <typename scalar_t>
struct Quaternion;

template <typename scalar_t>
struct Vector3 {
  scalar_t x, y, z;
  __device__ Vector3() = default;
  __device__ Vector3(scalar_t x, scalar_t y, scalar_t z) : x(x), y(y), z(z) {}
  __device__ Vector3(const Vector3& rhs) : x(rhs.x), y(rhs.y), z(rhs.z) {}

  __device__ Vector3& operator+=(const Vector3& rhs) {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    return *this;
  }

  __device__ Vector3& operator-=(const Vector3& rhs) {
    x -= rhs.x;
    y -= rhs.y;
    z -= rhs.z;
    return *this;
  }

  __device__ Vector3& operator*=(const scalar_t& rhs) {
    x *= rhs;
    y *= rhs;
    z *= rhs;
    return *this;
  }

  __device__ Vector3& rotate(const Quaternion<scalar_t>& q) {
    Vector3 temp = q.apply(*this);
    x = temp.x;
    y = temp.y;
    z = temp.z;
    return *this;
  }
};

template <typename scalar_t>
inline __device__ Vector3<scalar_t> operator+(Vector3<scalar_t> lhs,
                                              const Vector3<scalar_t>& rhs) {
  lhs += rhs;
  return lhs;
}

template <typename scalar_t>
inline __device__ Vector3<scalar_t> operator-(Vector3<scalar_t> lhs,
                                              const Vector3<scalar_t>& rhs) {
  lhs -= rhs;
  return lhs;
}

template <typename scalar_t>
inline __device__ Vector3<scalar_t> operator*(scalar_t lhs, Vector3<scalar_t> rhs) {
  rhs *= lhs;
  return rhs;
}

template <typename scalar_t>
inline __device__ Vector3<scalar_t> operator*(Vector3<scalar_t> lhs, scalar_t rhs) {
  return rhs * lhs;
}

template <typename scalar_t>
inline __device__ scalar_t operator*(const Vector3<scalar_t>& lhs,
                                     const Vector3<scalar_t>& rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

template <typename scalar_t>
struct Quaternion {
  __device__ Quaternion() = default;
  __device__ Quaternion(const Quaternion& rhs)
      : x(rhs.x), y(rhs.y), z(rhs.z), w(rhs.w) {}
  __device__ Quaternion(scalar_t x, scalar_t y, scalar_t z, scalar_t w)
      : x(x), y(y), z(z), w(w) {}
  scalar_t x, y, z, w;

  __device__ void normalize() {
    scalar_t norm = r_qrt(x * x + y * y + z * z + w * w);
    *this *= norm;
  }

  __device__ void inverse() {
    x = -x;
    y = -y;
    z = -z;
  }

  __device__ Quaternion inversed() {
    Quaternion temp(*this);
    temp.inverse();
    return temp;
  }

  __device__ Vector3<scalar_t> apply(const Vector3<scalar_t>& rhs) const {
    Vector3<scalar_t> temp;
    temp.x = (1 - 2 * (y * y + z * z)) * rhs.x + 2 * (x * y - w * z) * rhs.y +
             2 * (x * z + w * y) * rhs.z;
    temp.y = 2 * (x * y + w * z) * rhs.x + (1 - 2 * (x * x + z * z)) * rhs.y +
             2 * (y * z - w * x) * rhs.z;
    temp.z = 2 * (x * z - w * y) * rhs.x + 2 * (y * z + w * x) * rhs.y +
             (1 - 2 * (x * x + y * y)) * rhs.z;
    return temp;
  }

  __device__ Quaternion& operator*=(const scalar_t& rhs) {
    x *= rhs;
    y *= rhs;
    z *= rhs;
    w *= rhs;
    return *this;
  }
};

/**
 * Compute normalized direction vector for a pixel.
 *
 * TODO
 */
template <typename scalar_t>
__device__ __forceinline__ Vector3<scalar_t> direction_from_pixel(
    const unsigned int row,
    const unsigned int col,
    const float cx,
    const float cy,
    const float fx,
    const float fy) {
  Vector3<scalar_t> d;
  d.x = (col + 0.5 - cx) / fx;
  d.y = - (row + 0.5 - cy) / fy;
  d.z = -1.0;

  // normalize direction vector
  scalar_t r_norm = rsqrt(d.x * d.x + d.y * d.y + 1);
  d *= r_norm;
  return d;
}

template <typename scalar_t>
__device__ __forceinline__ bool find_obb_ray_t(const Vector3<scalar_t>& d,
                                               const Vector3<scalar_t>& p_o2w,
                                               const Quaternion<scalar_t>& q_o2w,
                                               const scalar_t scale,
                                               scalar_t& t_min,
                                               scalar_t& t_max) {
  // TODO: sth proper here, numeric limits?
  t_min = -1e-10;
  t_max = 1e10;
  Vector3<scalar_t> a_c(0, 0, 0);
  Vector3<scalar_t> a[3]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  a_c = a_c.rotate(q_o2w) + p_o2w;
  Vector3<scalar_t> p(a_c);  // a_c - ray origin ( (0,0,0) here )

  for (size_t i = 0; i < 3; ++i) {
    a[i].rotate(q_o2w);
    scalar_t e = a[i] * p;
    scalar_t f = a[i] * d;

    if (abs(f) > 1e-20) {
      scalar_t t_1 = (e + scale) / f;
      scalar_t t_2 = (e - scale) / f;
      if (t_1 > t_2) {
        scalar_t temp = t_2;
        t_2 = t_1;
        t_1 = temp;
      }
      t_min = max(t_min, t_1);
      t_max = min(t_max, t_2);
      if (t_min > t_max || t_max < 0) return false;
    } else if (-e > scale || -e < -scale) {
      return false;
    }
  }

  t_min = max(t_min, 0.0);
  return true;
}

template <typename scalar_t>
__device__ __forceinline__ Vector3<size_t> point_2_index(const Vector3<scalar_t>& point,
                                                         int resolution) {
  return Vector3<size_t>{
      max(0, min(resolution - 2, __float2int_rd((point.x + scalar_t(1.0)) *
                                                (resolution - 1) * scalar_t(0.5)))),
      max(0, min(resolution - 2, __float2int_rd((point.y + scalar_t(1.0)) *
                                                (resolution - 1) * scalar_t(0.5)))),
      max(0, min(resolution - 2, __float2int_rd((point.z + scalar_t(1.0)) *
                                                (resolution - 1) * scalar_t(0.5)))),
  };
}

template <typename scalar_t>
__device__ __forceinline__ Vector3<scalar_t> index_2_point(const Vector3<size_t>& point,
                                                           scalar_t grid_size) {
  return Vector3<scalar_t>{point.x * grid_size - scalar_t(1.0),
                           point.y * grid_size - scalar_t(1.0),
                           point.z * grid_size - scalar_t(1.0)};
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t trilinear(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>
        sdf,
    Vector3<scalar_t> point,
    scalar_t inv_scale) {
  // TODO add resolution grid_size params
  point *= inv_scale;
  auto base_index = point_2_index(point, 64);
  Vector3<size_t> n_i[]{{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                        {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
  for (auto& n : n_i) n += base_index;
  auto pos_0 = index_2_point<scalar_t>(base_index, 2.0 / (64 - 1));
  Vector3<scalar_t> off = scalar_t((64 - 1) / 2.0) * (point - pos_0);
  auto c00 = ATV3(sdf, n_i[0]) * (1 - off.x) + ATV3(sdf, n_i[4]) * off.x;
  auto c01 = ATV3(sdf, n_i[1]) * (1 - off.x) + ATV3(sdf, n_i[5]) * off.x;
  auto c10 = ATV3(sdf, n_i[2]) * (1 - off.x) + ATV3(sdf, n_i[6]) * off.x;
  auto c11 = ATV3(sdf, n_i[3]) * (1 - off.x) + ATV3(sdf, n_i[7]) * off.x;
  auto c0 = c00 * (1 - off.y) + c10 * off.y;
  auto c1 = c01 * (1 - off.y) + c11 * off.y;

  return c0 * (1 - off.z) + c1 * off.z;
}

template <typename scalar_t>
__global__ void sdf_renderer_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>
        sdf,
    const scalar_t* __restrict__ p_o2w_data,
    const scalar_t* __restrict__ q_o2w_data,
    const scalar_t* __restrict__ inv_scale_data,
    const unsigned int width,
    const unsigned int height,
    const float cx,
    const float cy,
    const float fx,
    const float fy,
    const float threshold,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>
        depth_image) {
  const unsigned int row = threadIdx.y + blockIdx.y * THREADS_PER_DIM;
  const unsigned int col = threadIdx.x + blockIdx.x * THREADS_PER_DIM;
  scalar_t scale = 1. / *inv_scale_data;
  scalar_t inv_scale = *inv_scale_data;
  Vector3<scalar_t> p_o2w(p_o2w_data[0], p_o2w_data[1], p_o2w_data[2]);
  Quaternion<scalar_t> q_o2w(q_o2w_data[0], q_o2w_data[1], q_o2w_data[2],
                             q_o2w_data[3]);
  auto q_w2o(q_o2w.inversed());

  if (row < height && col < width) {
    // create ray for current pixel
    auto d = direction_from_pixel<scalar_t>(row, col, cx, cy, fx, fy);

    // check if ray intersects
    scalar_t t_min, t_max;
    auto intersects = find_obb_ray_t<scalar_t>(d, p_o2w, q_o2w, scale, t_min, t_max);

    if (intersects) {
      // sphere trace in object coordinates
      scalar_t current_t = t_min;
      auto d_o = d;
      d_o.rotate(q_w2o);
      auto origin_o = Vector3<scalar_t>(0, 0, 0);
      origin_o -= p_o2w;
      origin_o.rotate(q_w2o);

      while (current_t < t_max) {
        // transform world point to object point
        auto current_point = origin_o + current_t * d_o;
        scalar_t distance = trilinear(sdf, std::move(current_point), inv_scale) * scale;
        if (distance < threshold * current_t) {
          depth_image[row][col] = -current_t * d.z;
          break;
        } else {
          current_t += distance;
        }
      }
    } else {
      depth_image[row][col] = 0;
    }
  }
}

template <typename scalar_t>
__global__ void sdf_renderer_cuda_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>
        grad_depth_image,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>
        depth_image,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>
        sdf,
    const scalar_t* __restrict__ p_o2w_data,
    const scalar_t* __restrict__ q_o2w_data,
    const scalar_t* __restrict__ inv_scale_data,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> grad_sdf,
    scalar_t* __restrict__ grad_p,
    scalar_t* __restrict__ grad_q,
    scalar_t* __restrict__ grad_inv_scale,
    const unsigned int width,
    const unsigned int height,
    const float cx,
    const float cy,
    const float fx,
    const float fy) {
  // TODO: maybe change the logic here for backward to reduce blocking
  const unsigned int row = threadIdx.y + blockIdx.y * THREADS_PER_DIM;
  const unsigned int col = threadIdx.x + blockIdx.x * THREADS_PER_DIM;
  scalar_t scale = 1. / *inv_scale_data;
  scalar_t inv_scale = *inv_scale_data;
  // TODO params for this
  scalar_t grid_size = 2.0 / (64 - 1);
  scalar_t grid_size_inv = 1. / grid_size;
  Vector3<scalar_t> p_o2w(p_o2w_data[0], p_o2w_data[1], p_o2w_data[2]);
  Quaternion<scalar_t> q_o2w(q_o2w_data[0], q_o2w_data[1], q_o2w_data[2],
                             q_o2w_data[3]);
  auto q_w2o(q_o2w.inversed());

  if (row < height && col < width && depth_image[row][col] != 0) {
    // create ray for current pixel
    auto z = depth_image[row][col];
    auto d = direction_from_pixel<scalar_t>(row, col, cx, cy, fx, fy);
    auto t = -z / d.z; // need to inverse sign because depth image is positive 
    auto d_o = d;
    d_o.rotate(q_w2o);
    auto origin_o = Vector3<scalar_t>(0, 0, 0);
    origin_o -= p_o2w;
    origin_o.rotate(q_w2o);
    auto x = t * d;               // point in world coordinates
    auto o = origin_o + t * d_o;  // point in object coordinates
    auto no = o * inv_scale;      // point in normalized object coordinates
    auto base_index = point_2_index(no, 64);
    Vector3<size_t> n_i[]{{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                          {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
    for (auto& n : n_i) n += base_index;
    auto pos_0 = index_2_point<scalar_t>(
        base_index, grid_size);  // cell origin in normalized object coordinates
    Vector3<scalar_t> c =
        grid_size_inv * (no - pos_0);  // point in normalized cell coordinate
    auto c000 = ATV3(sdf, n_i[0]);
    auto c001 = ATV3(sdf, n_i[1]);
    auto c010 = ATV3(sdf, n_i[2]);
    auto c011 = ATV3(sdf, n_i[3]);
    auto c100 = ATV3(sdf, n_i[4]);
    auto c101 = ATV3(sdf, n_i[5]);
    auto c110 = ATV3(sdf, n_i[6]);
    auto c111 = ATV3(sdf, n_i[7]);
    auto c00 = c000 * (1 - c.x) + c100 * c.x;
    auto c01 = c001 * (1 - c.x) + c101 * c.x;
    auto c10 = c010 * (1 - c.x) + c110 * c.x;
    auto c11 = c011 * (1 - c.x) + c111 * c.x;
    auto c0 = c00 * (1 - c.y) + c10 * c.y;
    auto c1 = c01 * (1 - c.y) + c11 * c.y;
    auto t_diff = c0 * (1 - c.z) + c1 * c.z;

    // compute the derivatives
    auto f = scale * abs(d.z);  // factor used in many subsequenet operations
    atomicAdd(&ATV3(grad_sdf, n_i[0]),
              grad_depth_image[row][col] * (1 - c.x) * (1 - c.y) * c.z * f);
    atomicAdd(&ATV3(grad_sdf, n_i[1]),
              grad_depth_image[row][col] * (1 - c.x) * c.y * (1 - c.z) * f);
    atomicAdd(&ATV3(grad_sdf, n_i[2]),
              grad_depth_image[row][col] * (1 - c.x) * c.y * c.z * f);
    atomicAdd(&ATV3(grad_sdf, n_i[3]),
              grad_depth_image[row][col] * c.x * (1 - c.y) * (1 - c.z) * f);
    atomicAdd(&ATV3(grad_sdf, n_i[4]),
              grad_depth_image[row][col] * c.x * (1 - c.y) * c.z * f);
    atomicAdd(&ATV3(grad_sdf, n_i[5]),
              grad_depth_image[row][col] * c.x * (1 - c.y) * c.z * f);
    atomicAdd(&ATV3(grad_sdf, n_i[6]),
              grad_depth_image[row][col] * c.x * c.y * (1 - c.z) * f);
    atomicAdd(&ATV3(grad_sdf, n_i[7]),
              grad_depth_image[row][col] * c.x * c.y * c.z * f);


    auto s = inv_scale * grid_size_inv;
    auto ox = x - p_o2w;  // vector from object center to point in world coordinates
    Vector3<scalar_t> dc_x((2 * (q_o2w.y * q_o2w.y + q_o2w.z * q_o2w.z) - 1) * s,
                           2 * (q_o2w.w * q_o2w.z - q_o2w.x * q_o2w.y) * s,
                           -2 * (q_o2w.x * q_o2w.z + q_o2w.w * q_o2w.y) * s);
    Vector3<scalar_t> dc_y(-2 * (q_o2w.x * q_o2w.y + q_o2w.w * q_o2w.z) * s,
                           (2 * (q_o2w.x * q_o2w.x + q_o2w.z * q_o2w.z) - 1) * s,
                           2 * (q_o2w.w * q_o2w.x - q_o2w.y * q_o2w.z) * s);
    Vector3<scalar_t> dc_z(2 * (q_o2w.w * q_o2w.y - q_o2w.x * q_o2w.z) * s,
                           -2 * (q_o2w.y * q_o2w.z + q_o2w.w * q_o2w.x) * s,
                           (2 * (q_o2w.x * q_o2w.x + q_o2w.y * q_o2w.y) - 1) * s);
    Vector3<scalar_t> dc_qw((2 * q_o2w.w * ox.x + 2 * q_o2w.z * ox.y -
                             2 * q_o2w.y * ox.z - 2 * q_o2w.w * o.x) *
                                s,
                            (-2 * q_o2w.z * ox.x + 2 * q_o2w.w * ox.y +
                             2 * q_o2w.x * ox.z - 2 * q_o2w.w * o.y) *
                                s,
                            (2 * q_o2w.y * ox.x - 2 * q_o2w.x * ox.y +
                             2 * q_o2w.w * ox.z - 2 * q_o2w.w * o.z) *
                                s);
    Vector3<scalar_t> dc_qx((2 * q_o2w.x * ox.x + 2 * q_o2w.y * ox.y +
                             2 * q_o2w.z * ox.z - 2 * q_o2w.x * o.x) *
                                s,
                            (2 * q_o2w.y * ox.x - 2 * q_o2w.x * ox.y +
                             2 * q_o2w.w * ox.z - 2 * q_o2w.x * o.y) *
                                s,
                            (2 * q_o2w.z * ox.x - 2 * q_o2w.w * ox.y -
                             2 * q_o2w.x * ox.z - 2 * q_o2w.x * o.z) *
                                s);
    Vector3<scalar_t> dc_qy((-2 * q_o2w.y * ox.x + 2 * q_o2w.x * ox.y -
                             2 * q_o2w.w * ox.z - 2 * q_o2w.y * o.x) *
                                s,
                            (2 * q_o2w.x * ox.x + 2 * q_o2w.y * ox.y +
                             2 * q_o2w.z * ox.z - 2 * q_o2w.y * o.y) *
                                s,
                            (2 * q_o2w.w * ox.x + 2 * q_o2w.z * ox.y -
                             2 * q_o2w.y * ox.z - 2 * q_o2w.y * o.z) *
                                s);
    Vector3<scalar_t> dc_qz((-2 * q_o2w.z * ox.x + 2 * q_o2w.w * ox.y +
                             2 * q_o2w.x * ox.z - 2 * q_o2w.z * o.x) *
                                s,
                            (-2 * q_o2w.w * ox.x - 2 * q_o2w.z * ox.y +
                             2 * q_o2w.y * ox.z - 2 * q_o2w.z * o.y) *
                                s,
                            (2 * q_o2w.x * ox.x + 2 * q_o2w.y * ox.y +
                             2 * q_o2w.z * ox.z - 2 * q_o2w.z * o.z) *
                                s);
    Vector3<scalar_t> dc_sinv(o * grid_size_inv);

    Vector3<scalar_t>* dc[] = {&dc_x,  &dc_y,  &dc_z,  &dc_qx,
                               &dc_qy, &dc_qz, &dc_qw, &dc_sinv};
    scalar_t dz[8];

    for (size_t i = 0; i < 8; ++i) {
      auto dc00 = -c000 * dc[i]->x + c100 * dc[i]->x;
      auto dc01 = -c001 * dc[i]->x + c101 * dc[i]->x;
      auto dc10 = -c010 * dc[i]->x + c110 * dc[i]->x;
      auto dc11 = -c011 * dc[i]->x + c111 * dc[i]->x;

      auto dc0 = dc00 * (1 - c.y) - c00 * dc[i]->y + dc10 * c.y + c10 * dc[i]->y;
      auto dc1 = dc01 * (1 - c.y) - c01 * dc[i]->y + dc11 * c.y + c11 * dc[i]->y;

      auto dtdiff = dc0 * (1 - c.z) - c0 * dc[i]->z + dc1 * c.z + c1 * dc[i]->z;

      dz[i] = scale * dtdiff * abs(d.z);
    }
    dz[7] -= (t_diff * scale * scale) * abs(d.z);  // product rule for scale

    atomicAdd(&grad_p[0], dz[0] * grad_depth_image[row][col]);
    atomicAdd(&grad_p[1], dz[1] * grad_depth_image[row][col]);
    atomicAdd(&grad_p[2], dz[2] * grad_depth_image[row][col]);
    atomicAdd(&grad_q[0], dz[3] * grad_depth_image[row][col]);
    atomicAdd(&grad_q[1], dz[4] * grad_depth_image[row][col]);
    atomicAdd(&grad_q[2], dz[5] * grad_depth_image[row][col]);
    atomicAdd(&grad_q[3], dz[6] * grad_depth_image[row][col]);
    atomicAdd(grad_inv_scale, dz[7] * grad_depth_image[row][col]);
  }
}

}  // namespace

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
                                                     const float threshold) {
  /* auto begin = std::chrono::steady_clock::now(); */
  auto depth_image = torch::zeros(
      {height, width}, torch::device(sdf.device()).requires_grad(sdf.requires_grad()));

  // parallelize into small blocks, to minimize unused threads if rays do not
  // intersect with the object
  const dim3 blocks((width + THREADS_PER_DIM - 1) / THREADS_PER_DIM,
                    (height + THREADS_PER_DIM - 1) / THREADS_PER_DIM);
  const dim3 threads(THREADS_PER_DIM, THREADS_PER_DIM);

  AT_DISPATCH_FLOATING_TYPES(
      position.type(), "sdf_renderer_cuda_forward", ([&] {
        sdf_renderer_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            sdf.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            position.data<scalar_t>(), orientation.data<scalar_t>(),
            inv_scale.data<scalar_t>(), width, height, cx, cy, fx, fy, threshold,
            depth_image
                .packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
      }));
  /* cudaDeviceSynchronize(); */
  /* auto end = std::chrono::steady_clock::now(); */
  /* std::cout */
  /*     << "Kernel forward execution took: " */
  /*     << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() */
  /*     << std::endl; */

  return {depth_image};
}

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
                                                      const float fy) {
  /* auto begin = std::chrono::steady_clock::now(); */
  auto grad_sdf = torch::zeros_like(sdf);
  auto grad_position = torch::zeros_like(position);
  auto grad_orientation = torch::zeros_like(orientation);
  auto grad_inv_scale = torch::zeros_like(inv_scale);

  const dim3 blocks((width + THREADS_PER_DIM - 1) / THREADS_PER_DIM,
                    (height + THREADS_PER_DIM - 1) / THREADS_PER_DIM);
  const dim3 threads(THREADS_PER_DIM, THREADS_PER_DIM);

  AT_DISPATCH_FLOATING_TYPES(
      position.type(), "sdf_renderer_cuda_backward", ([&] {
        sdf_renderer_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_depth_image
                .packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            depth_image
                .packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            sdf.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            position.data<scalar_t>(), orientation.data<scalar_t>(),
            inv_scale.data<scalar_t>(),
            grad_sdf.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            grad_position.data<scalar_t>(), grad_orientation.data<scalar_t>(),
            grad_inv_scale.data<scalar_t>(), width, height, cx, cy, fx, fy);
      }));

  /* cudaDeviceSynchronize(); */
  /* auto end = std::chrono::steady_clock::now(); */
  /* std::cout */
  /*     << "Kernel backward execution took: " */
  /*     << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() */
  /*     << std::endl; */
  return {grad_sdf, grad_position, grad_orientation, grad_inv_scale};
}
