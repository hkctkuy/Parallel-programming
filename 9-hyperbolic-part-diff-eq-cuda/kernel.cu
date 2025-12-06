#include <cuda_runtime.h>

#include "gpu.hpp"
#include "kernel.cuh"
#include "solver.hpp"

namespace solver {
namespace gpu {
namespace kernel {

// Impls
template <typename T>
__global__ void calc_init_nodes_impl(
    typename GPUSolverImpl<T>::Grid::DeviceView u,
    typename GPUSolverImpl<T>::View v,
    typename GPUSolverImpl<T>::DevicePoolManager::DeviceView e
) {
    using size_type = GPUSolverImpl<T>::size_type;
    size_type i = threadIdx.z + blockIdx.z * blockDim.z;
    size_type j = threadIdx.y + blockIdx.y * blockDim.y;
    size_type k = threadIdx.x + blockIdx.x * blockDim.x;

    if (i > v.Nx || j > v.Ny || k > v.Nz) { return; }

    ACCESS(u, i, j, k, NEXT) = v.is_boundary_node(i, j, k)
                             ? v.boundary(i, j, k, 0)
                             : v.phi(i, j, k);

    e(i, j, k) = v.diff(u, i, j, k, 0);
}

template <typename T>
__global__ void calc_phi_inner_nodes_impl(
    typename GPUSolverImpl<T>::Grid::DeviceView u,
    typename GPUSolverImpl<T>::View v,
    typename GPUSolverImpl<T>::DevicePoolManager::DeviceView e
) {
    using size_type = GPUSolverImpl<T>::size_type;
    size_type i = threadIdx.z + blockIdx.z * blockDim.z;
    size_type j = threadIdx.y + blockIdx.y * blockDim.y;
    size_type k = threadIdx.x + blockIdx.x * blockDim.x;

    if (i > v.Nx || j > v.Ny || k > v.Nz) { return; }
    if (!v.is_inner_node(i, j, k)) { return; }

    auto left  = v.phi(i - 1, j, k);
    auto right = v.phi(i + 1, j, k);
    auto back  = v.phi(i, j - 1, k);
    auto front = v.phi(i, j + 1, k);
    auto down  = v.phi(i, j, k - 1);
    auto up    = v.phi(i, j, k + 1);
    auto tmp = -6 * v.phi(i, j, k)
             + left + right + back + front + down + up;
    ACCESS(u, i, j, k, NEXT) = ACCESS(u, i, j, k, CURR) + v.c / 2 * tmp;

    e(i, j, k) = v.diff(u, i, j, k, 1);
}

template <typename T>
__global__ void calc_phi_boundary_nodes_impl(
    typename GPUSolverImpl<T>::Grid::DeviceView u,
    typename GPUSolverImpl<T>::View v,
    typename GPUSolverImpl<T>::DevicePoolManager::DeviceView e
) {
    using size_type = GPUSolverImpl<T>::size_type;
    size_type i = threadIdx.z + blockIdx.z * blockDim.z;
    size_type j = threadIdx.y + blockIdx.y * blockDim.y;
    size_type k = threadIdx.x + blockIdx.x * blockDim.x;

    if (i > v.Nx || j > v.Ny || k > v.Nz) { return; }
    if (v.is_inner_node(i, j, k)) { return; }

    if (v.is_boundary_node(i, j, k)) {
        ACCESS(u, i, j, k, NEXT) = v.boundary(i, j, k, 1);
    } else {
        auto left  = (i == 0)    ? v.ccomm.left( j, k) : v.phi(i - 1, j, k);
        auto right = (i == v.Nx) ? v.ccomm.right(j, k) : v.phi(i + 1, j, k);
        auto down  = (k == 0)    ? v.ccomm.down( i, j) : v.phi(i, j, k - 1);
        auto up    = (k == v.Nz) ? v.ccomm.up(   i, j) : v.phi(i, j, k + 1);
        auto back  = (j == 0)
                   ? v.ccomm.back
                   ? v.ccomm.back( i, k)
                   : v.phi(i, v.Ny - 1, k)
                   : v.phi(i, j    - 1, k);
        auto front = (j == v.Ny)
                   ? v.ccomm.front
                   ? v.ccomm.front(i, k)
                   : v.phi(i,       1, k)
                   : v.phi(i, j   + 1, k);
        auto tmp = -6 * v.phi(i, j, k)
                 + left + right + back + front + down + up;

        ACCESS(u, i, j, k, NEXT) = ACCESS(u, i, j, k, CURR) + v.c / 2 * tmp;
    }

    e(i, j, k) = v.diff(u, i, j, k, 1);
}

template <typename T>
__global__ void calc_inner_nodes_impl(
    typename GPUSolverImpl<T>::Grid::DeviceView u,
    typename GPUSolverImpl<T>::View v,
    typename GPUSolverImpl<T>::DevicePoolManager::DeviceView e,
    typename GPUSolverImpl<T>::size_type n
) {
    using size_type = GPUSolverImpl<T>::size_type;
    size_type i = threadIdx.z + blockIdx.z * blockDim.z;
    size_type j = threadIdx.y + blockIdx.y * blockDim.y;
    size_type k = threadIdx.x + blockIdx.x * blockDim.x;

    if (i > v.Nx || j > v.Ny || k > v.Nz) { return; }
    if (!v.is_inner_node(i, j, k)) { return; }

    auto left  = ACCESS(u, i - 1, j, k, CURR);
    auto right = ACCESS(u, i + 1, j, k, CURR);
    auto back  = ACCESS(u, i, j - 1, k, CURR);
    auto front = ACCESS(u, i, j + 1, k, CURR);
    auto down  = ACCESS(u, i, j, k - 1, CURR);
    auto up    = ACCESS(u, i, j, k + 1, CURR);
    auto tmp = -6 * ACCESS(u, i, j, k, CURR)
             + left + right + back + front + down + up;

    ACCESS(u, i, j, k, NEXT) = v.c * tmp
                             + 2 * ACCESS(u, i, j, k, CURR)
                             - ACCESS(u, i, j, k, PREV);

    e(i, j, k) = v.diff(u, i, j, k, n + 1);
}

template <typename T>
__global__ void calc_boundary_nodes_impl(
    typename GPUSolverImpl<T>::Grid::DeviceView u,
    typename GPUSolverImpl<T>::View v,
    typename GPUSolverImpl<T>::DevicePoolManager::DeviceView e,
    typename GPUSolverImpl<T>::size_type n
) {
    using size_type = GPUSolverImpl<T>::size_type;
    size_type i = threadIdx.z + blockIdx.z * blockDim.z;
    size_type j = threadIdx.y + blockIdx.y * blockDim.y;
    size_type k = threadIdx.x + blockIdx.x * blockDim.x;

    if (i > v.Nx || j > v.Ny || k > v.Nz) { return; }
    if (v.is_inner_node(i, j, k)) { return; }

    if (v.is_boundary_node(i, j, k)) {
        ACCESS(u, i, j, k, NEXT) = v.boundary(i, j, k, n);
    } else {
        auto left  = (i == 0)    ? v.ccomm.left( j, k) : ACCESS(u, i - 1, j, k, CURR);
        auto right = (i == v.Nx) ? v.ccomm.right(j, k) : ACCESS(u, i + 1, j, k, CURR);
        auto down  = (k == 0)    ? v.ccomm.down( i, j) : ACCESS(u, i, j, k - 1, CURR);
        auto up    = (k == v.Nz) ? v.ccomm.up(   i, j) : ACCESS(u, i, j, k + 1, CURR);
        auto back  = (j == 0)
                   ? v.ccomm.back
                   ? v.ccomm.back( i, k)
                   : ACCESS(u, i, v.Ny - 1, k, CURR)
                   : ACCESS(u, i, j    - 1, k, CURR);
        auto front = (j == v.Ny)
                   ? v.ccomm.front
                   ? v.ccomm.front(i, k)
                   : ACCESS(u, i,        1, k, CURR)
                   : ACCESS(u, i, j    + 1, k, CURR);
        auto tmp = -6 * ACCESS(u, i, j, k, CURR)
                 + left + right + back + front + down + up;

        ACCESS(u, i, j, k, NEXT) = v.c * tmp
                                 + 2 * ACCESS(u, i, j, k, CURR)
                                 - ACCESS(u, i, j, k, PREV);
    }

    e(i, j, k) = v.diff(u, i, j, k, n + 1);
}

// Launchers
template <typename T>
void calc_init_nodes(
    typename GPUSolverImpl<T>::Grid::DeviceView u,
    typename GPUSolverImpl<T>::View v,
    typename GPUSolverImpl<T>::DevicePoolManager::DeviceView e
) {
    CUDA_CHECK((calc_init_nodes_impl<T><<<v.grid, v.block>>>(u, v, e)));
    CUDA_CHECK((cudaDeviceSynchronize()));
}

template <typename T>
void calc_phi_inner_nodes(
    typename GPUSolverImpl<T>::Grid::DeviceView u,
    typename GPUSolverImpl<T>::View v,
    typename GPUSolverImpl<T>::DevicePoolManager::DeviceView e
) {
    CUDA_CHECK((cudaDeviceSynchronize()));
    CUDA_CHECK((calc_phi_inner_nodes_impl<T><<<v.grid, v.block>>>(u, v, e)));
    CUDA_CHECK((cudaDeviceSynchronize()));
}

template <typename T>
void calc_phi_boundary_nodes(
    typename GPUSolverImpl<T>::Grid::DeviceView u,
    typename GPUSolverImpl<T>::View v,
    typename GPUSolverImpl<T>::DevicePoolManager::DeviceView e
) {
    CUDA_CHECK((cudaDeviceSynchronize()));
    CUDA_CHECK((calc_phi_boundary_nodes_impl<T><<<v.grid, v.block>>>(u, v, e)));
    CUDA_CHECK((cudaDeviceSynchronize()));
}

template <typename T>
void calc_inner_nodes(
    typename GPUSolverImpl<T>::Grid::DeviceView u,
    typename GPUSolverImpl<T>::View v,
    typename GPUSolverImpl<T>::DevicePoolManager::DeviceView e,
    typename GPUSolverImpl<T>::size_type n
) {
    CUDA_CHECK((cudaDeviceSynchronize()));
    CUDA_CHECK((calc_inner_nodes_impl<T><<<v.grid, v.block>>>(u, v, e, n)));
    CUDA_CHECK((cudaDeviceSynchronize()));
}

template <typename T>
void calc_boundary_nodes(
    typename GPUSolverImpl<T>::Grid::DeviceView u,
    typename GPUSolverImpl<T>::View v,
    typename GPUSolverImpl<T>::DevicePoolManager::DeviceView e,
    typename GPUSolverImpl<T>::size_type n
) {
    CUDA_CHECK((cudaDeviceSynchronize()));
    CUDA_CHECK((calc_boundary_nodes_impl<T><<<v.grid, v.block>>>(u, v, e, n)));
    CUDA_CHECK((cudaDeviceSynchronize()));
}

INSTANTIATE_ALL_KERNELS(double)

} // namespace kernel
} // namespace gpu
} // namespace solver
