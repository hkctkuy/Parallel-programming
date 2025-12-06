#pragma once

#include <cuda_runtime.h>

#include "gpu.hpp"
#include "solver.hpp"

namespace solver {
namespace gpu {
namespace kernel {

// Check CUDA errors
#ifdef CUDA_CHECK
#undef CUDA_CHECK
#endif

#define CUDA_CHECK(call) { \
    (call); \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
	    fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        std::abort(); \
    } \
}

// Explicit kernel instant macro
#define INSTANTIATE_KERNEL(KERNEL_NAME, T) \
    template void KERNEL_NAME<T>( \
        GPUSolverImpl<T>::Grid::DeviceView, \
        GPUSolverImpl<T>::View, \
        GPUSolverImpl<T>::DevicePoolManager::DeviceView \
    );

#define INSTANTIATE_KERNEL_N(KERNEL_NAME, T) \
    template void KERNEL_NAME<T>( \
        GPUSolverImpl<T>::Grid::DeviceView, \
        GPUSolverImpl<T>::View, \
        GPUSolverImpl<T>::DevicePoolManager::DeviceView, \
        GPUSolverImpl<T>::size_type \
    );

// Instant all kernels macro
#define INSTANTIATE_ALL_KERNELS(T) \
    INSTANTIATE_KERNEL(calc_init_nodes, T) \
    INSTANTIATE_KERNEL(calc_phi_inner_nodes, T) \
    INSTANTIATE_KERNEL(calc_phi_boundary_nodes, T) \
    INSTANTIATE_KERNEL_N(calc_inner_nodes, T) \
    INSTANTIATE_KERNEL_N(calc_boundary_nodes, T)

#define ACCESS(v, i, j, k, n) (v).operator()<(n)>((i), (j), (k))
#define PREV 2
#define CURR 1
#define NEXT 0

template <typename T>
void calc_init_nodes(
    typename GPUSolverImpl<T>::Grid::DeviceView u,
    typename GPUSolverImpl<T>::View v,
    typename GPUSolverImpl<T>::DevicePoolManager::DeviceView e
);

template <typename T>
void calc_phi_inner_nodes(
    typename GPUSolverImpl<T>::Grid::DeviceView u,
    typename GPUSolverImpl<T>::View v,
    typename GPUSolverImpl<T>::DevicePoolManager::DeviceView e
);

template <typename T>
void calc_phi_boundary_nodes(
    typename GPUSolverImpl<T>::Grid::DeviceView u,
    typename GPUSolverImpl<T>::View v,
    typename GPUSolverImpl<T>::DevicePoolManager::DeviceView e
);

template <typename T>
void calc_inner_nodes(
    typename GPUSolverImpl<T>::Grid::DeviceView u,
    typename GPUSolverImpl<T>::View v,
    typename GPUSolverImpl<T>::DevicePoolManager::DeviceView e,
    typename GPUSolverImpl<T>::size_type n
);

template <typename T>
void calc_boundary_nodes(
    typename GPUSolverImpl<T>::Grid::DeviceView u,
    typename GPUSolverImpl<T>::View v,
    typename GPUSolverImpl<T>::DevicePoolManager::DeviceView e,
    typename GPUSolverImpl<T>::size_type n
);

} // namespace kernel
} // namespace gpu
} // namespace solver
