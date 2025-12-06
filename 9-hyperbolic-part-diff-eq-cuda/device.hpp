#pragma once

#include <cassert>
#include <cstdio>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#define CUDA_CHECK(call) { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        std::abort(); \
    } \
}

namespace solver {
namespace device {

/*
 * Max value reduction
 */
template <typename T>
T max_(T* ptr, size_t size);

/*
 * Device memory manager for multi dimension data
 * Support from 1 to 3 dims
 * Support data queue with compile time known size
 */
template <typename T, typename size_type = size_t, size_t queue_size = 1>
class DevicePoolManager {
    static_assert(queue_size > 0, "Queue size can't be 0");

public:
    using value_type = T;

private:
    size_type Nx, Ny, Nz;
    size_t size_ = 0;
    size_t head_ = 0; // Ring buffer head
    T* d_ptr_storage_[queue_size] = {nullptr};

    T* alloc() {
        T* d_ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ptr, size_ * sizeof(T)));
        return d_ptr;
    }

    void free() {
        for (size_t i = 0; i < queue_size; i++) {
            auto& d_ptr = d_ptr_storage_[i];
            if (!d_ptr) { continue; }
            cudaFree(d_ptr);
            d_ptr = nullptr;
        }
    }

public:
    DevicePoolManager(size_type Nx = 1, size_type Ny = 1, size_type Nz = 1)
        : Nx(Nx), Ny(Ny), Nz(Nz), size_(Nx * Ny * Nz), head_(0)
    { 
        d_ptr_storage_[head_] = alloc();
    }

    ~DevicePoolManager() { free(); }

    void to_gpu(T* data) {
        auto d_ptr = d_ptr_storage_[head_];
        if (!d_ptr) { return; }
        CUDA_CHECK(cudaMemcpy(d_ptr, data, size_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    void from_gpu(T* data) {
        auto d_ptr = d_ptr_storage_[head_];
        if (!d_ptr) { return; }
        CUDA_CHECK(cudaMemcpy(data, d_ptr, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

    // Roll queue, alloc memory if needed
    void roll() noexcept {
        if constexpr (queue_size == 1) {
            return;
        }
        // Ring buffer head update
        head_ = (head_ + 1) % queue_size;
        if (d_ptr_storage_[head_] == nullptr) {
            d_ptr_storage_[head_] = alloc();
        }
    }

    // Max value reduction
    template <size_t n = 0>
    T max() const noexcept {
        static_assert(n < queue_size, "Queue step is out of range");
        auto i = (head_ + n) % queue_size;
        return max_(d_ptr_storage_[i], size_);
    }

    /*
     * Multi dimension data view for CUDA devices
     * Support dims and queue too
     * NOTE: Trivial class (can be passed to kernel function by value)
     */
    class DeviceView {
        size_type Nx, Ny, Nz;
        T* __restrict__ d_ptr_queue_[queue_size];

        friend class DevicePoolManager;

        DeviceView(
            size_type Nx, size_type Ny, size_type Nz,
            T* const (&d_ptr_queue)[queue_size]
        ):
            Nx(Nx), Ny(Ny), Nz(Nz) {
                for (size_t i = 0; i < queue_size; ++i) {
                    d_ptr_queue_[i] = d_ptr_queue[i];
                }
            }

    public:
        // Stub constructor, can't be used
        DeviceView(): Nx(0), Ny(0), Nz(0) {
            for (size_t i = 0; i < queue_size; ++i)  {
                d_ptr_queue_[i] = nullptr;
            }
        }

        __device__ inline operator bool() const {
            return d_ptr_queue_[0] != nullptr;
        }

        template <size_t n = 0>
        __device__ inline T* ptr() const noexcept {
            static_assert(n < queue_size, "Queue step is out of range");
            return d_ptr_queue_[n];
        }

        __device__ inline const size_type index(
            size_type i, size_type j, size_type k
        ) const noexcept {
#ifdef DEBUG
            assert(i < Nx);
            assert(j < Ny);
            assert(k < Nz);
#endif
            return (i * Ny + j) * Nz + k;
        }

        template <size_t n = 0>
        __device__ inline const T& operator()(
            size_type i = 0, size_type j = 0, size_type k = 0
        ) const noexcept {
            static_assert(n < queue_size, "Queue step is out of range");
            return d_ptr_queue_[n][index(i, j, k)];
        }

        template <size_t n = 0>
        __device__ inline T& operator()(
            size_type i = 0, size_type j = 0, size_type k = 0
        ) noexcept {
            static_assert(n < queue_size, "Queue step is out of range");
            return d_ptr_queue_[n][index(i, j, k)];
        }
    };

    // Check DeviceView triviality
    static_assert(std::is_trivially_copyable_v<DeviceView> == true);
    static_assert(std::is_standard_layout_v<DeviceView> == true);

    DeviceView view() const noexcept {
        T* queue[queue_size];
        for (size_t i = 0; i < queue_size; i++) {
            auto index = (head_ - i) % queue_size;
            queue[i] = d_ptr_storage_[index];
        }
        return DeviceView(Nx, Ny, Nz, queue);
    }
};

} // namespace device
} // namespace solver
