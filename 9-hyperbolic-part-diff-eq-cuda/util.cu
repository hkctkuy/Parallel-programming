#include <limits>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

namespace solver {
namespace device {

/*
 * Max value reduction
 */
template <typename T>
T max_(T* ptr, size_t size) {
    thrust::device_ptr<T> begin = thrust::device_pointer_cast(ptr);
    thrust::device_ptr<T> end = begin + size;
    T init = -std::numeric_limits<T>::infinity();
    return thrust::reduce(thrust::cuda::par, begin, end, init, thrust::maximum<T>());
}

// Explicit instantiation for dummy nvcc
template double max_(double*, size_t);

} // namespace device
} // namespace solver
