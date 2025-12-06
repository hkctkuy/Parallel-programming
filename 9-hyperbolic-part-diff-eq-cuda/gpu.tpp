#include <omp.h>

#include "kernel.cuh"

namespace solver {
namespace gpu {

template <typename T>
void GPUSolverImpl<T>::setup(Grid& u) {
    u.enable_gpu();
    ccomm_.enable_gpu();
}

// Kernel impls
template <typename T>
void GPUSolverImpl<T>::calc_init_nodes(Grid& u, Time& t, value_type& e) {
    size_type n = 0;
    t.inner += TIME(kernel::calc_init_nodes<value_type>(u.view(), view(), edm_.view()));
    t.store += TIME(u.from_gpu(n));
    e = std::max(e, edm_.max());
}

template <typename T>
void GPUSolverImpl<T>::calc_phi_inner_nodes(Grid& u, Time& t, value_type& e) {
    size_type n = 1;
    t.inner += TIME(kernel::calc_phi_inner_nodes<value_type>(u.view(), view(), edm_.view()));
    // NOTE: Do only one reduction in the end of the step
}

template <typename T>
void GPUSolverImpl<T>::calc_phi_boundary_nodes(Grid& u, Time& t, value_type& e) {
    size_type n = 1;
    t.load += TIME(ccomm_.to_gpu());
    t.boundary += TIME(kernel::calc_phi_boundary_nodes<value_type>(u.view(), view(), edm_.view()));
    t.store += TIME(u.from_gpu(n));
    e = std::max(e, edm_.max());
}

template <typename T>
void GPUSolverImpl<T>::calc_inner_nodes(Grid& u, size_type n, Time& t, value_type& e) {
    t.inner += TIME(kernel::calc_inner_nodes<value_type>(u.view(), view(), edm_.view(), n));
    // NOTE: Do only one reduction in the end of the step
}

template <typename T>
void GPUSolverImpl<T>::calc_boundary_nodes(Grid& u, size_type n, Time& t, value_type& e) {
    t.load += TIME(ccomm_.to_gpu());
    t.boundary += TIME(kernel::calc_boundary_nodes<value_type>(u.view(), view(), edm_.view(), n));
    t.store += TIME(u.from_gpu(n + 1));
    e = std::max(e, edm_.max());
}

template <typename T>
__device__ inline auto GPUSolverImpl<T>::View::diff(
    typename Grid::DeviceView& u,
    size_type i, size_type j, size_type k, size_type n
) const noexcept {
    return std::abs(u.operator()<0>(i, j, k) - analytical(i, j, k, n));
}

template <typename T>
__device__ inline bool
GPUSolverImpl<T>::View::is_boundary_node(
    size_type i, size_type j, size_type k
) const noexcept {
    return i == 0  && !communicatables.left
        || i == Nx && !communicatables.right
        || k == 0  && !communicatables.down
        || k == Nz && !communicatables.up;
}

template <typename T>
__device__ inline bool GPUSolverImpl<T>::View::is_inner_node(
    size_type i, size_type j, size_type k
) const noexcept {
    return i > 0 && i < Nx
        && j > 0 && j < Ny
        && k > 0 && k < Nz;
}

template <typename T>
auto GPUSolverImpl<T>::view() const noexcept {
    typename View::Communicatables communicatables {
        communicatables_[comm::Edge::Left ],
        communicatables_[comm::Edge::Right],
        communicatables_[comm::Edge::Back ],
        communicatables_[comm::Edge::Front],
        communicatables_[comm::Edge::Down ],
        communicatables_[comm::Edge::Up   ]
    };
    dim3 block(BLOCK, BLOCK, BLOCK);
    // NOTE: Use column-major indexation
    dim3 grid(
        (this->Nz_ + block.x) / block.x,
        (this->Ny_ + block.y) / block.y,
        (this->Nx_ + block.z) / block.z
    );
    return View {
        this->Nx_, this->Ny_, this->Nz_,
        this->c_,
        this->phi_, this->boundary_, this->u_analytical_,
        this->ccomm_.view(), communicatables,
        block, grid
    };
}

} // namespace gpu
} // namespace solver

