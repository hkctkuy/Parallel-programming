#include <omp.h>

namespace solver {
namespace cpu {

template <typename T>
template <bool initial>
inline auto CPUSolverImpl<T>::func_helper(
    const Grid& u, size_type i, size_type j, size_type k, size_type n
) const noexcept {
    if constexpr (initial) {
        return phi_(i, j, k);
    } else {
        return u(i, j, k, n);
    }
}

template <typename T>
template <bool initial, bool inner>
inline auto CPUSolverImpl<T>::calc_node_helper(
    const Grid& u, size_type i, size_type j, size_type k, size_type n
) const noexcept {
    // Side nodes
    value_type left, right, back, front, down, up;
    if constexpr (inner) {
        left  = func_helper<initial>(u, i - 1, j, k, n);
        right = func_helper<initial>(u, i + 1, j, k, n);
        back  = func_helper<initial>(u, i, j - 1, k, n);
        front = func_helper<initial>(u, i, j + 1, k, n);
        down  = func_helper<initial>(u, i, j, k - 1, n);
        up    = func_helper<initial>(u, i, j, k + 1, n);
    } else {
        // Believe to loop unrolling...
        left  = (i == 0)
              ? ccomm_[comm::Edge::Left ](j, k)
              : func_helper<initial>(u, i - 1, j, k, n);
        right = (i == Nx_)
              ? ccomm_[comm::Edge::Right](j, k)
              : func_helper<initial>(u, i + 1, j, k, n);
        down  = (k == 0)
              ? ccomm_[comm::Edge::Down ](i, j)
              : func_helper<initial>(u, i, j, k - 1, n);
        up    = (k == Nz_)
              ? ccomm_[comm::Edge::Up   ](i, j)
              : func_helper<initial>(u, i, j, k + 1, n);
        back  = (j == 0)
              ? communicatables_[comm::Edge::Back]
              ? ccomm_[comm::Edge::Back](i, k)
              : func_helper<initial>(u, i, Ny_ - 1, k, n)
              : func_helper<initial>(u, i, j   - 1, k, n);
        front = (j == Ny_)
              ? communicatables_[comm::Edge::Front]
              ? ccomm_[comm::Edge::Front](i, k)
              : func_helper<initial>(u, i,       1, k, n)
              : func_helper<initial>(u, i, j   + 1, k, n);
    }
    return -6 * func_helper<initial>(u, i, j, k, n)
           + left + right + back + front + down + up;
}

template <typename T>
template <bool inner>
inline auto CPUSolverImpl<T>::calc_node(
    const Grid& u, size_type i, size_type j, size_type k, size_type n
) const noexcept {
    auto tmp = calc_node_helper<false, inner>(u, i, j, k, n);
    return c_ * tmp + 2 * u(i, j, k, n) - u(i, j, k, n - 1);
}

template <typename T>
template <bool inner>
inline auto CPUSolverImpl<T>::calc_phi_node(
    const Grid& u, size_type i, size_type j, size_type k
) const noexcept {
    auto tmp = calc_node_helper<true, inner>(u, i, j, k, 0);
    return u(i, j, k, 0) + c_ / 2 * tmp;
}

// Kernel impls
template <typename T>
typename CPUSolverImpl<T>::value_type
CPUSolverImpl<T>::calc_init_nodes_impl(Grid& u) const noexcept {
    size_type n = 0;
    value_type error = 0;
    #pragma omp parallel
    {
        value_type t_error = 0;
        #pragma omp for
        for (size_type i = 0; i <= Nx_; i++) {
            for (size_type j = 0; j <= Ny_; j++) {
                for (size_type k = 0; k <= Nz_; k++) {
                    u(i, j, k, n) = is_boundary_node(i, j, k)
                                      ? boundary_(i, j, k, n)
                                      : phi_(i, j, k);
                    t_error = std::max(t_error, diff(u, i, j, k, n));
                }
            }
        }
        #pragma omp critical
        if (error < t_error) {
            error = t_error;
        }
    } // end pragma omp parallel
    return error;
}

template <typename T>
typename CPUSolverImpl<T>::value_type
CPUSolverImpl<T>::calc_phi_inner_nodes_impl(Grid& u) const noexcept {
    size_type n = 1;
    value_type error = 0;
    #pragma omp parallel
    {
        value_type t_error = 0;
        #pragma omp for
        for (size_type i = 1; i < Nx_; i++) {
            for (size_type j = 1; j < Ny_; j++) {
                for (size_type k = 1; k < Nz_; k++) {
                    u(i, j, k, n) = calc_phi_node<true>(u, i, j, k);
                    t_error = std::max(t_error, diff(u, i, j, k, n));
                }
            }
        }
        #pragma omp critical
        if (error < t_error) {
            error = t_error;
        }
    } // end pragma omp parallel
    return error;
}

template <typename T>
typename CPUSolverImpl<T>::value_type
CPUSolverImpl<T>::calc_phi_boundary_nodes_impl(Grid& u) const noexcept {
    size_type n = 1;
    value_type error = 0;
    #pragma omp parallel
    {
        value_type t_error = 0;
        #pragma omp for
        for (size_type i = 0; i <= Nx_; i++) {
            for (size_type j = 0; j <= Ny_; j++) {
                for (size_type k = 0; k <= Nz_; k++) {
                    if (is_inner_node(i, j, k)) {
                        continue;
                    }
                    u(i, j, k, n) = is_boundary_node(i, j, k)
                                  ? boundary_(i, j, k, n)
                                  : calc_phi_node<false>(u, i, j, k);
                    t_error = std::max(t_error, diff(u, i, j, k, n));
                }
            }
        }
        #pragma omp critical
        if (error < t_error) {
            error = t_error;
        }
    } // end pragma omp parallel
    return error;
}

template <typename T>
typename CPUSolverImpl<T>::value_type
CPUSolverImpl<T>::calc_inner_nodes_impl(Grid& u, size_type n) const noexcept {
    value_type error = 0;
    #pragma omp parallel
    {
        value_type t_error = 0;
        #pragma omp for
        for (size_type i = 1; i < Nx_; i++) {
            for (size_type j = 1; j < Ny_; j++) {
                for (size_type k = 1; k < Nz_; k++) {
                    u(i, j, k, n + 1) = calc_node<true>(u, i, j, k, n);
                    t_error = std::max(t_error, diff(u, i, j, k, n + 1));
                }
            }
        }
        #pragma omp critical
        if (error < t_error) {
            error = t_error;
        }
    } // end pragma omp parallel
    return error;
}

template <typename T>
typename CPUSolverImpl<T>::value_type
CPUSolverImpl<T>::calc_boundary_nodes_impl(Grid& u, size_type n) const noexcept {
    value_type error = 0;
    #pragma omp parallel
    {
        value_type t_error = 0;
        #pragma omp for
        for (size_type i = 0; i <= Nx_; i++) {
            for (size_type j = 0; j <= Ny_; j++) {
                for (size_type k = 0; k <= Nz_; k++) {
                    if (is_inner_node(i, j, k)) {
                        continue;
                    }
                    u(i, j, k, n + 1) = is_boundary_node(i, j, k)
                                      ? boundary_(i, j, k, n)
                                      : calc_node<false>(u, i, j, k, n);
                    t_error = std::max(t_error, diff(u, i, j, k, n + 1));
                }
            }
        }
        #pragma omp critical
        if (error < t_error) {
            error = t_error;
        }
    } // end pragma omp parallel
    return error;
}

// Stats collectors
template <typename T>
void CPUSolverImpl<T>::calc_init_nodes(Grid& u, Time& t, value_type& e) {
    UPDATE_STATS(calc_init_nodes_impl(u), t.inner, e);
}

template <typename T>
void CPUSolverImpl<T>::calc_phi_inner_nodes(Grid& u, Time& t, value_type& e) {
    UPDATE_STATS(calc_phi_inner_nodes_impl(u), t.inner, e);
}

template <typename T>
void CPUSolverImpl<T>::calc_phi_boundary_nodes(Grid& u, Time& t, value_type& e) {
    UPDATE_STATS(calc_phi_boundary_nodes_impl(u), t.boundary, e);
}

template <typename T>
void CPUSolverImpl<T>::calc_inner_nodes(Grid& u, size_type n, Time& t, value_type& e) {
    UPDATE_STATS(calc_inner_nodes_impl(u, n), t.inner, e);
}

template <typename T>
void CPUSolverImpl<T>::calc_boundary_nodes(Grid& u, size_type n, Time& t, value_type& e) {
    UPDATE_STATS(calc_boundary_nodes_impl(u, n), t.boundary, e);
}

} // namespace cpu
} // namespace solver

