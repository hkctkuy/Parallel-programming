#pragma once

#include <omp.h>

#define sqr(x) ((x) * (x))

namespace solver {

template <typename T>
inline auto Solver<T>::diff(
    Grid& u, size_type i, size_type j, size_type k, size_type n
) const noexcept {
    return std::abs(u_analytical_(i, j, k, n) - u(i, j, k, n));
}

template <typename T>
inline bool Solver<T>::is_inner_node(
    size_type i, size_type j, size_type k
) const noexcept {
    return i > 0 && i < Nx_
        && j > 0 && j < Ny_
        && k > 0 && k < Nz_;
}

template <typename T>
inline bool Solver<T>::is_boundary_node(
    size_type i, size_type j, size_type k
) const noexcept {
    return i == 0   && !communicatables_[comm::Edge::Left ]
        || i == Nx_ && !communicatables_[comm::Edge::Right]
        || k == 0   && !communicatables_[comm::Edge::Down ]
        || k == Nz_ && !communicatables_[comm::Edge::Up   ];
}

template <typename T>
auto Solver<T>::Time::sync() noexcept {
    Time t;
    MPI_Reduce(&total,    &t.total,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&inner,    &t.inner,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&boundary, &t.boundary, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&pack,     &t.pack,     1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&wait,     &t.wait,     1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&load,     &t.load,     1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&store,    &t.store,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return t;
}

template <typename T>
Solver<T>::Solver(
    value_type L, size_type N, size_type K,
    size_type Nx, size_type Ny, size_type Nz,
    size_type bx, size_type by, size_type bz,
    CComm ccomm,
    value_type a, value_type g
):
    Nx_(Nx), Ny_(Ny), Nz_(Nz), K_(K),
    h_(L / N), t_(g * h_), a_(a), c_(sqr(a_ * t_ / h_)),
    u_analytical_(AnalyticalFunction(L), h_, t_, bx, by, bz),
    phi_(u_analytical_), boundary_(BoundaryFunction(), h_, t_, bx, by, bz),
    ccomm_(ccomm), communicatables_(ccomm.get_communicatables()) {}

template <typename T>
auto Solver<T>::solve() noexcept {
    Grid u(Nx_, Ny_, Nz_, K_);
    Time time;
    value_type error = 0;
    setup(u);
    auto start = MPI_Wtime();

    // Zero (initial) step
    calc_init_nodes(u, time, error);

    // First (phi) step
    time.pack += TIME(ccomm_.initialize_comm(u, 0));
    calc_phi_inner_nodes(u, time, error);
    time.wait += TIME(ccomm_.finalize_comm());
    calc_phi_boundary_nodes(u, time, error);

    // Tail (ordinary) steps
    for (size_type n = 1; n < K_; n++) {
        time.pack += TIME(ccomm_.initialize_comm(u, n));
        calc_inner_nodes(u, n, time, error);
        time.wait += TIME(ccomm_.finalize_comm());
        calc_boundary_nodes(u, n, time, error);
    }

    // Check time
    auto end = MPI_Wtime();
    time.total = end - start;

    // Aggregate error
    value_type global_error;
    MPI_Reduce(&error, &global_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    return std::tuple(u, global_error, time.sync());
}

} // namespace solver
