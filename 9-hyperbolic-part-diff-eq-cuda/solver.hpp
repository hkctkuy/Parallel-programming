#pragma once

#include <type_traits>
#include <utility>

#include <omp.h>

#include "communicator.hpp"
#include "grid.hpp"
#include "func.hpp"

#define TIME(call)               \
([&]() {                         \
    double _start = MPI_Wtime(); \
    (call);                      \
    return MPI_Wtime() - _start; \
}())

#define UPDATE_STATS(call, time, error) \
do {                                    \
    double _start = MPI_Wtime();        \
    auto _err = (call);                 \
    (error) = std::max((error), _err);  \
    (time) += MPI_Wtime() - _start;     \
} while (0)

#define STEPS 4

namespace solver {

/*
 * Main class: hyperbolic partition differential equation solver
 */
template<typename T>
class Solver {
public:
    using CComm = typename comm::CubeCommunicator<T>;
    using Grid = typename grid::Grid<T, STEPS>;
    using AnalyticalFunction = typename func::AnalyticalFunction<T>;
    using BoundaryFunction = typename func::BoundaryFunction<T>;

    using value_type = typename Grid::value_type;
    using size_type = typename Grid::size_type;
    using rank_type = typename CComm::rank_type;

    template<class F>
    using GridView = typename func::FunctionGridView<F, size_type>;
    using AnalyticalGridView = GridView<AnalyticalFunction>;
    using BoundaryGridView = GridView<BoundaryFunction>;

    struct Time {
        using time_type = double;   // MPI_Wtime return type

        time_type total = 0;        // Total time
        time_type inner = 0;        // Calculate inner nodes time
        time_type boundary = 0;     // Calculate boundary nodes time
        time_type pack = 0;         // Pack data for MPI send time
        time_type wait = 0;         // Wait/unpack data for MPI recv time
        time_type load = 0;         // Load data to device time
        time_type store = 0;        // Store data from device time

        auto sync() noexcept;
    };

protected:
    // Params
    size_type Nx_, Ny_, Nz_;
    size_type K_;
    value_type h_;
    value_type t_;
    value_type a_;
    value_type c_;

    // Functions
    // Grid analytical origin function
    AnalyticalGridView u_analytical_;
    // Grid boundary condition function
    BoundaryGridView boundary_;
    // Grid initial function
    class InitialGridView: AnalyticalGridView {
    public:
        InitialGridView(const AnalyticalGridView& func)
            : AnalyticalGridView(func) {}

        __host__ __device__
        auto operator()(size_type i, size_type j, size_type k) const noexcept {
            return this->AnalyticalGridView::operator()(i, j, k, 0);
        }
    } phi_;

    // Check AnalyticalFunction triviality
    static_assert(std::is_trivially_copyable_v<InitialGridView> == true);
    static_assert(std::is_standard_layout_v<InitialGridView> == true);

    // Comm
    CComm ccomm_;
    const comm::Communicatables communicatables_; // Edge checking opt

    // Utils
    inline auto diff(Grid& u, size_type i, size_type j, size_type k, size_type n) const noexcept;
    inline bool is_boundary_node(size_type i, size_type j, size_type k) const noexcept;
    inline bool is_inner_node(size_type i, size_type j, size_type k) const noexcept;

    // For specific pre actions
    virtual void setup(Grid& u) { return; }
    // Kernels
    // Calculate u for initial/zero step for all nodes and return error
    virtual void calc_init_nodes(Grid& u, Time& t, value_type& e) = 0;
    // Calculate u for phi/first step for inner nodes and return error
    virtual void calc_phi_inner_nodes(Grid& u, Time& t, value_type& e) = 0;
    // Calculate u for phi/first step for boundary nodes and return error
    virtual void calc_phi_boundary_nodes(Grid& u, Time& t, value_type& e) = 0;
    // Calculate u for ordinary/tail step for inner nodes and return error
    virtual void calc_inner_nodes(Grid& u, size_type n, Time& t, value_type& e) = 0;
    // Calculate u for ordinary/tail step for boundary nodes and return error
    virtual void calc_boundary_nodes(Grid& u, size_type n, Time& t, value_type& e) = 0;

public:
    Solver(
        value_type L, size_type N, size_type K,
        size_type Nx, size_type Ny, size_type Nz,
        size_type bx, size_type by, size_type bz,
        CComm ccomm,
        value_type a = value_type(0.5),
        value_type g = value_type(0.5)
    );

    auto solve() noexcept;

    virtual ~Solver() = default;
};

} // namespace solver

#include "solver.tpp"
