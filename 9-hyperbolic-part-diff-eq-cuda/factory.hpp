#pragma once

#include <memory>

#include "communicator.hpp"
#include "solver.hpp"

namespace solver {

/*
 * Solver Factory
 */
template<typename T = double>
class SolverFactory {
public:
    using Solver = Solver<T>;

    using value_type = typename Solver::value_type;
    using size_type = typename Solver::size_type;
    using rank_type = typename Solver::rank_type;

    using CComm = typename Solver::CComm;
    using Comm = typename CComm::Comm;
    using Comms = typename CComm::Comms;

private:
    // Common
    value_type L_;
    size_type N_;
    size_type K_;
    // Process depend
    rank_type px_, py_, pz_;
    rank_type pn_;
    rank_type pr_;
    // GPU using flag
    bool gpu_;

    // Get rank begin index and size on grid axis
    auto get_grid_axis_parms(size_type N, rank_type pn, rank_type pr) const noexcept;
    inline auto get_rank(rank_type rx, rank_type ry, rank_type rz) const noexcept;

    auto make_communicator(
        rank_type rx, rank_type ry, rank_type rz,
        size_type Nx, size_type Ny, size_type Nz
    );

public:
    SolverFactory(
        value_type L, size_type N, size_type K,
        rank_type px, rank_type py, rank_type pz,
        rank_type pn, rank_type pr,
        bool gpu = false
    );

    std::shared_ptr<Solver> make_solver();
};

} // namespace solver

#include "factory.tpp"
