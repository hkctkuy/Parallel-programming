#pragma once

#include "solver.hpp"

namespace solver {
namespace cpu {

/*
 * Solver Implementation for CPU-only Calculation
 */
template<typename T = double>
class CPUSolverImpl: public Solver<T> {
public:
    using Solver = Solver<T>;
    using value_type = typename Solver::value_type;
    using size_type = typename Solver::size_type;
    using rank_type = typename Solver::rank_type;
    using Grid = typename Solver::Grid;
    using Time = typename Solver::Time;

private:
    // Bring base class members into scope
    using Solver::c_;
    using Solver::Nx_;
    using Solver::Ny_;
    using Solver::Nz_;
    using Solver::boundary_;
    using Solver::phi_;
    using Solver::ccomm_;
    using Solver::communicatables_;
    // Methods
    using Solver::diff;
    using Solver::is_inner_node;
    using Solver::is_boundary_node;

    // Choosing function helper
    template <bool initial>
    inline auto func_helper(
        const Grid& u, size_type i, size_type j, size_type k, size_type n
    ) const noexcept;

    // Node calculation helper
    template <bool initial, bool inner>
    inline auto calc_node_helper(
        const Grid& u, size_type i, size_type j, size_type k, size_type n
    ) const noexcept;

    // Calculate ordinary node
    template <bool inner>
    inline auto calc_node(
        const Grid& u, size_type i, size_type j, size_type k, size_type n
    ) const noexcept;

    // Calculate phi node
    template <bool inner>
    inline auto calc_phi_node(
        const Grid& u, size_type i, size_type j, size_type k
    ) const noexcept;

    virtual value_type calc_init_nodes_impl(Grid& u) const noexcept;
    virtual value_type calc_phi_inner_nodes_impl(Grid& u) const noexcept;
    virtual value_type calc_phi_boundary_nodes_impl(Grid& u) const noexcept;
    virtual value_type calc_inner_nodes_impl(Grid& u, size_type n) const noexcept;
    virtual value_type calc_boundary_nodes_impl(Grid& u, size_type n) const noexcept;

    virtual void calc_init_nodes(Grid& u, Time& t, value_type& e) override;
    virtual void calc_phi_inner_nodes(Grid& u, Time& t, value_type& e) override;
    virtual void calc_phi_boundary_nodes(Grid& u, Time& t, value_type& e) override;
    virtual void calc_inner_nodes(Grid& u, size_type n, Time& t, value_type& e) override;
    virtual void calc_boundary_nodes(Grid& u, size_type n, Time& t, value_type& e) override;

public:
    CPUSolverImpl(
        value_type L, size_type N, size_type K,
        size_type Nx, size_type Ny, size_type Nz,
        size_type bx, size_type by, size_type bz,
        comm::CubeCommunicator<value_type> ccomm
    ):
        Solver(L, N, K, Nx, Ny, Nz, bx, by, bz, ccomm) {}
};

} // namespace cpu
} // namespace solver

#include "cpu.tpp"
