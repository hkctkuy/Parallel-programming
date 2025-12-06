#pragma once

#include <cuda_runtime.h>

#include "device.hpp"
#include "solver.hpp"

namespace solver {
namespace gpu {

#define BLOCK 8

/*
 * Solver Implementation for GPU-only Calculation
 */
template<typename T = double>
class GPUSolverImpl: public Solver<T> {
public:
    using Solver = Solver<T>;

    using value_type = typename Solver::value_type;
    using size_type = typename Solver::size_type;
    using rank_type = typename Solver::rank_type;

    using Grid = typename Solver::Grid;
    using Time = typename Solver::Time;

    using InitialGridView = typename Solver::InitialGridView;
    using BoundaryGridView = typename Solver::BoundaryGridView;
    using AnalyticalGridView = typename Solver::AnalyticalGridView;

    using CComm = typename Solver::CComm;

    using DevicePoolManager = typename device::DevicePoolManager<value_type, size_type>;

private:
    // Bring base class members into scope
    using Solver::c_;
    using Solver::ccomm_;
    using Solver::communicatables_;

    // Error Device Pool
    DevicePoolManager edm_;


    virtual void setup(Grid& u) override;

    virtual void calc_init_nodes(Grid& u, Time& t, value_type& e) override;
    virtual void calc_phi_inner_nodes(Grid& u, Time& t, value_type& e) override;
    virtual void calc_phi_boundary_nodes(Grid& u, Time& t, value_type& e) override;
    virtual void calc_inner_nodes(Grid& u, size_type n, Time& t, value_type& e) override;
    virtual void calc_boundary_nodes(Grid& u, size_type n, Time& t, value_type& e) override;

public:
    /*
     * Solver POD View
     */
    struct View {
        size_type Nx, Ny, Nz;
        value_type c;

        InitialGridView phi;
        BoundaryGridView boundary;
        AnalyticalGridView analytical;

        typename CComm::CommDeviceView ccomm;
        struct Communicatables {
            bool left, right, back, front, down, up;
        } communicatables;

        dim3 block;
        dim3 grid;

        __device__ inline auto diff(
            typename Grid::DeviceView& u,
            size_type i, size_type j, size_type k, size_type n
        ) const noexcept;

        __device__ inline bool is_boundary_node(
            size_type i, size_type j, size_type k
        ) const noexcept;

        __device__ inline bool is_inner_node(
            size_type i, size_type j, size_type k
        ) const noexcept;
    };

    // Check AnalyticalFunction triviality
    static_assert(std::is_trivially_copyable_v<View> == true);
    static_assert(std::is_standard_layout_v<View> == true);

    auto view() const noexcept;

    GPUSolverImpl(
        value_type L, size_type N, size_type K,
        size_type Nx, size_type Ny, size_type Nz,
        size_type bx, size_type by, size_type bz,
        CComm ccomm
    ):
        Solver(L, N, K, Nx, Ny, Nz, bx, by, bz, ccomm),
        edm_(Nx + 1, Ny + 1, Nz + 1) {}

};

} // namespace gpu
} // namespace solver

#include "gpu.tpp"
