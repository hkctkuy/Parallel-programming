#pragma once

#include <memory>
#include <tuple>

#include "communicator.hpp"
#include "cpu.hpp"
#include "gpu.hpp"
#include "solver.hpp"

namespace solver {

template <typename T>
auto SolverFactory<T>::get_grid_axis_parms(
    size_type N, rank_type pn, rank_type pr
) const noexcept {
    size_type d = (N + 1) / pn;
    size_type m = (N + 1) % pn;
    size_type b = pr * d + std::min<size_type>(pr, m);
    size_type n = d + (pr < m ? 1 : 0) - 1;
    return std::tuple(b, n);
}

template <typename T>
inline auto SolverFactory<T>::get_rank(
    rank_type rx, rank_type ry, rank_type rz
) const noexcept {
    return (rx * py_ + ry) * pz_ + rz;
}

template <typename T>
auto SolverFactory<T>::make_communicator(
    rank_type rx, rank_type ry, rank_type rz,
    size_type Nx, size_type Ny, size_type Nz
) {
    Comms comms = {nullptr};
    if (rx != 0) {
        auto r = get_rank(rx - 1, ry, rz);
        Comm comm(r, Ny, Nz);
        for (size_type j = 0; j <= Ny; j++) {
            for (size_type k = 0; k <= Nz; k++) {
                comm.index(j, k) = j * (Nz + 1) + k;
            }
        }
        comms[comm::Edge::Left] = std::make_shared<Comm>(comm);
    }
    if (rx != px_ - 1) {
        auto r = get_rank(rx + 1, ry, rz);
        Comm comm(r, Ny, Nz);
        auto offset = Nx * (Ny + 1) * (Nz + 1);
        for (size_type j = 0; j <= Ny; j++) {
            for (size_type k = 0; k <= Nz; k++) {
                comm.index(j, k) = offset + j * (Nz + 1) + k;
            }
        }
        comms[comm::Edge::Right] = std::make_shared<Comm>(comm);
    }
    if (py_ != 1) {
        if (ry != 0) {
            auto r = get_rank(rx, ry - 1, rz);
            Comm comm(r, Nx, Nz);
            for (size_type i = 0; i <= Nx; i++) {
                for (size_type k = 0; k <= Nz; k++) {
                    comm.index(i, k) = i * (Ny + 1) * (Nz + 1) + k;
                }
            }
            comms[comm::Edge::Back] = std::make_shared<Comm>(comm);
        } else {
            // Periodic
            auto r = get_rank(rx, py_ - 1, rz);
            Comm comm(r, Nx, Nz, comm::PERIODIC);
            auto offset = Nz + 1;
            for (size_type i = 0; i <= Nx; i++) {
                for (size_type k = 0; k <= Nz; k++) {
                    comm.index(i, k) = offset + i * (Ny + 1) * (Nz + 1) + k;
                }
            }
            comms[comm::Edge::Back] = std::make_shared<Comm>(comm);
        }
        if (ry != py_ - 1) {
            auto r = get_rank(rx, ry + 1, rz);
            Comm comm(r, Nx, Nz);
            auto offset = Ny * (Nz + 1);
            for (size_type i = 0; i <= Nx; i++) {
                for (size_type k = 0; k <= Nz; k++) {
                    comm.index(i, k) = offset + i * (Ny + 1) * (Nz + 1) + k;
                }
            }
            comms[comm::Edge::Front] = std::make_shared<Comm>(comm);
        } else {
            // Periodic
            auto r = get_rank(rx, 0, rz);
            Comm comm(r, Nx, Nz, comm::PERIODIC);
            auto offset = (Ny - 1) * (Nz + 1);
            for (size_type i = 0; i <= Nx; i++) {
                for (size_type k = 0; k <= Nz; k++) {
                    comm.index(i, k) = offset + i * (Ny + 1) * (Nz + 1) + k;
                }
            }
            comms[comm::Edge::Front] = std::make_shared<Comm>(comm);
        }
    }
    if (rz != 0) {
        auto r = get_rank(rx, ry, rz - 1);
        Comm comm(r, Nx, Ny);
        for (size_type i = 0; i <= Nx; i++) {
            for (size_type j = 0; j <= Ny; j++) {
                comm.index(i, j) = (i * (Ny + 1) + j) * (Nz + 1);
            }
        }
        comms[comm::Edge::Down] = std::make_shared<Comm>(comm);
    }
    if (rz != pz_ - 1) {
        auto r = get_rank(rx, ry, rz + 1);
        Comm comm(r, Nx, Ny);
        auto offset = Nz;
        for (size_type i = 0; i <= Nx; i++) {
            for (size_type j = 0; j <= Ny; j++) {
                comm.index(i, j) = offset + (i * (Ny + 1) + j) * (Nz + 1);
            }
        }
        comms[comm::Edge::Up] = std::make_shared<Comm>(comm);
    }
    return comm::CubeCommunicator<value_type>(comms);
}

template <typename T>
SolverFactory<T>::SolverFactory(
    value_type L, size_type N, size_type K,
    rank_type px, rank_type py, rank_type pz,
    rank_type pn, rank_type pr,
    bool gpu
):
    L_(L), N_(N), K_(K),
    px_(px), py_(py), pz_(pz), pn_(pn), pr_(pr),
    gpu_(gpu)
{
#ifdef DEBUG
    assert(pn == px * py * pz);
#endif
}

template <typename T>
typename std::shared_ptr<typename SolverFactory<T>::Solver>
SolverFactory<T>::make_solver() {
    // Axis ranks
    rank_type rz = (pr_ % pz_);
    rank_type ry = (pr_ / pz_) % py_;
    rank_type rx = (pr_ / pz_) / py_;
    auto [bx, Nx] = get_grid_axis_parms(N_, px_, rx);
    auto [by, Ny] = get_grid_axis_parms(N_, py_, ry);
    auto [bz, Nz] = get_grid_axis_parms(N_, pz_, rz);
    auto ccomm = make_communicator(rx, ry, rz, Nx, Ny, Nz);
    if (!gpu_) {
        return std::make_unique<cpu::CPUSolverImpl<T>>(L_, N_, K_, Nx, Ny, Nz, bx, by, bz, ccomm);
    } else {
        return std::make_unique<gpu::GPUSolverImpl<T>>(L_, N_, K_, Nx, Ny, Nz, bx, by, bz, ccomm);
    }
}

} // namespace solver
