#pragma once

#include <mpi.h>
#include <omp.h>

#include "grid.hpp"

namespace solver {
namespace comm {

/*
 * EdgeArray<T> Realization
 */
template <typename T>
EdgeArray<T>::EdgeArray(const array_type& arr): data_(arr) {}

template <typename T>
EdgeArray<T>::EdgeArray(std::initializer_list<value_type> init) {
    size_t i = 0;
    for (auto& v : init) {
        if (i >= data_.size()) {
            break;
        }
        data_[i++] = v;
    }
}

template <typename T>
typename EdgeArray<T>::value_type&
EdgeArray<T>::operator[](Edge edge) noexcept {
    return data_[static_cast<size_t>(edge)];
}

template <typename T>
const typename EdgeArray<T>::value_type&
EdgeArray<T>::operator[](Edge edge) const noexcept {
    return data_[static_cast<size_t>(edge)];
}

/*
 * Communicator<T> Realization
 */
template <typename T>
Communicator<T>::Communicator(rank_type pr, size_type Nx, size_type Ny, Tag tag):
    pr(pr), tag(tag), index(Nx, Ny), send(index.size()), recv(index.size()) {}

template <typename T>
inline const typename Communicator<T>::value_type&
Communicator<T>::operator()(size_type i, size_type j) const noexcept {
    return recv[index.index(i, j)];
}

/*
 * CubeCommunicator<T> Realization
 */
template <typename T>
typename CubeCommunicator<T>::Edges
CubeCommunicator<T>::get_edges(const Comms& comms) {
    Edges edges;
    for (auto edge: all_edges) {
        if (comms[edge]) {
            edges.push_back(edge);
        }
    }
    return edges;
}

template <typename T>
CubeCommunicator<T>::CubeCommunicator(Comms comms, Edges edges):
    comms_(comms), edges_(edges), cn_(edges.size()),
    requests_(2 * cn_), statuses_(2 * cn_) {}

template <typename T>
CubeCommunicator<T>::CubeCommunicator(Comms comms):
    CubeCommunicator(comms, get_edges(comms)) {}

template <typename T>
template <size_t queue_size>
void CubeCommunicator<T>::initialize_comm(
    grid::Grid<value_type, queue_size>& u,
    size_type n
) {
    for (size_t c = 0; c < cn_; c++) {
        auto& comm = *comms_[edges_[c]];
        auto size = comm.index.size();
        auto offset = u.step(n);
        #pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            auto index = comm.index[i] + offset;
            comm.send[i] = u[index];
        }
        MPI_Isend(comm.send.data(), size, MPI_DOUBLE, comm.pr,
                  comm.tag, MPI_COMM_WORLD, &requests_[2 * c]);
        MPI_Irecv(comm.recv.data(), size, MPI_DOUBLE, comm.pr,
                  comm.tag, MPI_COMM_WORLD, &requests_[2 * c + 1]);
    }
}

template <typename T>
void CubeCommunicator<T>::finalize_comm() {
    MPI_Waitall(2 * cn_, requests_.data(), statuses_.data());
}

template <typename T>
const typename CubeCommunicator<T>::Comm&
CubeCommunicator<T>::operator[](Edge edge) const noexcept {
    return *comms_[edge];
}

template <typename T>
bool CubeCommunicator<T>::contains(Edge edge) const noexcept {
    return comms_[edge] != nullptr;
}

template <typename T>
auto CubeCommunicator<T>::get_communicatables() const noexcept {
    Communicatables communicatables;
    for (auto edge: all_edges) {
        communicatables[edge] = contains(edge);
    }
    return communicatables;
}

/*
 * CommDevicePoolManager Realization
 */
template <typename T>
CubeCommunicator<T>::CommDevicePoolManager::CommDevicePoolManager(
    const Edges& edges, const Comms& comms
)
    : edges_(edges), comms_(comms), pools_({nullptr})
{
    for (auto edge: edges_) {
        auto& comm = comms_[edge];
        pools_[edge] = std::make_shared<Pool>(comm->index.Nx(), comm->index.Ny());
    }
}

template <typename T>
void CubeCommunicator<T>::CommDevicePoolManager::to_gpu() {
    for (auto edge: edges_) {
        auto& comm = comms_[edge];
        pools_[edge]->to_gpu(comm->recv.data());
    }
}

template <typename T>
auto CubeCommunicator<T>::CommDevicePoolManager::view(Edge edge) const noexcept {
    return pools_[edge] != nullptr ? pools_[edge]->view() : DeviceView();
}

template <typename T>
auto CubeCommunicator<T>::CommDevicePoolManager::view() const noexcept {
    return CommDeviceView {
        view(Edge::Left ),
        view(Edge::Right),
        view(Edge::Back ),
        view(Edge::Front),
        view(Edge::Down ),
        view(Edge::Up   )
    };
}

} // namespace grid
} // namespace solver
