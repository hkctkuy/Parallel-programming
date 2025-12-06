#pragma once

#include <array>
#include <memory>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include "grid.hpp"
#include "device.hpp"

namespace solver {
namespace comm {

/*
 * Cube Edges Enum
 */
enum class Edge: uint8_t {
    Left, Right,   // x edges
    Back, Front,   // y edges
    Down, Up,      // z edges
    Count          // sentinel value
};

constexpr size_t edge_count() noexcept {
    return static_cast<size_t>(Edge::Count);
}

/*
 * Array of all edges
 */
constexpr std::array<Edge, edge_count()> all_edges {
    Edge::Left, Edge::Right,
    Edge::Back, Edge::Front,
    Edge::Down, Edge::Up
};

/*
 * Array with all edges as indices
 */
template <typename T>
class EdgeArray {
public:
    using value_type = T;
    using array_type = std::array<value_type, edge_count()>;

private:
    array_type data_;

public:
    EdgeArray() = default;
    EdgeArray(const array_type& arr);
    EdgeArray(std::initializer_list<value_type> init);

    value_type& operator[](Edge edge) noexcept;
    const value_type& operator[](Edge edge) const noexcept;
};

/*
 * MPI tag enum
 */
enum Tag {
    DEFAULT, // For ordinary neighbor data exchange
    PERIODIC // For "periodic" data exchange (between back and from edges)
};

/*
 * Struct for interprocess data exchange
 */
template <typename T>
struct Communicator {
    using value_type = T;
    using size_type = typename std::vector<T>::size_type;
    using rank_type = int;

    rank_type pr;   // process rank to communicate
    Tag tag;        // Tag to separate ordinary and periodic communication
 
    grid::Grid<size_type> index;  // indecies to send (opt)
    std::vector<value_type> send; // preallocated buff to send
    std::vector<value_type> recv; // preallocated buff to receive

    Communicator(rank_type pr, size_type Nx, size_type Ny, Tag tag = DEFAULT);

    inline const value_type& operator()(size_type i, size_type j) const noexcept;
};

/*
 * Bool array of all available for communication edges
 */
using Communicatables = EdgeArray<bool>;

/*
 * Abstraction for interprocess communication with cube topology
 */
template <typename T>
class CubeCommunicator {
public:
    using Comm = Communicator<T>;

    using value_type = typename Comm::value_type;
    using size_type = typename Comm::size_type;
    using rank_type = typename Comm::rank_type;

    using Pool = typename device::DevicePoolManager<value_type, size_type>;
    using DeviceView = typename Pool::DeviceView;

    using Edges = std::vector<Edge>;
    using Comms = EdgeArray<std::shared_ptr<Comm>>;
    using Pools = EdgeArray<std::shared_ptr<Pool>>;

private:
    Comms comms_;       // Comm ptrs for each cube side
    rank_type cn_;      // Number of side with valid comm
    const Edges edges_; // Valid side with comm

    std::vector<MPI_Request> requests_; // MPI requests prealloceted verctor
    std::vector<MPI_Status> statuses_;  // MPI statuses prealloceted verctor

    static Edges get_edges(const Comms& comms);

    CubeCommunicator(Comms comms, Edges edges);
public:
    explicit CubeCommunicator(Comms comms);

    // Interporcess exchage
    template <size_t queue_size>
    void initialize_comm(grid::Grid<value_type, queue_size>& u, size_type n);
    void finalize_comm();

    // Access
    const Comm& operator[](Edge edge) const noexcept;
    bool contains(Edge edge) const noexcept;

    auto get_communicatables() const noexcept;

    /*
     * Special Device View for Cube Communicator
     */
    struct CommDeviceView {
        // NOTE: shared ptr is not trivially copyable
        DeviceView left;
        DeviceView right;
        DeviceView front;
        DeviceView back;
        DeviceView down;
        DeviceView up;
    };

    // Check DeviceView triviality
    static_assert(std::is_trivially_copyable_v<CommDeviceView> == true);
    static_assert(std::is_standard_layout_v<CommDeviceView> == true);

private:
    /*
     * Special Device Pool for Cube Communicator
     */
    class CommDevicePoolManager {
        Pools pools_;
        const Edges& edges_;
        const Comms& comms_;

        auto view(Edge edge) const noexcept;

    public:
        CommDevicePoolManager(const Edges& edges, const Comms& comms);

        // NOTE: No need to load data back from GPU
        void to_gpu();

        auto view() const noexcept;
    };

    std::shared_ptr<CommDevicePoolManager> dm_ = nullptr;

public:
    void enable_gpu() {
        if (dm_) { return; }
        dm_ = std::make_shared<CommDevicePoolManager>(edges_, comms_);
    }

    void disable_gpu() {
        if (!dm_) { return; }
        dm_ = nullptr;
    }

    void to_gpu() {
        if (!dm_) { return; }
        dm_->to_gpu();
    }

    auto view() const noexcept {
        if (!dm_) {
            return CommDeviceView {
                DeviceView(), DeviceView(), DeviceView(),
                DeviceView(), DeviceView(), DeviceView()
            };
        }
        return dm_->view();
    }
};

} // namespace comm
} // namespace solver

#include "communicator.tpp"
