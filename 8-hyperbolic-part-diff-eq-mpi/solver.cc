#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#ifdef DEBUG
#include <cassert>
#endif

#include <mpi.h>
#include <omp.h>

#define _USE_MATH_DEFINES
#define sqr(x) ((x) * (x))

/*
 * 2D Grid (3D Grid with time dim)
 */
template <typename T = double>
class Grid2D {
public:
    using value_type = T;
    using size_type = typename std::vector<T>::size_type;

private:
    size_type Nx_, Ny_;
    std::vector<T> data_;

public:
    Grid2D(size_type Nx, size_type Ny)
        : Nx_(Nx), Ny_(Ny),
        data_((Nx + 1) * (Ny + 1)) {}

    Grid2D(size_type N): Grid2D(N, N) {}

    inline size_type Nx()   const noexcept { return Nx_ + 1; }
    inline size_type Ny()   const noexcept { return Ny_ + 1; }
    inline size_type size() const noexcept { return data_.size(); }

    inline const size_type index(
        size_type i, size_type j
    ) const noexcept {
#ifdef DEBUG
        assert(i <= Nx_);
        assert(j <= Ny_);
#endif
        return i * Ny() + j;
    }

    const value_type& operator[](size_type index) const noexcept {
        return data_[index];
    }

    inline const value_type& operator()(size_type i, size_type j) const noexcept {
        return data_[index(i, j)];
    }

    inline value_type& operator()(size_type i, size_type j) noexcept {
        return data_[index(i, j)];
    }
};

/*
 * 4D Grid (3D Grid with time dim)
 */
template <typename T = double>
class Grid4D {
public:
    using value_type = T;
    using size_type = typename std::vector<T>::size_type;

private:
    size_type Nx_, Ny_, Nz_, Nt_;
    std::vector<T> data_;

public:
    Grid4D(size_type Nx, size_type Ny, size_type Nz, size_type Nt)
        : Nx_(Nx), Ny_(Ny), Nz_(Nz), Nt_(Nt),
        data_((Nx + 1) * (Ny + 1) * (Nz + 1) * (Nt + 1)) {}

    Grid4D(size_type N, size_type Nt): Grid4D(N, N, N, Nt) {}

    inline size_type Nx()   const noexcept { return Nx_ + 1; }
    inline size_type Ny()   const noexcept { return Ny_ + 1; }
    inline size_type Nz()   const noexcept { return Nz_ + 1; }
    inline size_type Nt()   const noexcept { return Nt_ + 1; }
    inline size_type size() const noexcept { return data_.size(); }

    inline const size_type index(
        size_type i, size_type j, size_type k, size_type n
    ) const noexcept {
#ifdef DEBUG
        assert(i <= Nx_);
        assert(j <= Ny_);
        assert(k <= Nz_);
        assert(n <= Nt_);
#endif
        return ((i * Ny() + j) * Nz() + k) * Nt() + n;
    }

    const value_type& operator[](size_type index) const noexcept {
        return data_[index];
    }

    inline const value_type& operator()(
        size_type i, size_type j, size_type k, size_type n
    ) const noexcept {
        return data_[index(i, j, k, n)];
    }

    inline value_type& operator()(
        size_type i, size_type j, size_type k, size_type n
    ) noexcept {
        return data_[index(i, j, k, n)];
    }
};

/*
 * Cube Edges Enum
 */
enum Edge {
    LEFT, RIGHT, // x edges
    BACK, FRONT, // y edges
    DOWN, UP     // z edges
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
template <typename T = double>
struct Communicator {
    using value_type = T;
    using size_type = typename std::vector<T>::size_type;
    using rank_type = int;

    rank_type pr;                 // process rank to communicate
    Tag tag;                      // Tag to separate ordinary and periodic communication
    Grid2D<size_type> index;      // indecies to send (opt)
    std::vector<value_type> send; // preallocated buff to send
    std::vector<value_type> recv; // preallocated buff to receive

    Communicator(rank_type rank, size_type Nx, size_type Ny, Tag tag = DEFAULT):
        pr(rank), tag(tag), index(Nx, Ny), send(index.size()), recv(index.size()) {}

    inline const value_type& operator()(size_type i, size_type j) const noexcept {
        return recv[index.index(i, j)];
    }
};

#define EDGE_NUMBER (UP + 1)

/*
 * Abstraction for interprocess communication with cube topology
 */
template <typename T = double>
class CubeCommunicator {
public:
    using value_type = T;
    using size_type = typename Communicator<T>::size_type;
    using rank_type = typename Communicator<T>::rank_type;
    using Comm = Communicator<T>;
    using Comms = std::array<std::shared_ptr<Comm>, EDGE_NUMBER>;
    using Edges = std::vector<Edge>;
    using Communicatables = std::array<bool, EDGE_NUMBER>;

private:
    Comms comms_;                       // Comm ptrs for each cube side
    rank_type cn_;                      // Number of side with valid comm
    const Edges edges_;                 // Valid side with comm
    std::vector<MPI_Request> requests_; // MPI requests prealloceted verctor
    std::vector<MPI_Status> statuses_;  // MPI statuses prealloceted verctor

    static Edges get_edges(const Comms& comms) {
        Edges edges;
        for (int i = LEFT; i < EDGE_NUMBER; i++) {
            auto edge = static_cast<Edge>(i);
            if (comms[edge]) {
                edges.push_back(edge);
            }
        }
        return edges;
    }

    CubeCommunicator(Comms comms, Edges edges):
        comms_(comms), edges_(edges), cn_(edges.size()),
        requests_(2 * cn_), statuses_(2 * cn_) {}

public:
    CubeCommunicator(Comms comms): CubeCommunicator(comms, get_edges(comms)) {}

    void initialize_comm(Grid4D<value_type>& u, size_type n) {
        for (size_t c = 0; c < cn_; c++) {
            auto& comm = *comms_[edges_[c]];
            auto size = comm.index.size();
            #pragma omp parallel for
            for (size_t i = 0; i < size; i++) {
                auto index = comm.index[i] * u.Nt() + n;
                comm.send[i] = u[index];
            }
            MPI_Isend(comm.send.data(), size, MPI_DOUBLE, comm.pr, comm.tag, MPI_COMM_WORLD, &requests_[2 * c]);
            MPI_Irecv(comm.recv.data(), size, MPI_DOUBLE, comm.pr, comm.tag, MPI_COMM_WORLD, &requests_[2 * c + 1]);
        }
    }

    void finalize_comm() {
        MPI_Waitall(2 * cn_, requests_.data(), statuses_.data());
    }


    const Comm& operator[](Edge edge) const noexcept {
        return *comms_[edge];
    }

    bool contains(Edge edge) const noexcept {
        return comms_[edge] != nullptr;
    }

    Communicatables get_communicatables() const noexcept {
        Communicatables communicatables;
        for (int i = LEFT; i < EDGE_NUMBER; i++) {
            auto edge = static_cast<Edge>(i);
            communicatables[edge] = contains(edge);
        }
        return communicatables;
    }
};

/*
 * Analytical function
 */
template <typename T = double>
class AnalyticalFunction {
public:
    using value_type = T;

private:
    T lx_ = 1;
    T ly_ = 2;
    T lz_ = 3;
    T at_;
    
protected:
    T Lx_, Ly_, Lz_;

public:
    AnalyticalFunction(T Lx, T Ly, T Lz)
        : Lx_(Lx), Ly_(Ly), Lz_(Lz),
        at_(M_PI / 2 * std::sqrt(
            sqr(lx_ / Lx)
          + sqr(ly_ / Ly)
          + sqr(lz_ / Lz)
        ))
    {}
    
    T operator()(T x, T y, T z, T t) const noexcept {
        return std::sin(M_PI * lx_ / Lx_ * x)
             * std::sin(M_PI * ly_ / Ly_ * y)
             * std::sin(M_PI * lz_ / Lz_ * z)
             * std::cos(at_ * t);
    }
};

template<typename T = double>
class Solver {
public:
    using value_type = T;
    using size_type = long long;// typename std::vector<T>::size_type;
    using rank_type = typename Communicator<T>::rank_type;
    using Communicatables = typename CubeCommunicator<T>::Communicatables;

private:
    size_type Nx_, Ny_, Nz_;
    size_type K_;
    value_type h_;
    value_type t_;
    value_type a_;
    value_type c_;
    CubeCommunicator<value_type> ccomm_;
    const Communicatables communicatables_; // Edge checking opt

    // Analytical Function as Grid
    class AnalyticalFunctionGrid: AnalyticalFunction<T> {
    private:
        size_type bx_, by_, bz_;
        T hx_, hy_, hz_;
        T t_;

    public:
        AnalyticalFunctionGrid(
            T Lx, T Ly, T Lz,
            T hx, T hy, T hz, T t,
            size_type bx, size_type by, size_type bz
        ):
            AnalyticalFunction<T>(Lx, Ly, Lz),
            hx_(hx), hy_(hy), hz_(hz), t_(t),
            bx_(bx), by_(by), bz_(bz) {}

        AnalyticalFunctionGrid(T L, T h, T t, size_type bx, size_type by, size_type bz):
            AnalyticalFunctionGrid(L, L, L, h, h, h, t, bx, by, bz) {}
        T operator()(size_type i, size_type j, size_type k, size_type n) const noexcept {
            return this->AnalyticalFunction<T>::operator()(
                hx_ * (bx_ + i),
                hy_ * (by_ + j),
                hz_ * (bz_ + k),
                 t_ * n
            );
        }
    } u_analytical_;

    // Initial Data Function as Grid
    class InitialFunctionGrid: AnalyticalFunctionGrid {
    public:
        InitialFunctionGrid(const AnalyticalFunctionGrid& func)
            : AnalyticalFunctionGrid(func) {}

        T operator()(size_type i, size_type j, size_type k) const noexcept {
            return this->AnalyticalFunctionGrid::operator()(i, j, k, 0);
        }
    } phi_;

    inline auto diff(
        Grid4D<value_type>& u, size_type i, size_type j, size_type k, size_type n
    ) const noexcept {
        return std::abs(u_analytical_(i, j, k, n) - u(i, j, k, n));
    }

    inline bool is_inner_node(
            size_type i, size_type j, size_type k
    ) const noexcept {
        return i > 0 && i < Nx_
            && j > 0 && j < Ny_
            && k > 0 && k < Nz_;
    }

    inline bool is_boundary_node(
            size_type i, size_type j, size_type k
    ) const noexcept {
        return i == 0   && !communicatables_[LEFT]
            || i == Nx_ && !communicatables_[RIGHT]
            || k == 0   && !communicatables_[DOWN]
            || k == Nz_ && !communicatables_[UP];
    }

    inline bool calc_boundary(
            size_type i, size_type j, size_type k
    ) const noexcept {
            return 0;
    }

    inline value_type calc_phi_node(
            const Grid4D<value_type>& u,
            size_type i, size_type j, size_type k
    ) const noexcept {
        // Believe to loop unrolling...
        auto left  = (i == 0)   ? ccomm_[LEFT ](j, k) : phi_(i - 1, j, k);
        auto right = (i == Nx_) ? ccomm_[RIGHT](j, k) : phi_(i + 1, j, k);
        auto down  = (k == 0)   ? ccomm_[DOWN ](i, j) : phi_(i, j, k - 1);
        auto up    = (k == Nz_) ? ccomm_[UP   ](i, j) : phi_(i, j, k + 1);
        auto back  = (j == 0)
                   ? communicatables_[BACK]
                   ? ccomm_[BACK](i, k)
                   : phi_(i, Ny_ - 1, k)
                   : phi_(i, j   - 1, k);
        auto front = (j == Ny_)
                   ? communicatables_[FRONT]
                   ? ccomm_[FRONT](i, k)
                   : phi_(i,     1, k)
                   : phi_(i, j + 1, k);
        auto tmp = -6 * phi_(i, j, k) + left + right + back + front + down + up;
        return u(i, j, k, 0) + c_ / 2 * tmp;
    }

    // Optimized version for inner nodes
    // NOTE: There is no constexpr in c++11 :(
    inline value_type calc_phi_inner_node(
            const Grid4D<value_type>& u,
            size_type i, size_type j, size_type k
    ) const noexcept {
        auto left  = phi_(i - 1, j, k);
        auto right = phi_(i + 1, j, k);
        auto back  = phi_(i, j - 1, k);
        auto front = phi_(i, j + 1, k);
        auto down  = phi_(i, j, k - 1);
        auto up    = phi_(i, j, k + 1);
        auto tmp = -6 * phi_(i, j, k) + left + right + back + front + down + up;
        return u(i, j, k, 0) + c_ / 2 * tmp;
    }

    inline value_type calc_node(
            const Grid4D<value_type>& u,
            size_type i, size_type j, size_type k, size_type n
    ) const noexcept {
        // Believe to loop unrolling...
        auto left  = (i == 0)   ? ccomm_[LEFT ](j, k) : u(i - 1, j, k, n);
        auto right = (i == Nx_) ? ccomm_[RIGHT](j, k) : u(i + 1, j, k, n);
        auto down  = (k == 0)   ? ccomm_[DOWN ](i, j) : u(i, j, k - 1, n);
        auto up    = (k == Nz_) ? ccomm_[UP   ](i, j) : u(i, j, k + 1, n);
        auto back  = (j == 0)
                   ? communicatables_[BACK]
                   ? ccomm_[BACK](i, k)
                   : u(i, Ny_ - 1, k, n)
                   : u(i, j   - 1, k, n);
        auto front = (j == Ny_)
                   ? communicatables_[FRONT]
                   ? ccomm_[FRONT](i, k)
                   : u(i,     1, k, n)
                   : u(i, j + 1, k, n);
        auto tmp = -6 * u(i, j, k, n) + left + right + back + front + down + up;
        return c_ * tmp + 2 * u(i, j, k, n) - u(i, j, k, n - 1);
    }

    // Optimized version for inner nodes
    // NOTE: There is no constexpr in c++11 :(
    inline value_type calc_inner_node(
            const Grid4D<value_type>& u,
            size_type i, size_type j, size_type k, size_type n
    ) const noexcept {
        auto left  = u(i - 1, j, k, n);
        auto right = u(i + 1, j, k, n);
        auto back  = u(i, j - 1, k, n);
        auto front = u(i, j + 1, k, n);
        auto down  = u(i, j, k - 1, n);
        auto up    = u(i, j, k + 1, n);
        auto tmp = -6 * u(i, j, k, n) + left + right + back + front + down + up;
        return c_ * tmp + 2 * u(i, j, k, n) - u(i, j, k, n - 1);
    }

public:
    Solver(
        value_type L, size_type N, size_type K,
        size_type Nx, size_type Ny, size_type Nz,
        size_type bx, size_type by, size_type bz,
        CubeCommunicator<value_type> ccomm,
        value_type a = value_type(0.5), value_type g = value_type(0.5)
    ):
        Nx_(Nx), Ny_(Ny), Nz_(Nz), K_(K),
        h_(L / N), t_(g * h_), a_(a), c_(sqr(a_ * t_ / h_)),
        u_analytical_(L, h_, t_, bx, by, bz), phi_(u_analytical_),
        ccomm_(ccomm), communicatables_(ccomm.get_communicatables()) {}

    auto solve() noexcept {
        Grid4D<value_type> u(Nx_, Ny_, Nz_, K_);
        value_type error = 0;
        auto start = omp_get_wtime();
        // 0 step
        size_type n = 0;
        #pragma omp parallel
        {
            value_type t_error = 0;
            #pragma omp for
            for (size_type i = 0; i <= Nx_; i++) {
                for (size_type j = 0; j <= Ny_; j++) {
                    for (size_type k = 0; k <= Nz_; k++) {
                        u(i, j, k, n) = is_boundary_node(i, j, k)
                                          ? calc_boundary(i, j, k)
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
        // 1 step
        ccomm_.initialize_comm(u, n);
        n++;
        // Inner
        #pragma omp parallel
        {
            value_type t_error = 0;
            #pragma omp for
            for (size_type i = 1; i < Nx_; i++) {
                for (size_type j = 1; j < Ny_; j++) {
                    for (size_type k = 1; k < Nz_; k++) {
                        u(i, j, k, n) = calc_phi_inner_node(u, i, j, k);
                        t_error = std::max(t_error, diff(u, i, j, k, n));
                    }
                }
            }
            #pragma omp critical
            if (error < t_error) {
                error = t_error;
            }
        } // end pragma omp parallel
        ccomm_.finalize_comm();
        // Edge
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
                                      ? calc_boundary(i, j, k)
                                      : calc_phi_node(u, i, j, k);
                        t_error = std::max(t_error, diff(u, i, j, k, n));
                    }
                }
            }
            #pragma omp critical
            if (error < t_error) {
                error = t_error;
            }
        } // end pragma omp parallel
        // Boundary condition
        // Tail steps
        for (; n < K_; n++) {
            ccomm_.initialize_comm(u, n);
            // Inner
            #pragma omp parallel
            {
                value_type t_error = 0;
                #pragma omp for
                for (size_type i = 1; i < Nx_; i++) {
                    for (size_type j = 1; j < Ny_; j++) {
                        for (size_type k = 1; k < Nz_; k++) {
                            u(i, j, k, n + 1) = calc_inner_node(u, i, j, k, n);
                            t_error = std::max(t_error, diff(u, i, j, k, n + 1));
                        }
                    }
                }
                #pragma omp critical
                if (error < t_error) {
                    error = t_error;
                }
            } // end pragma omp parallel
            ccomm_.finalize_comm();
            // Edge
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
                                              ? calc_boundary(i, j, k)
                                              : calc_node(u, i, j, k, n);
                            t_error = std::max(t_error, diff(u, i, j, k, n + 1));
                        }
                    }
                }
                #pragma omp critical
                if (error < t_error) {
                    error = t_error;
                }
            } // end pragma omp parallel
        }
        auto end = omp_get_wtime();
        auto time = end - start;
        // Aggregate metrics
        value_type global_error;
        MPI_Reduce(&error, &global_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        value_type global_time;
        MPI_Reduce(&time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        return std::tuple(std::move(u), global_error, global_time);
    }
};

/*
 * Solver Factory
 */
template<typename T = double>
class SolverFactory {
public:
    using value_type = T;
    using size_type = typename Solver<T>::size_type;
    using rank_type = typename Communicator<T>::rank_type;
    using Comm = typename CubeCommunicator<T>::Comm;
    using Comms = typename CubeCommunicator<T>::Comms;

private:
    value_type L_;
    size_type N_;
    size_type K_;
    rank_type px_, py_, pz_;
    rank_type pn_;
    rank_type pr_;

    // Get rank begin index and size on grid axis
    auto get_grid_axis_parms(size_type N, rank_type pn, rank_type pr) const noexcept {
        size_type d = (N + 1) / pn;
        size_type m = (N + 1) % pn;
        size_type b = pr * d + std::min<size_type>(pr, m);
        size_type n = d + (pr < m ? 1 : 0) - 1;
        return std::tuple(b, n);
    }

    inline auto get_rank(
        rank_type rx, rank_type ry, rank_type rz
    ) const noexcept {
        return (rx * py_ + ry) * pz_ + rz;
    }

    auto make_communicator(
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
            comms[LEFT] = std::make_shared<Comm>(comm);
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
            comms[RIGHT] = std::make_shared<Comm>(comm);
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
                comms[BACK] = std::make_shared<Comm>(comm);
            } else {
                // Periodic
                auto r = get_rank(rx, py_ - 1, rz);
                Comm comm(r, Nx, Nz, PERIODIC);
                auto offset = Nz + 1;
                for (size_type i = 0; i <= Nx; i++) {
                    for (size_type k = 0; k <= Nz; k++) {
                        comm.index(i, k) = offset + i * (Ny + 1) * (Nz + 1) + k;
                    }
                }
                comms[BACK] = std::make_shared<Comm>(comm);
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
                comms[FRONT] = std::make_shared<Comm>(comm);
            } else {
                // Periodic
                auto r = get_rank(rx, 0, rz);
                Comm comm(r, Nx, Nz, PERIODIC);
                auto offset = (Ny - 1) * (Nz + 1);
                for (size_type i = 0; i <= Nx; i++) {
                    for (size_type k = 0; k <= Nz; k++) {
                        comm.index(i, k) = offset + i * (Ny + 1) * (Nz + 1) + k;
                    }
                }
                comms[FRONT] = std::make_shared<Comm>(comm);
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
            comms[DOWN] = std::make_shared<Comm>(comm);
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
            comms[UP] = std::make_shared<Comm>(comm);
        }
        return CubeCommunicator<value_type>(comms);
    }

public:
    SolverFactory(
        value_type L, size_type N, size_type K,
        rank_type px, rank_type py, rank_type pz,
        rank_type pn, rank_type pr
    ):
        L_(L), N_(N), K_(K),
        px_(px), py_(py), pz_(pz), pn_(pn), pr_(pr)
    {
#ifdef DEBUG
        assert(pn == px * py * pz);
#endif
    }

    Solver<value_type> make_solver() {
        // Axis ranks
        rank_type rz = (pr_ % pz_);
        rank_type ry = (pr_ / pz_) % py_;
        rank_type rx = (pr_ / pz_) / py_;
        auto [bx, Nx] = get_grid_axis_parms(N_, px_, rx);
        auto [by, Ny] = get_grid_axis_parms(N_, py_, ry);
        auto [bz, Nz] = get_grid_axis_parms(N_, pz_, rz);
        auto ccomm = make_communicator(rx, ry, rz, Nx, Ny, Nz);
        return Solver(L_, N_, K_, Nx, Ny, Nz, bx, by, bz, ccomm);
    }
};

int main(int argc, char** argv) {
    // Help
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " N px py pz\n"
            << "N is a Node Number per Side\n"
            << "px is a x axis decomposition param\n"
            << "py is a y axis decomposition param\n"
            << "pz is a z axis decomposition param\n";
        return 1;
    }
    // Get args
    auto N = std::stoll(argv[1]);
    auto px = std::stoi(argv[2]);
    auto py = std::stoi(argv[3]);
    auto pz = std::stoi(argv[4]);
    size_t K = 20;
    // Init MPI
    // NOTE: I would like to move it inside solver maker using std::optional
    // But we have to deal with c++11 in patch version
    // So keep it here is the most logical decision
    int status, pn, pr;
    status = MPI_Init(&argc, &argv);
    if (status != MPI_SUCCESS) {
        std::cerr << "Failed to init MPI: " << status << "\n";
        return 1;
    }
    status = MPI_Comm_size(MPI_COMM_WORLD, &pn);
    if (status != MPI_SUCCESS) {
        std::cerr << "Failed to get process number: " << status << "\n";
        return 1;
    }
    status = MPI_Comm_rank(MPI_COMM_WORLD, &pr);
    if (status != MPI_SUCCESS) {
        std::cerr << "Failed to get process rank: " << status << "\n";
        return 1;
    }
    // Solve
    for(auto L: {1.0, M_PI}) {
        if (pr == 0) {
            std::cout << "L = " << L << " "
                      << "N = " << N << " "
                      << "K = " << K << "\n";
        }
        auto solver = SolverFactory(L, N, K, px, py, pz, pn, pr).make_solver();
        auto [_, err, time] = solver.solve();
        if (pr == 0) {
            std::cout << "Err:  " << err  << "\n"
                      << "Time: " << time << "\n"
                      << "\n";
        }
    }
    MPI_Finalize();
    return 0;
}
