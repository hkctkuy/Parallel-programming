#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

#ifdef DEBUG
#include <cassert>
#endif

#include <omp.h>

#define _USE_MATH_DEFINES
#define sqr(x) ((x) * (x))

/*
 * 4D Grid (3D Grid with time dim)
 */
template <typename T = double>
class Grid4D {
public:
    using value_type = T;
    using size_type = typename std::vector<T>::size_type;

private:
    size_type Nx_;
    size_type Ny_;
    size_type Nz_;
    size_type Nt_;
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

    inline const value_type& operator()(
        size_type i, size_type j, size_type k, size_type n
    ) const noexcept {
#ifdef DEBUG
        assert(i <= Nx_);
        assert(j <= Ny_);
        assert(k <= Nz_);
        assert(n <= Nt_);
#endif
        return data_[((i * Ny() + j) * Nz() + k) * Nt() + n];
    }

    inline value_type& operator()(
        size_type i, size_type j, size_type k, size_type n
    ) noexcept {
#ifdef DEBUG
        assert(i <= Nx_);
        assert(j <= Ny_);
        assert(k <= Nz_);
        assert(n <= Nt_);
#endif
        return data_[((i * Ny() + j) * Nz() + k) * Nt() + n];
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
    using size_type = typename std::vector<T>::size_type;

private:
    size_type N_;
    size_type K_;
    value_type h_;
    value_type t_;
    value_type a_;

    // Analytical Function as Grid
    class AnalyticalFunctionGrid: AnalyticalFunction<T> {
    private:
        T hx_;
        T hy_;
        T hz_;
        T t_;

    public:
        AnalyticalFunctionGrid(T Lx, T Ly, T Lz, T hx, T hy, T hz, T t):
            AnalyticalFunction<T>(Lx, Ly, Lz), hx_(hx), hy_(hy), hz_(hz), t_(t) {}

        AnalyticalFunctionGrid(T L, T h, T t):
            AnalyticalFunctionGrid(L, L, L, h, h, h, t) {}

        T operator()(size_type i, size_type j, size_type k, size_type n) const noexcept {
            return this->AnalyticalFunction<T>::operator()(
                hx_ * i,
                hy_ * j,
                hz_ * k,
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

public:
    Solver(
        value_type L, size_type N, size_type K,
        value_type a = value_type(0.5), value_type g = value_type(0.5)
    ):
        N_(N), K_(K), h_(L / N), t_(g * h_), a_(a),
        u_analytical_(L, h_, t_), phi_(u_analytical_)
    {}

    auto solve() const noexcept {
        Grid4D<value_type> u(N_, K_);
        value_type error = 0;
        auto start = omp_get_wtime();
        auto coef = sqr(a_ * t_ / h_);
        #pragma omp parallel
        {
            value_type t_error = 0;
            // x boundary condition
            #pragma omp for
            for (size_type j = 0; j <= N_; j++) {
                for (size_type k = 0; k <= N_; k++) {
                    for (size_type n = 0; n <= K_; n++) {
                        u(0 , j, k, n) = 0;
                        u(N_, j, k, n) = 0;
                        t_error = std::max(t_error, diff(u, 0 , j, k, n));
                        t_error = std::max(t_error, diff(u, N_, j, k, n));
                    }
                }
            }
            // z boundary condition
            #pragma omp for
            for (size_type i = 1; i < N_; i++) {
                for (size_type j = 0; j <= N_; j++) {
                    for (size_type n = 0; n <= K_; n++) {
                        u(i, j, 0 , n) = 0;
                        u(i, j, N_, n) = 0;
                        t_error = std::max(t_error, diff(u, i, j, 0 , n));
                        t_error = std::max(t_error, diff(u, i, j, N_, n));
                    }
                }
            }
            // 0 step
            #pragma omp for
            for (size_type i = 1; i < N_; i++) {
                for (size_type j = 0; j <= N_; j++) {
                    for (size_type k = 1; k < N_; k++) {
                        u(i, j, k, 0) = phi_(i, j, k);
                        t_error = std::max(t_error, diff(u, i, j, k, 0));
                    }
                }
            }
            // 1 step
            // y periodic condition
            #pragma omp for
            for (size_type i = 1; i < N_; i++) {
                for (size_type k = 1; k < N_; k++) {
                    auto tmp = -6 * phi_(i, 0, k)
                             + phi_(i - 1,  0, k) + phi_(i + 1, 0, k)
                             + phi_(i, N_ - 1, k) + phi_(i,     1, k)
                             + phi_(i, 0,  k - 1) + phi_(i, 0, k + 1);
                    auto value = u(i, 0, k, 0) + coef / 2 * tmp;
                    u(i, 0 , k, 1) = value;
                    u(i, N_, k, 1) = value;
                    t_error = std::max(t_error, diff(u, i, 0 , k, 1));
                    t_error = std::max(t_error, diff(u, i, N_, k, 1));
                }
            }
            // inner
            #pragma omp for
            for (size_type i = 1; i < N_; i++) {
                for (size_type j = 1; j < N_; j++) {
                    for (size_type k = 1; k < N_; k++) {
                        auto tmp = -6 * phi_(i, j, k)
                                 + phi_(i - 1, j, k) + phi_(i + 1, j, k)
                                 + phi_(i, j - 1, k) + phi_(i, j + 1, k)
                                 + phi_(i, j, k - 1) + phi_(i, j, k + 1);
                        u(i, j, k, 1) = u(i, j, k, 0) + coef / 2 * tmp;
                        t_error = std::max(t_error, diff(u, i, j, k, 1));
                    }
                }
            }
            #pragma omp critical
            if (error < t_error) {
                error = t_error;
            }
        } // end pragma omp parallel
        // Tail steps
        for (size_type n = 1; n < K_; n++) {
            #pragma omp parallel
            {
                value_type t_error = 0;
                #pragma omp for
                for (size_type i = 1; i < N_; i++) {
                    // y periodic condition
                    for (size_type k = 1; k < N_; k++) {
                        auto tmp = -6 * u(i, 0, k, n)
                                 + u(i - 1,  0, k, n) + u(i + 1, 0, k, n)
                                 + u(i, N_ - 1, k, n) + u(i, 1,     k, n)
                                 + u(i, 0,  k - 1, n) + u(i, 0, k + 1, n);
                        auto value = coef * tmp + 2 * u(i, 0, k, n) - u(i, 0, k, n - 1);
                        u(i, 0 , k, n + 1) = value;
                        u(i, N_, k, n + 1) = value;
                        t_error = std::max(t_error, diff(u, i, 0 , k, n + 1));
                        t_error = std::max(t_error, diff(u, i, N_, k, n + 1));
                    }
                    // inner
                    for (size_type j = 1; j < N_; j++) {
                        for (size_type k = 1; k < N_; k++) {
                            auto tmp = -6 * u(i, j, k, n)
                                     + u(i - 1, j, k, n) + u(i + 1, j, k, n)
                                     + u(i, j - 1, k, n) + u(i, j + 1, k, n)
                                     + u(i, j, k - 1, n) + u(i, j, k + 1, n);
                            u(i, j, k, n + 1) = coef * tmp
                                    + 2 * u(i, j, k, n) - u(i, j, k, n - 1);
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
        return std::tuple(std::move(u), error, time);
    }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " N T\n"
            << "N is a Node Number per Side\n"
            << "T is a Thread Number\n";
        return 1;
    }
    size_t N = std::stoll(argv[1]);
    size_t T = std::stoll(argv[2]);
    omp_set_num_threads(T);
    size_t K = 20;
    for(auto L: {1.0, M_PI}) {
        std::cout << "L = " << L << " "
                  << "N = " << N << " "
                  << "K = " << K << std::endl;
        auto solver = Solver(L, N, K);
        auto [_, err, time] = solver.solve();
        std::cout << "Err:  " << err  << std::endl
                  << "Time: " << time << std::endl
                  << std::endl;
    }
    return 0;
}
