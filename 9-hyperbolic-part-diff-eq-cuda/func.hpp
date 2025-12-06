#pragma once

#include <cmath>

#define _USE_MATH_DEFINES
#define sqr(x) ((x) * (x))

namespace solver {
namespace func {

/*
 * Problem analytical function
 */
template <typename T>
class AnalyticalFunction {
public:
    using value_type = T;

private:
    T lx_ = 1;
    T ly_ = 2;
    T lz_ = 3;
    T Lx_, Ly_, Lz_;
    T at_;

public:
    AnalyticalFunction(T Lx, T Ly, T Lz)
        : Lx_(Lx), Ly_(Ly), Lz_(Lz),
        at_(M_PI / 2 * std::sqrt(
            sqr(lx_ / Lx)
          + sqr(ly_ / Ly)
          + sqr(lz_ / Lz)
        ))
    {}

    AnalyticalFunction(T L): AnalyticalFunction(L, L, L) {}

    __host__ __device__
    T operator()(T x, T y, T z, T t) const noexcept {
        return std::sin(M_PI * lx_ / Lx_ * x)
             * std::sin(M_PI * ly_ / Ly_ * y)
             * std::sin(M_PI * lz_ / Lz_ * z)
             * std::cos(at_ * t);
    }

};

// Check triviality
static_assert(std::is_trivially_copyable_v<AnalyticalFunction<double>> == true);
static_assert(std::is_standard_layout_v<AnalyticalFunction<double>> == true);

/*
 * Problem boundary condition function
 */
template <typename T>
class BoundaryFunction {
public:
    using value_type = T;

    __host__ __device__
    T operator()(T x, T y, T z, T t) const noexcept {
        return 0;
    }
};

// Check triviality
static_assert(std::is_trivially_copyable_v<BoundaryFunction<double>> == true);
static_assert(std::is_standard_layout_v<BoundaryFunction<double>> == true);

/*
 * Analytical Function View as Grid
 * template class Function heed to have Function::value_type type and operator()
 */
template <class Function, typename size_type>
class FunctionGridView {
public:
    using value_type = typename Function::value_type;

private:
    Function f_;
    value_type hx_, hy_, hz_, t_; // Grid steps
    size_type bx_, by_, bz_, bt_; // Beginning nodes

public:
    FunctionGridView(
        Function f,
        value_type hx, value_type hy, value_type hz, value_type t,
        size_type bx, size_type by, size_type bz, size_type bt
    ):
        f_(f),
        hx_(hx), hy_(hy), hz_(hz), t_(t),
        bx_(bx), by_(by), bz_(bz), bt_(bt) {}

    FunctionGridView(
        Function F,
        value_type h, value_type t,
        size_type bx, size_type by, size_type bz, size_type bt = 0
    ):
        FunctionGridView(F, h, h, h, t, bx, by, bz, bt) {}

    __host__ __device__
    auto operator()(
        size_type i, size_type j, size_type k, size_type n
    ) const noexcept {
        return f_(
            hx_ * (bx_ + i),
            hy_ * (by_ + j),
            hz_ * (bz_ + k),
             t_ * (bt_ + n)
        );
    }
};

// Check triviality
static_assert(std::is_trivially_copyable_v<FunctionGridView<AnalyticalFunction<double>, size_t>> == true);
static_assert(std::is_standard_layout_v<FunctionGridView<AnalyticalFunction<double>, size_t>> == true);

} // namespace func
} // namespace solver
