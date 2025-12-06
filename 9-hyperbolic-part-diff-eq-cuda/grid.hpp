#pragma once

#include <memory>
#include <vector>

#ifdef DEBUG
#include <cassert>
#endif

#include "device.hpp"

namespace solver {
namespace grid {

/*
 * Multidimensional Data Grid
 * Support from 0 to 4 dimensions
 */
template <typename T = double, size_t queue_size = 1>
class Grid {
public:
    using value_type = T;
    using size_type = typename std::vector<T>::size_type;

    using DevicePoolManager = typename device::DevicePoolManager<T, size_type, queue_size>;
    using DeviceView = typename DevicePoolManager::DeviceView;

private:
    size_type Nx_, Ny_, Nz_, Nt_;
    std::vector<T> data_;

    // CUDA support
    std::shared_ptr<DevicePoolManager> dm_ = nullptr;

public:
    Grid(
        size_type Nx = 0, size_type Ny = 0, size_type Nz = 0, size_type Nt = 0
    ):
        Nx_(Nx), Ny_(Ny), Nz_(Nz), Nt_(Nt),
        data_((Nx + 1) * (Ny + 1) * (Nz + 1) * (Nt + 1)) {}

    inline size_type Nx()   const noexcept { return Nx_ + 1; }
    inline size_type Ny()   const noexcept { return Ny_ + 1; }
    inline size_type Nz()   const noexcept { return Nz_ + 1; }
    inline size_type Nt()   const noexcept { return Nt_ + 1; }
    inline size_type size() const noexcept { return data_.size(); }

    inline size_type step(size_type n) const noexcept {
        return Nx() * Ny() * Nz() * n;
    }

    inline const size_type index(
        size_type i = 0, size_type j = 0, size_type k = 0, size_type n = 0
    ) const noexcept {
#ifdef DEBUG
        assert(i < Nx());
        assert(j < Ny());
        assert(k < Nz());
        assert(n < Nt());
#endif
        return ((n * Nx() + i) * Ny() + j) * Nz() + k;
    }

    const value_type& operator[](size_type index) const noexcept {
        return data_[index];
    }

    inline const value_type& operator()(
        size_type i = 0, size_type j = 0, size_type k = 0, size_type n = 0
    ) const noexcept {
        return data_[index(i, j, k, n)];
    }

    inline value_type& operator()(
        size_type i = 0, size_type j = 0, size_type k = 0, size_type n = 0
    ) noexcept {
        return data_[index(i, j, k, n)];
    }

    // CUDA sopport
    // NOTE: Enabling enabled GPU does nothing:)
    void enable_gpu() {
        if (dm_) { return; }
        dm_ = std::make_shared<DevicePoolManager>(Nx(), Ny(), Nz());
    }

    // NOTE: Disabling disabled GPU does nothing:)
    void disable_gpu() {
        if (!dm_) { return; }
        dm_ = nullptr;
    }

    void to_gpu(size_type n) {
        if (!dm_) { return; }
        dm_->to_gpu(data_.data() + step(n));
    }

    void from_gpu(size_type n) {
        if (!dm_) { return; }
        dm_->from_gpu(data_.data() + step(n));
        dm_->roll();
    }

    auto view() const noexcept {
        if (!dm_) {
            return DeviceView();
        }
        return dm_->view();
    }
};

} // namespace grid
} // namespace solver
