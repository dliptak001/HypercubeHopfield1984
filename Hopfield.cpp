// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#include "Hopfield.h"
#include "Rng.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

static_assert(sizeof(size_t) >= 8,
              "Hopfield requires a 64-bit build");

namespace {

/// Finite test on the exponent bits, so it still works under -ffast-math,
/// which may compile std::isfinite to a constant.
bool FiniteBits(float x)
{
    return (std::bit_cast<uint32_t>(x) & 0x7f800000u) != 0x7f800000u;
}

/// Integral from 0 to s of artanh(x), divided by gain.
/// At exactly ±1 this is the limit, ln(2) / gain.
double VertexTerm(float s, float gain)
{
    const double g = static_cast<double>(gain);
    const double x = static_cast<double>(s);
    if (x == 1.0 || x == -1.0)
        return std::log(2.0) / g;
    const double a = 1.0 + x;
    const double b = 1.0 - x;
    return 0.5 * (a * std::log(a) + b * std::log(b)) / g;
}

float Clamp11(float w)
{
    if (w > 1.f)
        return 1.f;
    if (w < -1.f)
        return -1.f;
    return w;
}

} // namespace

Hopfield::Hopfield(const HopfieldConfig& cfg)
    : cfg_(cfg)
{
    if (cfg_.dim < 1 || cfg_.dim > 31)
        throw std::invalid_argument("Hopfield dim must be in [1, 31]");
    if (!FiniteBits(cfg_.gain) || !(cfg_.gain > 0.f))
        throw std::invalid_argument("Hopfield gain must be finite and > 0");
    if (cfg_.sweep_budget < 1)
        throw std::invalid_argument("Hopfield sweep_budget must be >= 1");
    if (!FiniteBits(cfg_.eta) || !(cfg_.eta > 0.f))
        throw std::invalid_argument("Hopfield eta must be finite and > 0");
    if (!FiniteBits(cfg_.rest_tolerance) || cfg_.rest_tolerance < 0.f)
        throw std::invalid_argument("Hopfield rest_tolerance must be finite and >= 0");
    if (cfg_.distance < 1 || cfg_.distance > std::max<size_t>(1, cfg_.dim / 2))
        throw std::invalid_argument("Hopfield distance must be in [1, max(1, dim / 2)]");
    if (!FiniteBits(cfg_.init_scale) || cfg_.init_scale < 0.f || cfg_.init_scale > 1.f)
        throw std::invalid_argument("Hopfield init_scale must be finite and in [0, 1]");
    if (!FiniteBits(cfg_.input_gain) || cfg_.input_gain < 0.f)
        throw std::invalid_argument("Hopfield input_gain must be finite and >= 0");

    n_ = size_t{1} << cfg_.dim;

    // Gosper's hack walks every dim-bit integer with `distance` bits set, ascending.
    // At distance 1 this is 1, 2, 4, ..., the axes in order.
    {
        uint64_t count = 1;   // C(dim, distance), exact in 64 bits for dim <= 31
        for (uint64_t i = 1; i <= cfg_.distance; ++i)
            count = count * (cfg_.dim - cfg_.distance + i) / i;
        masks_.reserve(count);
        top_.reserve(count);
        const uint64_t limit = uint64_t{1} << cfg_.dim;
        uint64_t m = (uint64_t{1} << cfg_.distance) - 1;
        while (m < limit)
        {
            masks_.push_back(static_cast<uint32_t>(m));
            top_.push_back(uint32_t{1} << (std::bit_width(m) - 1));
            const uint64_t c = m & (~m + 1);
            const uint64_t r = m + c;
            m = (((r ^ m) >> 2) / c) | r;
        }
    }

    const size_t degree = masks_.size();
    if (n_ > std::numeric_limits<size_t>::max() / degree)
        throw std::invalid_argument("Hopfield weight count overflows");

    weight_.assign(n_ * degree, 0.f);
    state_.assign(n_, 0.f);
    input_.assign(n_, 0.f);

    if (cfg_.init_scale == 0.f)
        return;

    // One draw per edge, at the owner, vertices in index order and masks in
    // order. Written to both ends so the matrix is symmetric from the start.
    std::mt19937_64 rng(cfg_.seed);
    for (size_t v = 0; v < n_; ++v)
    {
        for (size_t k = 0; k < degree; ++k)
        {
            if ((static_cast<uint32_t>(v) & top_[k]) != 0)
                continue;
            float w = static_cast<float>(RngUniform11(rng));
            // The draw is in [-1, 1). A value within half an ulp of 1
            // rounds to 1 when stored as float. Keep that weight below 1
            // so only the training clamp reaches the upper end.
            if (w == 1.f)
                w = std::nextafter(1.f, 0.f);
            w *= cfg_.init_scale;
            const size_t u = v ^ masks_[k];
            weight_[v * degree + k] = w;
            weight_[u * degree + k] = w;
        }
    }
}

void Hopfield::CheckExposure(std::span<const float> exposure) const
{
    const size_t m = exposure.size();
    if (m == 0 || m > n_ || (m & (m - 1)) != 0)
        throw std::invalid_argument(
            "Hopfield exposure length must be a power of two in [1, N]");
    for (float x : exposure)
    {
        if (!FiniteBits(x))
            throw std::invalid_argument("Hopfield exposure must be finite");
    }
}

float Hopfield::Field(size_t vertex) const
{
    const size_t degree = masks_.size();
    const float* row = weight_.data() + vertex * degree;
    float h = cfg_.input_gain * input_[vertex];
    for (size_t k = 0; k < degree; ++k)
        h += state_[vertex ^ masks_[k]] * row[k];
    return h;
}

bool Hopfield::Sweep()
{
    bool same = true;
    auto rewrite = [&](bool even) {
        for (size_t v = 0; v < n_; ++v)
        {
            const bool even_popcount = (std::popcount(static_cast<uint32_t>(v)) & 1u) == 0u;
            if (even_popcount != even)
                continue;
            const float next = std::tanh(cfg_.gain * Field(v));
            if (std::fabs(next - state_[v]) > cfg_.rest_tolerance)
                same = false;
            state_[v] = next;
        }
    };
    rewrite(true);
    rewrite(false);
    return same;
}

double Hopfield::Energy(std::span<const float> state, std::span<const float> input) const
{
    CheckExposure(state);
    CheckExposure(input);
    for (float s : state)
    {
        if (s < -1.f || s > 1.f)
            throw std::invalid_argument("Hopfield energy is undefined outside [-1, 1]");
    }

    const size_t degree = masks_.size();
    const size_t smask = state.size() - 1;
    const size_t imask = input.size() - 1;
    const double beta = static_cast<double>(cfg_.input_gain);
    double e = 0.0;
    for (size_t v = 0; v < n_; ++v)
    {
        const float* row = weight_.data() + v * degree;
        const double sv = static_cast<double>(state[v & smask]);
        for (size_t k = 0; k < degree; ++k)
        {
            if ((static_cast<uint32_t>(v) & top_[k]) != 0)
                continue;
            const size_t u = v ^ masks_[k];
            const double w = static_cast<double>(row[k]);
            const double su = static_cast<double>(state[u & smask]);
            e -= w * sv * su;
        }
        e -= beta * static_cast<double>(input[v & imask]) * sv;
        e += VertexTerm(state[v & smask], cfg_.gain);
    }
    return e;
}

std::span<const float> Hopfield::Latch(std::span<const float> exposure)
{
    CheckExposure(exposure);
    // input_ is never handed out, so it cannot alias the exposure. Tile into
    // it first, then copy to state_, which the exposure may be a view of.
    const size_t mask = exposure.size() - 1;
    for (size_t v = 0; v < n_; ++v)
        input_[v] = exposure[v & mask];
    state_ = input_;

    sweep_count_ = 0;
    at_rest_ = false;
    for (size_t sweep = 0; sweep < cfg_.sweep_budget; ++sweep)
    {
        const bool unchanged = Sweep();
        ++sweep_count_;
        if (unchanged)
        {
            at_rest_ = true;
            break;
        }
    }

    latched_ = true;
    return state_;
}

void Hopfield::Train(std::span<const float> exposure)
{
    CheckExposure(exposure);
    const size_t degree = masks_.size();
    const size_t mask = exposure.size() - 1;
    for (size_t v = 0; v < n_; ++v)
    {
        float* row = weight_.data() + v * degree;
        for (size_t k = 0; k < degree; ++k)
        {
            if ((static_cast<uint32_t>(v) & top_[k]) != 0)
                continue;
            const size_t u = v ^ masks_[k];
            const float ev = exposure[v & mask];
            const float eu = exposure[u & mask];
            const float w = Clamp11(row[k] + cfg_.eta * ev * eu);
            row[k] = w;
            weight_[u * degree + k] = w;
        }
    }
}

void Hopfield::LoadWeights(std::span<const float> weights)
{
    const size_t degree = masks_.size();
    if (weights.size() != weight_.size())
        throw std::invalid_argument("Hopfield::LoadWeights length must be N * degree");
    // Loading the live matrix into itself is a no-op; std::copy onto its own
    // source is undefined, so return before it.
    if (weights.data() == weight_.data())
        return;
    for (float w : weights)
    {
        if (!FiniteBits(w) || w < -1.f || w > 1.f)
            throw std::invalid_argument("Hopfield::LoadWeights weights must be finite and in [-1, 1]");
    }
    for (size_t v = 0; v < n_; ++v)
    {
        for (size_t k = 0; k < degree; ++k)
        {
            if ((static_cast<uint32_t>(v) & top_[k]) != 0)
                continue;
            const size_t u = v ^ masks_[k];
            if (weights[v * degree + k] != weights[u * degree + k])
                throw std::invalid_argument("Hopfield::LoadWeights weights must be symmetric across each edge");
        }
    }
    std::copy(weights.begin(), weights.end(), weight_.begin());
}

std::span<const float> Hopfield::Cube() const
{
    if (!latched_)
        throw std::logic_error("Hopfield::Cube before Latch");
    return state_;
}

size_t Hopfield::SweepCount() const
{
    if (!latched_)
        throw std::logic_error("Hopfield::SweepCount before Latch");
    return sweep_count_;
}

bool Hopfield::AtRest() const
{
    if (!latched_)
        throw std::logic_error("Hopfield::AtRest before Latch");
    return at_rest_;
}
