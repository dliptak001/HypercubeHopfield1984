// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

/// Construction parameters, fixed when the class is built. dim, gain,
/// sweep_budget, and eta have no valid default and must be set.
/// docs/hopfield.md, "Operating points", says how to pick the values.
struct HopfieldConfig
{
    /// Hypercube dimension, in [1, 31]. N = 2^dim vertices.
    size_t dim = 0;

    /// Scale on the local field inside tanh. Finite, > 0. Train never reads
    /// it. Of order 1 / degree keeps a trained field in tanh's working range.
    float gain = 0.f;

    /// Most sweeps one settle will run. >= 1.
    size_t sweep_budget = 0;

    /// Largest |next - current| at any vertex for a sweep to count as rest.
    /// Finite, >= 0. Zero requires the written float to equal the old one.
    float rest_tolerance = 0.f;

    /// Step one exposure adds to an edge weight. Finite, > 0. Weights are
    /// clamped to [-1, 1] and a clamped weight has lost its graded value;
    /// keep eta * passes * sqrt(patterns) well under 4.
    float eta = 0.f;

    /// Scale of the input current I_v = input_gain * e_v that the exposure
    /// applies to every field during a settle. Finite, >= 0. Zero makes the
    /// exposure a starting state only. Of order degree / 10 balances the
    /// weight sum.
    float input_gain = 1.f;

    /// Seeds the draw of the starting weights when init_scale > 0; unused
    /// otherwise. The same seed gives the same start.
    uint64_t seed = 0;

    /// Scale of the starting weights. Finite, in [0, 1]. Zero, the default
    /// and Hopfield's model, starts every weight at 0. Above zero each edge
    /// starts at init_scale times one uniform draw from [-1, 1).
    float init_scale = 0.f;

    /// Hamming distance between neighbors, in [1, max(1, dim / 2)]. Each
    /// vertex has C(dim, distance) neighbors. 1 is the hypercube itself.
    size_t distance = 1;

    bool operator==(const HopfieldConfig&) const = default;
};

/// Hopfield's graded associative memory (J. J. Hopfield, Proc. Natl. Acad.
/// Sci. USA 81, 3088-3092, 1984) on the graph that joins hypercube vertices
/// at one Hamming distance. Latch loads an exposure as the starting state,
/// holds it as the input current, and sweeps even-popcount vertices then
/// odd-popcount vertices, each in place and in index order, to
/// tanh(gain * (sum of w * s_u + I_v)), until a sweep moves every vertex by
/// at most the rest tolerance or the budget runs out. Train adds eta * e_v *
/// e_u to each edge and clamps it to [-1, 1]. Each edge's weight is stored in
/// both endpoints' rows so a field is one contiguous dot product; Train
/// writes both. An exposure of 2^k values, k <= dim, is read at the low k
/// bits of each vertex, so a short exposure repeats on every copy of the
/// low-k cube. Plain value type. Not thread-safe for concurrent calls on one
/// instance. docs/hopfield.md has the full description.
///
/// Buffers. All storage is allocated once in the constructor and never
/// reallocated: 4 * N * degree bytes of weights plus two N-float cubes.
/// The class never keeps a pointer into a caller's buffer; every argument
/// span is consumed within the call, so a temporary is safe. Spans returned
/// by Latch and Cube, and the reference from Weights, point into this object:
/// they stay valid and live (Train and LoadWeights change the contents in
/// place) until the object is destroyed or moved from, and the next Latch
/// overwrites the cube. A language binding should copy them out rather than
/// expose a view. A moved-from Hopfield may only be assigned to or destroyed.
///
/// Tuning. eta is the training rate and gain the attractor strength, and
/// they are independent: lower eta for slower training, lower gain for
/// shallower basins (gain * degree of order 1 is the example's point; well
/// below that an exposure can escape a stored pattern). gain * input_gain
/// sets how much of the exposure passes through, so when gain drops, raise
/// input_gain in proportion to keep it. distance and dim set capacity, not
/// strength. examples/recall_benchmark.cpp measures any point.
class Hopfield
{
public:
    /// Validate @p cfg, allocate, and set the starting weights.
    /// @throws std::invalid_argument if dim is not in [1, 31], distance is
    ///         not in [1, max(1, dim / 2)], gain is not finite and > 0,
    ///         sweep_budget is 0, eta is not finite and > 0,
    ///         rest_tolerance is not finite and >= 0, init_scale is
    ///         not finite and in [0, 1], or input_gain is not finite and >= 0.
    /// @throws std::bad_alloc if the N x degree weights do not fit in memory.
    explicit Hopfield(const HopfieldConfig& cfg);

    /// Load @p exposure as the starting state and the input current, then
    /// settle. Weights unchanged. @p exposure has length 2^k, 0 <= k <= dim,
    /// every value finite; a view of the current cube is allowed. Returns the
    /// settled cube, length N, overwritten by the next Latch.
    /// @throws std::invalid_argument if the length is not a power of two in
    ///         [1, N], or a value is not finite.
    std::span<const float> Latch(std::span<const float> exposure);

    /// Add eta * e_v * e_u to each edge and clamp to [-1, 1]. No sweep; the
    /// cube is unchanged. A view of the cube is a valid exposure; a view of
    /// Weights() is not, since Train rewrites those rows while reading them.
    /// @throws std::invalid_argument on the same exposure rules as Latch.
    void Train(std::span<const float> exposure);

    /// Hopfield's energy of @p state under the current weights with @p input
    /// applied as the current I_v = input_gain * input_v. Each is tiled onto
    /// the cube like an exposure; they need not have the same length.
    /// @throws std::invalid_argument on the exposure rules for either
    ///         argument, or if a state value lies outside [-1, 1].
    [[nodiscard]] double Energy(std::span<const float> state,
                                std::span<const float> input) const;

    /// Energy(exposure, exposure): where a settle of @p exposure starts.
    /// Energy(Cube(), exposure) is where it ended.
    [[nodiscard]] double Energy(std::span<const float> exposure) const
    {
        return Energy(exposure, exposure);
    }

    /// True once Latch has been called; Cube, SweepCount, and AtRest are
    /// valid only then.
    [[nodiscard]] bool Latched() const { return latched_; }

    /// The settled cube from the most recent Latch. Length N, overwritten by
    /// the next Latch.
    /// @throws std::logic_error before the first Latch.
    [[nodiscard]] std::span<const float> Cube() const;

    /// Sweeps the most recent Latch ran, 1 through sweep_budget.
    /// @throws std::logic_error before the first Latch.
    [[nodiscard]] size_t SweepCount() const;

    /// True if the most recent Latch stopped on the rest tolerance, false if
    /// the budget stopped it while the cube was still moving.
    /// @throws std::logic_error before the first Latch.
    [[nodiscard]] bool AtRest() const;

    /// All weights, row-major N x degree: entry v * Degree() + k is the edge
    /// from v along mask k and equals the entry at (v ^ mask k, k). Every
    /// value is in [-1, 1]. A live reference: Train changes it in place.
    /// Persistence is the caller's: write these out and hand them to
    /// LoadWeights on a Hopfield with the same dim and distance.
    [[nodiscard]] const std::vector<float>& Weights() const { return weight_; }

    /// Replace every weight. The cube, SweepCount, and AtRest are left as
    /// they were; the cube is a settle of the old weights until the next Latch.
    /// Passing this object's own Weights() is a no-op.
    /// @throws std::invalid_argument if @p weights is not N x Degree() long,
    ///         holds a value that is not finite or lies outside [-1, 1], or
    ///         differs between the two rows of any edge.
    void LoadWeights(std::span<const float> weights);

    [[nodiscard]] size_t Dim() const { return cfg_.dim; }
    [[nodiscard]] size_t N() const { return n_; }
    [[nodiscard]] size_t Distance() const { return cfg_.distance; }
    /// Neighbors per vertex, C(dim, distance).
    [[nodiscard]] size_t Degree() const { return masks_.size(); }
    [[nodiscard]] HopfieldConfig Config() const { return cfg_; }

private:
    void CheckExposure(std::span<const float> exposure) const;
    [[nodiscard]] float Field(size_t vertex) const;
    /// @return true when every rewrite moved its vertex by at most the rest tolerance.
    [[nodiscard]] bool Sweep();

    HopfieldConfig cfg_{};
    size_t n_ = 0;
    /// Every index with exactly cfg_.distance bits set, ascending.
    /// Neighbor k of v is v ^ masks_[k].
    std::vector<uint32_t> masks_;
    /// Highest set bit of each mask. The edge along mask k is owned by the
    /// endpoint whose bit there is clear; Train and Energy visit each edge once.
    std::vector<uint32_t> top_;
    /// Row-major N x degree. weight_[v * degree + k] is the edge from v along
    /// mask k, equal to weight_[(v ^ masks_[k]) * degree + k].
    std::vector<float> weight_;
    std::vector<float> state_;
    /// The exposure of the current settle, tiled to length N.
    std::vector<float> input_;
    bool latched_ = false;
    size_t sweep_count_ = 0;
    bool at_rest_ = false;
};
