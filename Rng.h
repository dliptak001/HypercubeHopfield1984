// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak

#pragma once

#include <cstdint>
#include <random>

/// Portable uniform draws from std::mt19937_64. The engine is fully
/// specified by the standard; the library distributions are not, so these
/// take the engine's bits directly and give the same sequence on every
/// platform. Hopfield uses them for its starting weights when init_scale is
/// above zero; the diagnostics use them for patterns and noise.

/// 53 high bits of one draw, mapped to [0, 1).
[[nodiscard]] inline double RngUniform01(std::mt19937_64& rng)
{
    return static_cast<double>(rng() >> 11) * 0x1p-53;
}

/// RngUniform01 mapped to [-1, 1).
[[nodiscard]] inline double RngUniform11(std::mt19937_64& rng)
{
    return RngUniform01(rng) * 2.0 - 1.0;
}
