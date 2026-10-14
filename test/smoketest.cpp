// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak
//
// Smoke test of the Hopfield public API: construction and argument checks,
// exposure rules and tiling, latch and train contracts, determinism, value
// semantics, the Weights / LoadWeights round-trip, and energy descent at
// every distance of dim 6 and dim 10. Registered with CTest.

#include "Hopfield.h"
#include "Rng.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

static int checks = 0, failures = 0;
// Every check prints one line, pass or fail, so the run is its own report.
#define CHECK(cond, ...) do { const bool ok_ = (cond); ++checks; if (!ok_) ++failures; \
    std::printf(ok_ ? "  pass  " : "  FAIL  "); std::printf(__VA_ARGS__); std::printf("\n"); } while (0)

template <typename F>
static bool Throws(F&& f)
{
    try { f(); } catch (const std::invalid_argument&) { return true; }
    return false;
}

// State errors, not argument errors: a binding maps these to RuntimeError.
// invalid_argument derives from logic_error, so it is excluded first.
template <typename F>
static bool ThrowsLogic(F&& f)
{
    try { f(); }
    catch (const std::invalid_argument&) { return false; }
    catch (const std::logic_error&) { return true; }
    catch (...) { return false; }
    return false;
}

static HopfieldConfig Base(size_t dim = 6, size_t distance = 1)
{
    return {.dim = dim, .gain = 0.1f, .sweep_budget = 32, .rest_tolerance = 0.001f,
            .eta = 0.01f, .input_gain = 5.f, .seed = 3, .init_scale = 0.f, .distance = distance};
}

static size_t Choose(size_t n, size_t k)
{
    size_t r = 1;
    for (size_t i = 1; i <= k; ++i) r = r * (n - k + i) / i;
    return r;
}

static std::vector<float> RandomPattern(size_t n, std::mt19937_64& rng)
{
    std::vector<float> p(n);
    for (auto& x : p) x = static_cast<float>(RngUniform11(rng));
    return p;
}

int main()
{
    std::mt19937_64 rng(11);
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();

    // ---- construction and config validation ----
    {
        CHECK(!Throws([] { Hopfield h(Base()); }), "base config should construct");
        auto bad = [](auto mutate) { HopfieldConfig c = Base(); mutate(c); return Throws([&] { Hopfield h(c); }); };
        CHECK(bad([](auto& c) { c.dim = 0; }), "dim 0 should throw");
        CHECK(bad([](auto& c) { c.dim = 32; }), "dim 32 should throw");
        CHECK(bad([](auto& c) { c.gain = 0.f; }), "gain 0 should throw");
        CHECK(bad([&](auto& c) { c.gain = inf; }), "infinite gain should throw");
        CHECK(bad([](auto& c) { c.sweep_budget = 0; }), "sweep_budget 0 should throw");
        CHECK(bad([](auto& c) { c.rest_tolerance = -1.f; }), "negative rest_tolerance should throw");
        CHECK(bad([&](auto& c) { c.rest_tolerance = nan; }), "NaN rest_tolerance should throw");
        CHECK(bad([](auto& c) { c.eta = 0.f; }), "eta 0 should throw");
        CHECK(bad([](auto& c) { c.input_gain = -0.5f; }), "negative input_gain should throw");
        CHECK(!bad([](auto& c) { c.input_gain = 0.f; }), "input_gain 0 is allowed");
        CHECK(bad([](auto& c) { c.init_scale = 1.5f; }), "init_scale above 1 should throw");
        CHECK(bad([](auto& c) { c.init_scale = -0.1f; }), "negative init_scale should throw");
        CHECK(bad([](auto& c) { c.distance = 0; }), "distance 0 should throw");
        CHECK(bad([](auto& c) { c.distance = 4; }), "distance above dim/2 should throw");
        CHECK(!bad([](auto& c) { c.dim = 1; c.distance = 1; }), "dim 1 distance 1 should construct");

        Hopfield h(Base(6, 2));
        CHECK(h.Dim() == 6 && h.N() == 64 && h.Distance() == 2 && h.Degree() == 15,
              "accessors: dim %zu N %zu distance %zu degree %zu", h.Dim(), h.N(), h.Distance(), h.Degree());
        CHECK(h.Weights().size() == 64 * 15, "weights size %zu", h.Weights().size());
        for (float w : h.Weights()) if (w != 0.f) { CHECK(false, "init_scale 0 should give all-zero weights"); break; }
        CHECK(!h.Latched(), "Latched should be false before the first Latch");
        CHECK(ThrowsLogic([&] { (void)h.Cube(); }), "Cube before Latch should throw logic_error");
        CHECK(ThrowsLogic([&] { (void)h.SweepCount(); }), "SweepCount before Latch should throw logic_error");
        CHECK(ThrowsLogic([&] { (void)h.AtRest(); }), "AtRest before Latch should throw logic_error");
        CHECK(h.Config() == Base(6, 2), "Config should compare equal to the config it was built from");
        { HopfieldConfig other = Base(6, 2); other.gain = 0.2f;
          CHECK(!(h.Config() == other), "Config should compare unequal after a field changes"); }
    }

    // ---- init_scale: seed reproducibility, values in range, symmetric ----
    {
        HopfieldConfig c = Base(6, 3); c.init_scale = 0.5f; c.seed = 42;
        Hopfield a(c), b(c);
        CHECK(a.Weights() == b.Weights(), "same seed should give same initial weights");
        c.seed = 43;
        Hopfield d(c);
        CHECK(a.Weights() != d.Weights(), "different seed should give different initial weights");
        bool in_range = true;
        for (float w : a.Weights()) if (std::fabs(w) > 0.5f) in_range = false;
        CHECK(in_range, "init_scale 0.5 should keep |w| <= 0.5");
        CHECK(!Throws([&] { Hopfield e(c); e.LoadWeights(a.Weights()); }), "initial weights should be symmetric");
    }

    // ---- exposure rules ----
    {
        Hopfield h(Base());
        std::vector<float> ok(64, 0.25f);
        CHECK(!Throws([&] { h.Latch(ok); }), "length-N exposure should be accepted");
        CHECK(!Throws([&] { h.Latch(std::span<const float>(ok).first(1)); }), "length-1 exposure should be accepted");
        CHECK(!Throws([&] { h.Latch(std::span<const float>(ok).first(8)); }), "length-8 exposure should be accepted");
        CHECK(Throws([&] { h.Latch(std::span<const float>(ok).first(0)); }), "empty exposure should throw");
        CHECK(Throws([&] { h.Latch(std::span<const float>(ok).first(3)); }), "length-3 exposure should throw");
        std::vector<float> big(128, 0.f);
        CHECK(Throws([&] { h.Latch(big); }), "exposure longer than N should throw");
        std::vector<float> nf = ok; nf[5] = inf;
        CHECK(Throws([&] { h.Latch(nf); }), "non-finite exposure should throw");
        CHECK(Throws([&] { h.Train(nf); }), "non-finite exposure should throw in Train");
        std::vector<float> wide = ok; wide[0] = 2.f;
        CHECK(!Throws([&] { h.Latch(wide); }), "exposure outside [-1, 1] is allowed in Latch");
        CHECK(Throws([&] { (void)h.Energy(wide); }), "state outside [-1, 1] should throw in Energy");
        CHECK(!Throws([&] { (void)h.Energy(ok, wide); }), "input outside [-1, 1] is allowed in Energy");
    }

    // ---- tiling: a 2^k exposure repeats onto every copy ----
    {
        HopfieldConfig c = Base(); c.input_gain = 10.f; c.gain = 0.1f;
        Hopfield h(c);
        std::vector<float> e = {0.5f, -0.5f, 0.25f, -0.25f};
        auto cube = h.Latch(e);
        CHECK(cube.size() == 64, "latch length %zu", cube.size());
        bool tiled = true;
        for (size_t v = 0; v < 64; ++v)
            if (std::fabs(cube[v] - cube[v & 3]) > 1e-6f) tiled = false;
        CHECK(tiled, "settled cube should be periodic with the exposure length");
        std::vector<float> full(64);
        for (size_t v = 0; v < 64; ++v) full[v] = e[v & 3];
        CHECK(h.Energy(e) == h.Energy(full), "short exposure and its tiling should have the same energy");
        CHECK(h.Energy(e) == h.Energy(full, e) && h.Energy(e) == h.Energy(e, full),
              "state and input may have different lengths and tile the same way");
    }

    // ---- latch and train contracts ----
    {
        HopfieldConfig c = Base(6, 3); c.sweep_budget = 64;
        Hopfield h(c);
        auto e = RandomPattern(64, rng);
        for (int i = 0; i < 20; ++i) h.Train(e);
        const auto w_before = h.Weights();
        CHECK(!h.Latched() && ThrowsLogic([&] { (void)h.Cube(); }), "Train should not latch");
        h.Latch(e);
        CHECK(h.Latched(), "Latched should be true after Latch");
        CHECK(h.Weights() == w_before, "Latch should not change weights");
        CHECK(h.SweepCount() >= 1 && h.SweepCount() <= 64, "sweep count %zu", h.SweepCount());
        CHECK(h.AtRest(), "a settle with a 64-sweep budget should come to rest here (count %zu)", h.SweepCount());
        const std::vector<float> cube_before(h.Cube().begin(), h.Cube().end());
        h.Train(e);
        CHECK(std::vector<float>(h.Cube().begin(), h.Cube().end()) == cube_before, "Train should not change the cube");
        bool in_range = true;
        for (float x : h.Cube()) if (std::fabs(x) >= 1.f) in_range = false;
        CHECK(in_range, "settled states should lie in (-1, 1)");

        HopfieldConfig c1 = c; c1.sweep_budget = 1;
        Hopfield h1(c1);
        h1.LoadWeights(h.Weights());
        h1.Latch(RandomPattern(64, rng));
        CHECK(h1.SweepCount() == 1 && !h1.AtRest(), "budget 1 on a moving cube: count %zu at rest %d", h1.SweepCount(), (int)h1.AtRest());

        const double e0 = h.Energy(e), e1 = h.Energy(h.Cube(), e);
        CHECK(e1 <= e0 + 1e-9, "settle should not raise energy: %.6f -> %.6f", e0, e1);

        HopfieldConfig c2 = c; c2.eta = 0.5f;
        Hopfield h2(c2);
        for (int i = 0; i < 50; ++i) h2.Train(e);
        bool clamped = true;
        for (float w : h2.Weights()) if (w < -1.f || w > 1.f) clamped = false;
        CHECK(clamped, "weights should stay in [-1, 1] after heavy training");

        auto again = h.Latch(h.Cube());
        CHECK(again.size() == 64, "latching a view of the own cube is safe");
    }

    // ---- determinism and value semantics ----
    {
        HopfieldConfig c = Base(6, 2);
        Hopfield a(c), b(c);
        auto e = RandomPattern(64, rng);
        for (int i = 0; i < 10; ++i) { a.Train(e); b.Train(e); }
        a.Latch(e); b.Latch(e);
        CHECK(a.Weights() == b.Weights(), "same training should give same weights");
        CHECK(std::vector<float>(a.Cube().begin(), a.Cube().end()) == std::vector<float>(b.Cube().begin(), b.Cube().end()),
              "same settle should give same cube");

        Hopfield copy = a;
        CHECK(copy.Weights() == a.Weights() && copy.SweepCount() == a.SweepCount(), "copy should carry weights and latch");
        copy.Train(e);
        CHECK(copy.Weights() != a.Weights(), "copy should be independent");
        Hopfield moved = std::move(copy);
        CHECK(moved.Weights().size() == a.Weights().size(), "move should carry weights");
    }

    // ---- every distance: degree, and energy descent with the start held as input ----
    // Even distances are the cases where the in-place index-order sweep, not the
    // bipartite split, carries the descent argument.
    for (size_t dim : {size_t{6}, size_t{10}})
        for (size_t d = 1; d <= dim / 2; ++d) {
            HopfieldConfig c = Base(dim, d); c.init_scale = 1.f; c.seed = 189;
            const size_t N = size_t{1} << dim;
            CHECK(Hopfield(c).Degree() == Choose(dim, d), "dim %zu d %zu degree should be C(dim, d)", dim, d);
            auto start = RandomPattern(N, rng);
            double prev = 0; bool monotone = true;
            for (size_t budget = 1; budget <= 6; ++budget) {
                c.sweep_budget = budget;
                Hopfield h(c);
                h.Latch(start);
                const double e = h.Energy(h.Cube(), start);
                if (budget > 1 && e > prev + 1e-6) monotone = false;
                prev = e;
            }
            CHECK(monotone, "dim %zu d %zu energy should not rise across sweeps 1..6", dim, d);
        }

    // ---- Weights / LoadWeights round-trip ----
    for (size_t d : {size_t{1}, size_t{2}, size_t{3}}) {
        Hopfield trained(Base(6, d));
        auto e = RandomPattern(64, rng);
        for (int i = 0; i < 20; ++i) trained.Train(e);
        HopfieldConfig other = Base(6, d); other.gain = 0.5f; other.input_gain = 1.f;
        Hopfield loaded(other);
        loaded.LoadWeights(trained.Weights());
        CHECK(loaded.Weights() == trained.Weights(), "d %zu weights should round-trip exactly", d);
        CHECK(loaded.Config().gain == 0.5f, "LoadWeights should not touch config");

        Hopfield same(Base(6, d));
        same.LoadWeights(trained.Weights());
        trained.Latch(e); same.Latch(e);
        CHECK(std::vector<float>(trained.Cube().begin(), trained.Cube().end()) ==
              std::vector<float>(same.Cube().begin(), same.Cube().end()), "d %zu loaded memory should settle identically", d);

        const auto& w = trained.Weights();
        CHECK(Throws([&] { loaded.LoadWeights(std::span<const float>(w).first(w.size() - 1)); }), "d %zu short blob should throw", d);
        std::vector<float> longer = w; longer.push_back(0.f);
        CHECK(Throws([&] { loaded.LoadWeights(longer); }), "d %zu long blob should throw", d);
        std::vector<float> range = w; range[7] = 1.5f;
        CHECK(Throws([&] { loaded.LoadWeights(range); }), "d %zu out-of-range blob should throw", d);
        std::vector<float> nanb = w; nanb[7] = nan;
        CHECK(Throws([&] { loaded.LoadWeights(nanb); }), "d %zu NaN blob should throw", d);
        std::vector<float> asym = w; asym[0] = asym[0] == 0.f ? 0.5f : -asym[0];
        CHECK(Throws([&] { loaded.LoadWeights(asym); }), "d %zu asymmetric blob should throw", d);
        CHECK(loaded.Weights() == w, "d %zu rejected blob should leave weights unchanged", d);
        loaded.LoadWeights(loaded.Weights());
        CHECK(loaded.Weights() == w, "d %zu loading the own weights is a no-op", d);

        if (d < 3) {
            Hopfield wrong(Base(6, d + 1));
            CHECK(Throws([&] { wrong.LoadWeights(w); }), "d %zu blob into distance %zu should throw", d, d + 1);
        }

        const size_t sweeps = trained.SweepCount();
        trained.LoadWeights(std::vector<float>(w.size(), 0.f));
        CHECK(trained.SweepCount() == sweeps && trained.Cube().size() == 64, "d %zu LoadWeights should leave the latch record", d);
    }

    std::printf("smoketest: %d checks, %d failed\n", checks, failures);
    return failures != 0;
}
