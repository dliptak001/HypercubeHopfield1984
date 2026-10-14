// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Charles Liptak
//
// How the Hopfield class handles graded exemplars, scored on magnitudes:
// cosine and RMS error of the latch against the stored pattern. Patterns are
// 1024 floats: the first 50 are the real exemplars, any beyond that are
// synthesized uniform in [-1, 1] from a fixed seed. At dim above 10 each
// pattern is tiled onto every copy of the low-10-bit cube. Three modes:
//   tune  dim [P] [d]   gain x input_gain x eta grid around the scaling rule
//   table dim [P]       the scaling-rule point at every distance, P from 1 up
//   scale dim [P] [d]   a wide gain x input_gain grid at one distance
// A fifth argument names the exemplar file; the default is kPath. If the
// file cannot be opened every pattern is synthesized and the banner says so.
// Columns: stored is a stored pattern latched as itself; repair is a pattern
// with 25% of its sites randomized, cosine to the pattern before and after the
// settle, and how often the latch is nearest the right pattern; novel is a
// never-stored pattern, cosine to its own input and to the nearest stored one.
// Grid cells are independent (one Hopfield each) and run on a pool of
// hardware_concurrency - 1 threads; rows print in grid order when all are done.

#include "Hopfield.h"
#include "Rng.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

constexpr size_t kE = 1024;
constexpr size_t kReal = 50;
// HOPFIELD_DATA_DIR is the repository's data/ directory, set by CMake.
constexpr char kPath[] = HOPFIELD_DATA_DIR "/exemplars.bin";
constexpr int kPasses = 50;
constexpr double kCorrupt = 0.25;
constexpr int kTrials = 3;

using Pat = std::vector<float>;

static HopfieldConfig Cfg(size_t dim, size_t distance, float input_gain, float gain, float eta)
{
    return {.dim = dim, .gain = gain, .sweep_budget = 64, .rest_tolerance = 0.01f,
            .eta = eta, .input_gain = input_gain, .seed = 0, .distance = distance};
}

// The scaling rule, normalized to the measured dim 10 point (degree 252:
// gain 0.03, input_gain 20).
static float RuleGain(size_t degree) { return 7.56f / static_cast<float>(degree); }
static float RuleInputGain(size_t degree) { return static_cast<float>(degree) / 12.6f; }

// cube has length N; e has length 2^k and is tiled the way Latch tiles it.
static double Cos(std::span<const float> cube, std::span<const float> e)
{
    const size_t m = e.size() - 1;
    double ab = 0, aa = 0, bb = 0;
    for (size_t v = 0; v < cube.size(); ++v) {
        const double x = cube[v], y = e[v & m];
        ab += x * y; aa += x * x; bb += y * y;
    }
    return (aa > 0 && bb > 0) ? ab / std::sqrt(aa * bb) : 0.0;
}

static double Rms(std::span<const float> cube, std::span<const float> e)
{
    const size_t m = e.size() - 1;
    double s = 0;
    for (size_t v = 0; v < cube.size(); ++v) { const double d = cube[v] - e[v & m]; s += d * d; }
    return std::sqrt(s / cube.size());
}

static size_t Nearest(std::span<const float> cube, const std::vector<Pat>& ps)
{
    size_t best = 0; double bc = -2;
    for (size_t j = 0; j < ps.size(); ++j) { const double c = Cos(cube, ps[j]); if (c > bc) { bc = c; best = j; } }
    return best;
}

// Reads up to min(P, kReal) real patterns from @p path and synthesizes the
// rest. @p real receives how many were actually read, which is what the
// banner must report: a missing or short file falls through to synthesis.
static std::vector<Pat> LoadPatterns(size_t P, const char* path, size_t& real)
{
    std::vector<Pat> ps;
    std::ifstream in(path, std::ios::binary);
    if (!in)
        std::fprintf(stderr, "cannot open %s; every pattern will be synthesized\n", path);
    Pat p(kE);
    while (ps.size() < std::min(P, kReal) && in.read(reinterpret_cast<char*>(p.data()), kE * sizeof(float)))
        ps.push_back(p);
    real = ps.size();
    std::mt19937_64 rng(1234);
    while (ps.size() < P) {
        for (auto& x : p) x = static_cast<float>(RngUniform11(rng));
        ps.push_back(p);
    }
    return ps;
}

struct Row { double stored_cos, stored_rms, in_cos, out_cos, id, novel_self, novel_max; };

static Row Measure(Hopfield& net, const std::vector<Pat>& ps, std::mt19937_64& rng)
{
    Row r{};
    const size_t P = ps.size();
    for (auto& e : ps) {
        auto latch = net.Latch(e);
        r.stored_cos += Cos(latch, e);
        r.stored_rms += Rms(latch, e);
    }
    r.stored_cos /= P; r.stored_rms /= P;

    int tot = 0;
    for (size_t i = 0; i < P; ++i)
        for (int t = 0; t < kTrials; ++t, ++tot) {
            Pat q = ps[i];
            for (auto& x : q) if (RngUniform01(rng) < kCorrupt) x = static_cast<float>(RngUniform11(rng));
            r.in_cos += Cos(q, ps[i]);
            auto latch = net.Latch(q);
            r.out_cos += Cos(latch, ps[i]);
            r.id += Nearest(latch, ps) == i;
        }
    r.in_cos /= tot; r.out_cos /= tot; r.id = 100.0 * r.id / tot;

    for (int t = 0; t < 5; ++t) {
        Pat q(kE);
        for (auto& x : q) x = static_cast<float>(RngUniform11(rng));
        auto latch = net.Latch(q);
        r.novel_self += Cos(latch, q);
        double mx = -2; for (auto& e : ps) mx = std::max(mx, Cos(latch, e));
        r.novel_max += mx;
    }
    r.novel_self /= 5; r.novel_max /= 5;
    return r;
}

static Row Run(size_t dim, size_t d, float ig, float gain, float eta, const std::vector<Pat>& ps)
{
    Hopfield net(Cfg(dim, d, ig, gain, eta));
    for (int it = 0; it < kPasses; ++it) for (auto& e : ps) net.Train(e);
    std::mt19937_64 rng(7);
    return Measure(net, ps, rng);
}

static void PrintRow(const Row& r)
{
    std::printf("%.3f / %.3f  | %.3f -> %.3f  %3.0f%%  | %.3f  %.3f\n",
                r.stored_cos, r.stored_rms, r.in_cos, r.out_cos, r.id, r.novel_self, r.novel_max);
}

// A grid cell: its row label and the work that produces its Row.
struct Cell { std::string label; std::function<Row()> run; };

// Run every cell on a worker pool, then print them in order.
static void RunCells(const std::vector<Cell>& cells)
{
    std::vector<Row> rows(cells.size());
    std::atomic<size_t> next{0}, done{0};
    const unsigned workers = std::max(1u, std::thread::hardware_concurrency() - 1);
    std::vector<std::thread> pool;
    for (size_t w = 0; w < std::min<size_t>(workers, cells.size()); ++w)
        pool.emplace_back([&] {
            for (size_t i = next++; i < cells.size(); i = next++) {
                rows[i] = cells[i].run();
                std::fprintf(stderr, "  cell %zu/%zu done\n", ++done, cells.size());
            }
        });
    for (auto& t : pool) t.join();
    for (size_t i = 0; i < cells.size(); ++i) { std::printf("%s", cells[i].label.c_str()); PrintRow(rows[i]); }
    std::fflush(stdout);
}

static std::string Fmt(const char* f, auto... a)
{
    char b[128]; std::snprintf(b, sizeof b, f, a...); return b;
}

int main(int argc, char** argv)
{
    const std::string_view mode = argc > 1 ? argv[1] : "table";
    const size_t dim = argc > 2 ? static_cast<size_t>(std::atoi(argv[2])) : 10;
    const size_t P = argc > 3 ? static_cast<size_t>(std::atoi(argv[3])) : 50;
    const size_t dmid = std::max<size_t>(1, dim / 2);
    const size_t d = argc > 4 ? static_cast<size_t>(std::atoi(argv[4])) : dmid;
    const char* path = argc > 5 ? argv[5] : kPath;
    if (dim < 10 || dim > 31) { std::printf("dim must be 10..31 (patterns are 1024 floats)\n"); return 1; }
    // Checked here, on the main thread: a throw inside a pool worker would terminate without a message.
    if (d < 1 || d > dmid) { std::printf("d must be 1..%zu for dim %zu\n", dmid, dim); return 1; }

    size_t real = 0;
    const std::vector<Pat> ps = LoadPatterns(P, path, real);
    std::printf("dim %zu, %zu patterns (%zu real from %s, %zu synthesized), %d passes, %.0f%% corrupted\n",
                dim, P, real, path, P - real, kPasses, 100 * kCorrupt);

    if (mode == "tune") {
        const size_t degree = Hopfield(Cfg(dim, d, 1.f, 1.f, 1.f)).Degree();
        const float g0 = RuleGain(degree), i0 = RuleInputGain(degree);
        std::printf("tune: distance %zu, degree %zu, rule gain %.4f input_gain %.1f\n", d, degree, g0, i0);
        std::printf("%7s %8s %8s | %-16s | %-24s | %s\n", "eta", "gain", "in_gain", "stored cos / rms", "repair in->out  id%", "novel self  max");
        std::vector<Cell> cells;
        for (float eta : {0.0015f, 0.003f})
            for (float gf : {0.5f, 1.f, 2.f})
                for (float igf : {0.5f, 1.f, 2.f}) {
                    const float gain = g0 * gf, ig = i0 * igf;
                    cells.push_back({Fmt("%7.4f %8.4f %8.1f | ", eta, gain, ig),
                                     [=, &ps] { return Run(dim, d, ig, gain, eta, ps); }});
                }
        RunCells(cells);
        return 0;
    }

    if (mode == "scale") {
        const size_t degree = Hopfield(Cfg(dim, d, 1.f, 1.f, 1.f)).Degree();
        std::printf("scale: distance %zu, degree %zu, eta 0.003\n", d, degree);
        std::printf("%8s %8s | %-16s | %-24s | %s\n", "gain", "in_gain", "stored cos / rms", "repair in->out  id%", "novel self  max");
        std::vector<Cell> cells;
        for (float gain : {1.5f, 0.5f, 0.15f, 0.05f, 0.015f, 0.005f})
            for (float ig : {1.f, 5.f, 20.f, 100.f})
                cells.push_back({Fmt("%8.4f %8.1f | ", gain, ig),
                                 [=, &ps] { return Run(dim, d, ig, gain, 0.003f, ps); }});
        RunCells(cells);
        return 0;
    }

    std::printf("table: scaling-rule point at every distance, eta 0.003\n");
    std::printf("%2s %6s %4s | %-16s | %-24s | %s\n", "d", "degree", "P", "stored cos / rms", "repair in->out  id%", "novel self  max");
    std::vector<Cell> cells;
    for (size_t dd = 1; dd <= dmid; ++dd) {
        const size_t degree = Hopfield(Cfg(dim, dd, 1.f, 1.f, 1.f)).Degree();
        for (size_t p : {size_t{1}, size_t{2}, size_t{5}, size_t{10}, size_t{20}, size_t{50}, size_t{100}, size_t{200}}) {
            if (p > ps.size()) break;
            cells.push_back({Fmt("%2zu %6zu %4zu | ", dd, degree, p), [=, &ps] {
                std::vector<Pat> sub(ps.begin(), ps.begin() + p);
                return Run(dim, dd, RuleInputGain(degree), RuleGain(degree), 0.003f, sub);
            }});
        }
    }
    RunCells(cells);
    return 0;
}
