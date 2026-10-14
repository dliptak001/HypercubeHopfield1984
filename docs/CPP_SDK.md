# HypercubeHopfield1984 C++ SDK

HypercubeHopfield1984 is Hopfield's **graded associative memory**, the
model of his 1984 paper, built on a Boolean hypercube. One class owns the
whole product: `Hopfield` trains, latches, and reports the energy. It lives
in one header and one source at the repository root, written in plain C++23
with no dependencies beyond the standard library.

It is a **shaper, not a recall**: in a latch the exposure is both the
starting state and a steady input current, so the output is the input
pulled part of the way toward what the memory has stored, never a stored
pattern looked up from a cue. For clean retrieval use
[HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield), the
modern Hopfield network in the same family.

The same product from Python: **[Python_SDK.md](Python_SDK.md)**.
The network, the energy, and the sweep: **[hopfield.md](hopfield.md)**.
Worked programs: the two sources under **[examples/](../examples/)**.

## Contents

- [Build and link](#build-and-link)
- [Quick start](#quick-start)
- [What a latch is](#what-a-latch-is)
- [API reference](#api-reference)
- [Input data layout](#input-data-layout)
- [Error handling](#error-handling)
- [Model persistence](#model-persistence)
- [Limitations](#limitations)
- [Dependencies](#dependencies)
- [Definitions](#definitions)

## Build and link

Requirements: a **C++23** compiler (GCC 13+, Clang 17+, MSVC 2022+) and
**CMake ≥ 3.20**. The library target is **`HypercubeHopfield1984Core`**:
`Hopfield.cpp` as a static library, exporting the repo root as a public
include directory. `Rng.h` is a header-only companion the library uses for
its optional random start; the examples use it for patterns and noise.

From your own project:

```cmake
add_subdirectory(path/to/HypercubeHopfield1984)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE HypercubeHopfield1984Core)
```

```cpp
#include "Hopfield.h"
```

The example executables build alongside; ignore them or exclude them from
your default target.

Building this repo directly (CLion: open, reload CMake, build, or any
shell with the toolchain available):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build build
```

```bash
ctest --test-dir build
```

Use **Release** for real runs. Release builds compile with
`-O3 -ffast-math` plus `-march` per the `HYPERCUBE_ARCH` cache variable
(the same knob as the Python wheel): `native` by default for local builds,
`x86-64-v2` for portable release binaries, `none` to skip `-march` (ARM,
MSVC). Fast-math in Release is the family convention; the library's finite
checks are written on the exponent bits so they hold under it.

### Example binaries

| Binary | Role |
|--------|------|
| `hopfield_example` | Trains on the fifty bundled exemplars at the dim 10 operating point; prints one exemplar's energy and sweeps per pass |
| `hopfield_recall_benchmark` | Stored, repair, and novelty scores across parameter grids; the tool behind the README's operating-point table |
| `hopfield_smoketest` | The API contract, one printed line per check; registered with CTest |

Both examples default to `data/exemplars.bin` (fifty patterns of 1024
float32, bundled); the path is compiled in from the source tree, and a
command-line argument overrides it.

## Quick start

A complete toy run: store one random pattern on a dim 6 cube, then latch
the pattern itself, a damaged copy, and a stranger. Verified against the
library: it compiles as shown and prints

```text
stored:  cosine 0.999, 3 sweeps, energy 4.30 -> -31.33
damaged: cosine 0.711 -> 0.756
novel:   cosine to itself 0.999
```

With one pattern stored and gain · input gain at 0.5, the stranger passes
through nearly unchanged; the README's operating point, with fifty
patterns stored, is where novelty shows as a measured move.

```cpp
#include "Hopfield.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <span>
#include <vector>

static double Cosine(std::span<const float> a, std::span<const float> b)
{
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i];
    }
    return ab / std::sqrt(aa * bb);
}

int main()
{
    HopfieldConfig cfg{
        .dim = 6, .gain = 0.1f, .sweep_budget = 64, .rest_tolerance = 0.001f,
        .eta = 0.01f, .input_gain = 5.f, .distance = 3,
    };
    Hopfield net(cfg);
    const size_t N = net.N();                 // 64

    std::mt19937_64 rng(1);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<float> pattern(N);
    for (auto& x : pattern) x = uni(rng);

    for (int pass = 0; pass < 20; ++pass)
        net.Train(pattern);                   // deepen the pattern's well

    const double before = net.Energy(pattern);
    std::span<const float> latch = net.Latch(pattern);
    std::printf("stored:  cosine %.3f, %zu sweeps, energy %.2f -> %.2f\n",
                Cosine(latch, pattern), net.SweepCount(),
                before, net.Energy(net.Cube(), pattern));

    std::vector<float> damaged = pattern;     // a quarter of the sites randomized
    for (size_t v = 0; v < N; v += 4) damaged[v] = uni(rng);
    const double in = Cosine(damaged, pattern);
    std::printf("damaged: cosine %.3f -> %.3f\n",
                in, Cosine(net.Latch(damaged), pattern));

    std::vector<float> novel(N);
    for (auto& x : novel) x = uni(rng);
    std::printf("novel:   cosine to itself %.3f\n",
                Cosine(net.Latch(novel), novel));
    return 0;
}
```

### The cycle

Every real run is these two calls:

```text
fill HopfieldConfig; construct Hopfield once
for each pass:
    for each pattern:
        Train                     (adds eta * e_v * e_u to every edge, clamps)
Latch(exposure)                   (settle; the returned span is the latch)
Energy(Cube(), exposure)          (where the settle ended; optional)
```

`Train` never sweeps and `Latch` never touches the weights, so the two can
be interleaved in any order. The latch is valid until the next `Latch`.

## What a latch is

```text
exposure  (2^k floats, tiled onto the cube)
    │
    ▼
 cube := exposure            (starting state)
 current := input_gain * e_v (held on every vertex for the whole settle)
    │
    ▼  repeat up to sweep_budget times:
 even-popcount vertices, in index order, in place:
     s_v := tanh(gain * (sum over neighbors of w * s_u + current_v))
 then odd-popcount vertices the same way
 stop when no vertex moved by more than rest_tolerance
    │
    ▼
 settled cube  =  the latch (length N)
```

- **N = 2ᵈⁱᵐ** vertices, dim 1 through 31. Each vertex has
  **C(dim, distance)** neighbors; the graph is implicit in the indices
  (a neighbor is one XOR away), nothing is stored but the weights.
- Every rewrite is a single-vertex descent on Hopfield's energy, so the
  settle stops at a stationary point of E for this exposure as input.
- The exposure never stops pulling: a stored pattern comes back nearly
  as it went in, a damaged one is repaired toward its stored form, a
  novel one is moved toward the nearest thing the memory knows.
- Host packing (images, spectra, sensors into 2ᵏ floats) is the host's
  job. A short exposure repeats onto every copy of the low-k cube.

## API reference

Authoritative signatures and contracts live in **`Hopfield.h`**, fully
doc-commented. This section is the host-oriented map.

### Configuration

Everything is fixed at construction; the constructor validates and throws
`std::invalid_argument` on violations. Four fields have no valid default
and must be set.

```cpp
struct HopfieldConfig {
    size_t dim = 0;              // required; [1, 31]; N = 1 << dim
    float gain = 0.f;            // required; finite, > 0
    size_t sweep_budget = 0;     // required; >= 1
    float rest_tolerance = 0.f;  // finite, >= 0; 0 = bit-for-bit rest
    float eta = 0.f;             // required; finite, > 0
    float input_gain = 1.f;      // finite, >= 0; 0 = starting state only
    uint64_t seed = 0;           // starting-weight draw when init_scale > 0
    float init_scale = 0.f;      // [0, 1]; 0 = every weight starts at 0
    size_t distance = 1;         // [1, max(1, dim / 2)]
};
```

| Knob | Guidance |
|------|----------|
| `dim`, `distance` | Degree is the capacity lever: more neighbors per vertex, more edges per pattern. dim 10 at distance 5 (degree 252) is the measured operating point |
| `gain` | Of order 1 / degree keeps a trained field in tanh's working range; the rule is 7.56 / degree |
| `input_gain` | Of order degree / 10 balances the weight sum; the rule is degree / 12.6. gain · input gain near 0.6 |
| `eta` | 0.003 at 50 passes over 50 patterns. Keep eta · passes · √patterns well under 4 so weights stay off the clamp |
| `sweep_budget`, `rest_tolerance` | 64 and 0.01: a trained settle takes 2 to 3 sweeps; the budget is the stop for a cube still moving |
| `init_scale`, `seed` | Leave at 0 for Hopfield's model. Above 0 gives a reproducible random start |

The README's operating-point table and [hopfield.md](hopfield.md),
"Operating points", say how those were measured.

**Sizing note:** the weight matrix is `4 · N · degree` bytes. dim 10 at
distance 5 is 1 MB; dim 12 at distance 6 is 15 MB; at the top of the dim
range allocation itself is the limit, and the constructor's failure mode
there is `std::bad_alloc`.

### Hopfield

```cpp
Hopfield net(cfg);                            // validates, allocates once

net.Train(std::span<const float>{e});         // 2^k values; no sweep
std::span<const float> latch = net.Latch(e);  // settle; length N; valid
                                              // until the next Latch
net.Cube();          // the same span; throws std::logic_error before Latch
net.SweepCount();    // sweeps the last Latch ran, 1..sweep_budget
net.AtRest();        // tolerance stopped it (true) or the budget (false)
net.Latched();       // whether a Latch has run

net.Energy(state, input);   // E of state with input as the current
net.Energy(exposure);       // Energy(exposure, exposure): a settle's start

net.Dim();  net.N();  net.Distance();  net.Degree();  net.Config();

const std::vector<float>& w = net.Weights();  // live reference, N x degree
net.LoadWeights(std::span<const float>{w2});  // exact length, symmetric
```

`Hopfield` is a plain value type: copy it to fork a memory, move it to hand
it off. A moved-from instance may only be assigned to or destroyed.

## Input data layout

- **Exposures** have length **2ᵏ, k ≤ dim**, every value finite. The low
  k bits of a vertex index name the value it carries, so a short exposure
  repeats on every copy of the low-k cube. Values outside [−1, 1] are
  accepted by `Train` and `Latch`; `Energy` requires its state in [−1, 1],
  which every settled cube satisfies.
- **Weights** are N rows of degree floats, row v in mask order (masks are
  every dim-bit integer with `distance` bits set, ascending), each edge
  present in both endpoints' rows with the same value.
- The class never keeps a pointer into a caller's buffer; a temporary is a
  safe argument. A view of the own cube is a valid exposure for `Latch`; a
  view of `Weights()` is not an exposure for `Train`.
- Everything is `float`; sizes and indices are `size_t`.

## Error handling

The constructor validates every config range and throws
`std::invalid_argument`; so do `Latch`, `Train`, and `Energy` on a bad
exposure, and `LoadWeights` on a bad blob. `Cube`, `SweepCount`, and
`AtRest` throw `std::logic_error` before the first `Latch`. Oversized
configs throw `std::bad_alloc` from allocation.

Typical mistakes:

| Symptom / assumption | Fix |
|----------------------|-----|
| Construct throws on a required field | `dim`, `gain`, `sweep_budget`, and `eta` have no valid default; set all four |
| Construct throws on distance | Valid range is [1, max(1, dim / 2)]; above the middle shell the degrees repeat |
| Throw on `Latch` or `Train` | Length must be a power of two in [1, N] and every value finite |
| Throw on `Energy` | The state must lie in [−1, 1]; the input may be anything finite |
| `Cube` throws | No `Latch` has run yet; check `Latched()` |
| Latch returns the exposure unchanged | Weights are still near zero, or gain · input gain is near 1 so the current dominates; train more or lower `input_gain` |
| Latch saturates at ±1 | gain · input gain is well above 1.5; lower one of them |
| Every settle hits the budget | `rest_tolerance` is too tight for graded creep; 0.01 is the measured value |
| `LoadWeights` rejected | Exact length N · degree, every value finite in [−1, 1], and the two rows of each edge equal; a blob from another dim or distance fails here |

## Model persistence

The library has no file format of its own: a memory is fully described by
its configuration plus the weight vector.

| Mechanism | What is stored | Latch record? |
|-----------|----------------|---------------|
| `net.Weights()` / `net.LoadWeights(...)` | All weights, N × degree, verbatim | **No** (the cube, sweep count, and at-rest flag are left as they were) |

Write the vector with any format you like; on load, reconstruct a
`Hopfield` with the **same dim and distance** (they decide the weight
count and layout) and `LoadWeights` the exact-length array. Gain, input
gain, sweep budget, and rest tolerance do not touch the weights, so one
trained memory can be loaded under any of them. `Latch` on the reloaded
instance reproduces the original bit for bit. The Python surface pickles
the same thing (config + weights).

## Limitations

- One `Hopfield` is **not thread-safe** for concurrent calls. Separate
  instances on separate threads are independent; `recall_benchmark.cpp`
  runs one per grid cell on a worker pool.
- Per-sweep cost is `N · degree` multiply-adds plus a tanh per vertex;
  `Train` and `Energy` visit each of the `N · degree / 2` edges once.
- Memory: `4 · N · degree` bytes of weights plus two N-float cubes,
  allocated once in the constructor and never reallocated.
- The settle is sequential by design: the in-place index order is what
  makes every rewrite a descent, so there is no parallel sweep.
- There is no built-in noise, corruption, or packing helper; the examples
  add those around the calls.

## Dependencies

| Layer | What |
|-------|------|
| Library | C++ standard library only |
| Build | C++23 compiler, CMake ≥ 3.20 |
| Examples | Same, plus the bundled `data/exemplars.bin` |

There are no third-party dependencies.

## Definitions

| Symbol / term | Meaning |
|---------------|---------|
| Hopfield | The one class: construct, Train, Latch, Energy, Weights, LoadWeights. |
| HopfieldConfig | The construction struct. Every field is fixed when the class is built. |
| dim | Hypercube dimension, 1 through 31. N = 2ᵈⁱᵐ vertices. |
| N | Vertex count, 2ᵈⁱᵐ. The length of a cube. |
| distance | Hamming distance between neighbors, 1 through max(1, dim / 2). 1 is the hypercube itself. |
| degree | Neighbors per vertex, C(dim, distance). |
| exposure, k | The caller's pattern: 2ᵏ finite floats, k from 0 through dim. A short one repeats onto every copy of the low-k cube. |
| cube | The network's state, one float per vertex. The latch is the settled cube. |
| eᵥ, sᵥ | The exposure value carried by vertex v; the state of v. |
| hᵥ | Local field at v: Σ w · sᵤ over the neighbors, plus the input current input gain · eᵥ. |
| gain | Scale on hᵥ inside tanh. |
| input gain | Scale of the input current held on every vertex during a settle. |
| η (eta) | The training step per exposure on an edge weight. |
| sweep | One in-place rewrite of every even-popcount vertex in index order, then every odd-popcount vertex. |
| settle | Sweeps repeated until one moves every vertex by at most the rest tolerance, or the sweep budget is spent. |
| rest tolerance | The largest move at one vertex, the absolute difference between the new and old state, that still counts as rest. Finite, 0 or more. Zero means every written float must equal the one it replaced. A sweep that moves every vertex by at most this ends the settle with AtRest true. |
| sweep budget | The most sweeps one settle will run, 1 or more. A cube still moving by more than the rest tolerance when the budget is spent stops there, with AtRest false and SweepCount equal to the budget. |
| weights | N · degree floats, row-major: row v holds v's weights in mask order, each edge in both endpoints' rows. |
| E | Hopfield's energy of a cube under an input current. A settle is a descent on it. |
