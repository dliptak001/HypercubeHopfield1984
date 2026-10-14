# Hypercube Hopfield 1984

[![Build wheels](https://github.com/dliptak001/HypercubeHopfield1984/actions/workflows/wheels.yml/badge.svg)](https://github.com/dliptak001/HypercubeHopfield1984/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/hypercube-hopfield1984)](https://pypi.org/project/hypercube-hopfield1984/)
[![Python](https://img.shields.io/pypi/pyversions/hypercube-hopfield1984)](https://pypi.org/project/hypercube-hopfield1984/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)

HypercubeHopfield1984 is Hopfield's **graded associative memory** — the
model of J. J. Hopfield's 1984 paper, "Neurons with graded response have
collective computational properties like those of two-state neurons"
(Proc. Natl. Acad. Sci. USA 81, 3088–3092) — built on a Boolean
hypercube. Every vertex holds a floating-point state and every edge a
floating-point weight; the weights are clamped to [−1, 1] and a settled
state lies in (−1, 1), the range of tanh. The caller shows the network a
pattern. In training mode the network adjusts its weights so the pattern
becomes a low point of its energy. In latch mode the pattern plays two
roles at once: it is the starting state of the cube, and it also stays
applied to every vertex for the whole settle as a steady push — Hopfield's
input current. The weights pull the cube toward what they have stored
while the pattern keeps pulling it back toward itself, and the network
returns the state where those two pulls balance.

That balance is the point of this network. Because the pattern never
stops pulling, the output is never a stored pattern looked up from a
fragment; it is the input, pulled part of the way toward what the memory
has seen. **It is a shaper, not a
recall.** A stored pattern comes back nearly as it went in. A damaged one
comes back repaired toward its stored form. An unknown one comes back
moved toward the nearest thing the memory knows, by an amount that says
how familiar it is. If you want a clean retrieval from a cue, use a
modern Hopfield network; if you want a signal bent by experience and
passed on, use this one.

There is one class, `Hopfield`, and it stays a float network throughout:
nothing is ever rounded to ±1. It stands beside
[HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield), the
modern (dense, softmax-over-patterns) Hopfield network in the same family,
which does the retrieval this one does not; this is the classical graded
model.

---

<p align="center">
  <strong>HypercubeAI ecosystem</strong><br/>
</p>

<p align="center">
  <a href="https://github.com/dliptak001/HypercubeCascade"><strong>HypercubeCascade</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeCNN"><strong>HypercubeCNN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeESN"><strong>HypercubeESN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeEtalon"><strong>HypercubeEtalon</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeHopfield"><strong>HypercubeHopfield</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeHopfield1984"><strong>HypercubeHopfield1984</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeLCN"><strong>HypercubeLCN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeWorldModel"><strong>HypercubeWorldModel</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeWTF"><strong>HypercubeWTF</strong></a>
</p>

<p align="center">
  📄 Foundational paper:
  <a href="docs/Boolean_hypercubes_as_a_neural_substrate.pdf"><em>Boolean Hypercubes as a Neural Substrate</em></a>
  (D.&nbsp;C.&nbsp;Liptak, 2026)
</p>

HypercubeHopfield1984 is an experiment in the **HypercubeAI** project —
our quest to systematically re-implement classical neural architectures
on a Boolean hypercube topology instead of Euclidean grids or random
graphs. The central thesis is "topology-native intelligence": the
hypercube's algebraic structure (vertex-transitive symmetry, Hamming
geometry, bitwise addressing) can serve as a first-class computational
substrate.

- **A topology you don't store** — connectivity is implicit in the vertex
  indices, and the neighborhood is programmable: two vertices are joined
  when their indices differ in exactly `distance` bits, so stepping to a
  neighbor is one XOR with a `distance`-bit mask. The dimension and that
  one number fix the whole graph, from the hypercube at distance 1 to the
  middle shell at dim/2.
- **Perfect homogeneity** — every vertex has the same degree and the same
  local world, so the dynamics mean the same thing everywhere.
- **Cheap navigation** — each neighbor is a few bit operations on the
  vertex index, not a pointer chase through a stored edge list.

Each product in the family is a different architecture on that same
foundation.

---

## The network

A vertex is an integer from 0 to 2ᵈⁱᵐ − 1. Its neighbors are the vertices
whose index differs from it in exactly `distance` bits, so at distance 1
the graph is the hypercube itself and at larger distances every vertex has
C(dim, distance) neighbors. Each edge carries one weight, read from both
ends.

A settle rewrites each vertex to tanh(gain · hᵥ), where the local field hᵥ
is the weighted sum of its neighbors' states plus the input current, the
input gain times the pattern's value at that vertex. The vertices are
rewritten one popcount class at a time, in place, so every rewrite is a
descent on Hopfield's energy and the settle stops at a stationary point of
it. Training adds η · eᵥ · eᵤ to every edge and clamps it to [−1, 1].
The full description, with the energy, the sweep, and the measured
operating points, is in [docs/hopfield.md](docs/hopfield.md).

## Using it

```cpp
#include "Hopfield.h"

HopfieldConfig cfg{
    .dim = 10, .gain = 0.03f, .sweep_budget = 64, .rest_tolerance = 0.01f,
    .eta = 0.003f, .input_gain = 20.f, .distance = 5,
};
Hopfield net(cfg);

std::vector<float> pattern(1024);       // 2^k values, k <= dim, in [-1, 1]
// ... fill it ...
for (int pass = 0; pass < 50; ++pass)
    net.Train(pattern);                 // adjust the weights toward it

std::span<const float> latch = net.Latch(pattern);   // settle; length 2^dim
double e = net.Energy(net.Cube(), pattern);          // energy at the latch

const std::vector<float>& w = net.Weights();         // N x degree, row-major;
// ... write w out; the class reads and writes no files ...
net.LoadWeights(stored);                             // a vector read back from disk,
                                                     // same dim and distance
```

`Latch` and `Train` accept any exposure of length 2ᵏ with k ≤ dim; a
shorter pattern is repeated onto every copy of the low-k cube. The class
never keeps a pointer into a caller's buffer, and the spans it returns are
overwritten in place by later calls — copy them out if you need them to
last. It reads and writes no files. The integration guide, with the
knobs, the error contract, and persistence, is
[docs/CPP_SDK.md](docs/CPP_SDK.md).

### From Python

```bash
pip install hypercube-hopfield1984
```

```python
import hypercube_hopfield1984 as hh

net = hh.Hopfield(dim=10, gain=0.03, sweep_budget=64, rest_tolerance=0.01,
                  eta=0.003, input_gain=20.0, distance=5)
for _ in range(50):
    net.train(pattern)                  # 2^k float32 values, k <= dim
latch = net.latch(pattern)              # (N,) float32: the settled cube
```

The same class one-to-one: `train`, `latch`, `energy`, `weights`
(settable), `sweep_count`, `at_rest`, and pickle. Wheels for Python 3.10
through 3.14 on Windows, Linux, and macOS; the package source is under
[python/](python/README.md) and the API reference is
[docs/Python_SDK.md](docs/Python_SDK.md).

## Operating points

Measured with 50 stored patterns and 50 training passes. "Stored" is a
stored pattern latched as itself, as cosine and RMS error; "repair" is the
cosine to the pattern after an exposure with a quarter of its sites
randomized settles, from 0.75 before.

| dim | distance | degree | η | gain | input gain | stored cos / RMS | repair |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 5 | 252 | 0.003 | 0.030 | 20 | 0.987 / 0.14 | 0.85 |
| 11 | 5 | 462 | 0.003 | 0.0164 | 36.7 | 0.991 / 0.14 | 0.85 |
| 12 | 6 | 924 | 0.003 | 0.0082 | 73.3 | 0.992 / 0.14 | 0.85 |

The rows are one rule: gain = 7.56 / degree, input gain = degree / 12.6.
How the numbers were measured, and why they come out the same at every
dimension, is in [docs/hopfield.md](docs/hopfield.md).

## Building

CMake 3.20 or later and a C++23 compiler. Release builds use `-O3 -ffast-math`
and `-march` from the `HYPERCUBE_ARCH` cache variable (`native` by default,
`x86-64-v2` for portable binaries, `none` to skip it). From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build build
```

```bash
ctest --test-dir build
```

Targets: `HypercubeHopfield1984Core` (the library; link it from your own
CMake project with `add_subdirectory` and `target_link_libraries`),
`hopfield_example`, `hopfield_smoketest` (registered with CTest), and
`hopfield_recall_benchmark` (the measurement tool behind the operating
points; `tune|table|scale dim [P] [d] [exemplars.bin]`). Both read
`data/exemplars.bin`, fifty patterns of 1024 float32 values bundled with
the repository; the path is compiled in, so they run from any directory.
The example takes another file as its one argument and the benchmark as its
fifth. When a file cannot be opened the example synthesizes its patterns
from a fixed seed and the benchmark says how many it read.

## Layout

```
Hopfield.h, Hopfield.cpp       the class
Rng.h                          portable uniform draws for the optional random start
docs/hopfield.md               the full description
docs/CPP_SDK.md                C++ integration guide
docs/Python_SDK.md             Python API reference
data/exemplars.bin             the fifty exemplars behind the operating points
examples/example.cpp           training on the exemplars at the operating point
examples/recall_benchmark.cpp  recall, repair, and novelty across parameter grids
test/smoketest.cpp             the API contract, one printed line per check
python/                        the pip package: bindings, tests, example
.github/workflows/wheels.yml   wheel build and PyPI publish on a v* tag
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Symbols

| Symbol | Meaning |
| --- | --- |
| dim | Cube dimension, 1 through 31. |
| N | Vertex count, 2ᵈⁱᵐ. The length of a cube. |
| v, u | Vertices: integer indices from 0 through N − 1. u is a neighbor of v. |
| distance | Hamming distance between neighbors, 1 through max(1, dim/2). |
| mask | An index with exactly `distance` bits set. A neighbor of v is v XOR a mask. |
| degree | Neighbors per vertex, C(dim, distance). |
| popcount | The number of 1-bits in an index. A sweep rewrites the even-popcount vertices, then the odd. |
| exposure, pattern, k | The caller's pattern: 2ᵏ finite floats, k ≤ dim. A short one repeats onto every copy of the low-k cube. |
| cube | The network's state: one float per vertex. |
| latch | The settled cube `Latch` returns: the exposure pulled toward what the memory has stored. |
| hᵥ | Local field at vertex v: the weighted sum over its neighbors plus the input current. |
| eᵥ, eᵤ | The pattern's value at vertex v, or u. |
| gain | Scale on hᵥ inside tanh. |
| input gain | Scale of the input current, input gain · eᵥ. |
| cosine, RMS | The scores: cosine similarity between a latch and a pattern, and the root mean square of their difference. |
| sweep | One in-place rewrite of every even-popcount vertex in index order, then every odd-popcount vertex. |
| settle | Sweeps repeated until one moves every vertex by at most the rest tolerance, or the sweep budget is spent. |
| rest tolerance | The largest move at one vertex that still counts as rest, 0 or more. Zero means bit-for-bit equality. Reaching it ends the settle at rest. |
| sweep budget | The most sweeps one settle will run, 1 or more. A cube still moving when the budget is spent stops there, not at rest. |
| η | The training step per pattern on an edge weight. |
| P | The number of stored patterns. |
