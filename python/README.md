# Hypercube Hopfield 1984

[![Build wheels](https://github.com/dliptak001/HypercubeHopfield1984/actions/workflows/wheels.yml/badge.svg)](https://github.com/dliptak001/HypercubeHopfield1984/actions/workflows/wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/hypercube-hopfield1984)](https://pypi.org/project/hypercube-hopfield1984/)
[![Python](https://img.shields.io/pypi/pyversions/hypercube-hopfield1984)](https://pypi.org/project/hypercube-hopfield1984/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/dliptak001/HypercubeHopfield1984/blob/master/LICENSE)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)

This package is the **Python** surface for HypercubeHopfield1984
(`import hypercube_hopfield1984`).
Full API reference: **[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeHopfield1984/blob/master/docs/Python_SDK.md)**.
C++ integration guide: **[docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeHopfield1984/blob/master/docs/CPP_SDK.md)**.
Project home: **[github.com/dliptak001/HypercubeHopfield1984](https://github.com/dliptak001/HypercubeHopfield1984)**.

HypercubeHopfield1984 is Hopfield's **graded associative memory**, the
model of J. J. Hopfield's 1984 paper, "Neurons with graded response have
collective computational properties like those of two-state neurons"
(Proc. Natl. Acad. Sci. USA 81, 3088–3092), built on a Boolean hypercube.
Every vertex holds a floating-point state and every edge a floating-point
weight, both in [−1, 1]. The caller shows the network a pattern. In
training mode the network adjusts its weights so the pattern becomes a
low point of its energy. In latch mode the pattern plays two roles at
once: it is the starting state of the cube, and it also stays applied to
every vertex for the whole settle as a steady push, Hopfield's input
current. The weights pull the cube toward what they have stored while the
pattern keeps pulling it back toward itself, and the network returns the
state where those two pulls balance.

That balance is the point of this network. Because the pattern never
stops pulling, the output is never a stored pattern looked up from a
fragment; it is the input, pulled part of the way toward what the memory
has seen. **It is a shaper, not a recall.** A stored pattern comes back
nearly as it went in. A damaged one comes back repaired toward its stored
form. An unknown one comes back moved toward the nearest thing the memory
knows, by an amount that says how familiar it is. If you want a clean
retrieval from a cue, use a modern Hopfield network; if you want a signal
bent by experience and passed on, use this one.

There is one class, `Hopfield`, and it stays a float network throughout:
nothing is ever rounded to ±1. It stands beside
[HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield), the
modern (dense, softmax-over-patterns) Hopfield network in the same family,
which does the retrieval this one does not.

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
  <a href="https://github.com/dliptak001/HypercubeHopfield1984/blob/master/docs/Boolean_hypercubes_as_a_neural_substrate.pdf"><em>Boolean Hypercubes as a Neural Substrate</em></a>
  (D.&nbsp;C.&nbsp;Liptak, 2026)
</p>

HypercubeHopfield1984 is an experiment in the **HypercubeAI** project,
our quest to systematically re-implement classical neural architectures
on a Boolean hypercube topology instead of Euclidean grids or random
graphs. The central thesis is "topology-native intelligence": the
hypercube's algebraic structure (vertex-transitive symmetry, Hamming
geometry, bitwise addressing) can serve as a first-class computational
substrate.

- **A topology you don't store**: connectivity is implicit in the vertex
  indices, and the neighborhood is programmable. Two vertices are joined
  when their indices differ in exactly `distance` bits, so stepping to a
  neighbor is one XOR with a `distance`-bit mask. The dimension and that
  one number fix the whole graph, from the hypercube at distance 1 to the
  middle shell at dim/2.
- **Perfect homogeneity**: every vertex has the same degree and the same
  local world, so the dynamics mean the same thing everywhere.
- **Cheap navigation**: each neighbor is a few bit operations on the
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
operating points, is in
[docs/hopfield.md](https://github.com/dliptak001/HypercubeHopfield1984/blob/master/docs/hopfield.md).

---

## Installation

**Preferred:** install a pre-built wheel from PyPI (no compiler).

```bash
pip install hypercube-hopfield1984
```

```python
import hypercube_hopfield1984 as hh
print(hh.__version__)
```

Package name on PyPI: **`hypercube-hopfield1984`**. Import name:
**`hypercube_hopfield1984`**. Main type: **`hh.Hopfield`**.

Wheels target Python 3.10 through 3.14 on common Windows, Linux, and macOS
machines. Runtime dependency: NumPy only.

### From source (full repository)

To compile the extension yourself, clone this **entire** repository (not a
minimal source-only download of the `python/` folder alone; the C++ core
lives next to `python/`). You need Python 3.10+, a C++23 compiler, and
CMake ≥ 3.20.

```bash
git clone https://github.com/dliptak001/HypercubeHopfield1984.git
cd HypercubeHopfield1984/python
pip install .
```

On Windows with CLion's MinGW, put that compiler's `bin` folder (and Ninja)
on your `PATH`, then:

```bash
pip install . --no-build-isolation --force-reinstall --no-deps
```

(Exact CLion paths change with the version.)

---

## Quick start

You bring each pattern as an array of 2ᵏ floats, k ≤ dim. How you get there
(pad an image, reshape a spectrum, invent a layout) is up to you. A shorter
pattern repeats onto every copy of the low-k cube.

```python
import numpy as np
import hypercube_hopfield1984 as hh

net = hh.Hopfield(dim=10, gain=0.03, sweep_budget=64, rest_tolerance=0.01,
                  eta=0.003, input_gain=20.0, distance=5)
N = net.N                                   # 1024

rng = np.random.default_rng(0)
patterns = rng.uniform(-1, 1, (50, N)).astype(np.float32)
for _ in range(50):
    for p in patterns:
        net.train(p)                        # deepen each pattern's well

latch = net.latch(patterns[0])              # (N,) float32: the settled cube
print(net.sweep_count, net.at_rest)
print(net)

net.save("memory.pkl")
loaded = hh.Hopfield.load("memory.pkl")
```

The operating point above (dim 10, distance 5, gain 0.03, input gain 20,
η 0.003) is the measured one: a stored pattern comes back with cosine
0.987 to itself and a pattern with a quarter of its sites randomized is
repaired from cosine 0.75 to 0.85. The rows for dim 11 and 12 are one
rule, gain = 7.56 / degree and input gain = degree / 12.6.

---

## Features

- **One class**: `hypercube_hopfield1984.Hopfield` is the whole product surface
- **`train` / `latch` / `energy`**: exposed one-to-one with the C++ API
- **dim 1 through 31**: N = 2ᵈⁱᵐ vertices; `distance` picks the degree, the capacity lever
- **Graded throughout**: states and weights are floats in [−1, 1]; nothing is rounded to ±1
- **Short exposures tile**: a 2ᵏ pattern repeats onto every copy of the low-k cube
- **Latch record**: `sweep_count` and `at_rest` say how the last settle ended
- **Weights as NumPy**: `net.weights` (settable), row-major N × degree
- **Save / load**: `save` / `load` (pickle: config + weights)
- **NumPy float32**: arrays converted for you; prefer contiguous float32
- **GIL released** during `latch`, `train`, `energy`, and a weights load

---

## Examples

For a first try, paste the [Quick start](#quick-start) after
`pip install hypercube-hopfield1984`. That is self-contained.

The demo script on GitHub under
[`python/examples/`](https://github.com/dliptak001/HypercubeHopfield1984/tree/master/python/examples)
is there to open or download; it is not added to your machine by pip.

| Script | What it is for |
|--------|----------------|
| [shaper.py](https://github.com/dliptak001/HypercubeHopfield1984/blob/master/python/examples/shaper.py) | Store the fifty bundled exemplars, then latch a stored, a damaged, and a novel pattern; the latch energy tells them apart |

```bash
# from a clone of HypercubeHopfield1984, after: pip install hypercube-hopfield1984
python python/examples/shaper.py
```

---

## Documentation

| Doc | Role |
|-----|------|
| **[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeHopfield1984/blob/master/docs/Python_SDK.md)** | Canonical Python API: every method, layout, pickle, limits |
| [docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeHopfield1984/blob/master/docs/CPP_SDK.md) | Native library guide (same product, C++) |
| [Project README](https://github.com/dliptak001/HypercubeHopfield1984#readme) | Product story and the C++ programs from the repo root |
| [docs/hopfield.md](https://github.com/dliptak001/HypercubeHopfield1984/blob/master/docs/hopfield.md) | The network, the energy, the sweep, and the measured operating points |

---

## Ecosystem

- **[HypercubeCascade](https://github.com/dliptak001/HypercubeCascade)**: both frozen stages in series + thin readout.
- **[HypercubeCNN](https://github.com/dliptak001/HypercubeCNN)**: cube-native conv stack with shared kernels.
- **[HypercubeESN](https://github.com/dliptak001/HypercubeESN)**: echo-state / reservoir computing on streams.
- **[HypercubeEtalon](https://github.com/dliptak001/HypercubeEtalon)**: frozen etalon transit + thin readout.
- **[HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield)**: the modern Hopfield network; clean retrieval from a cue.
- **[HypercubeHopfield1984](https://github.com/dliptak001/HypercubeHopfield1984)**: Hopfield's 1984 graded memory; a shaper, not a recall.
- **[HypercubeLCN](https://github.com/dliptak001/HypercubeLCN)**: the locally connected net, every weight trained.
- **[HypercubeWorldModel](https://github.com/dliptak001/HypercubeWorldModel)**: frozen WTF encoder + trained LCN predictor and decoder.
- **[HypercubeWTF](https://github.com/dliptak001/HypercubeWTF)**: frozen reservoir orbit + thin readout.

---

## License

Apache 2.0.

## Definitions

| Symbol / term | Meaning |
|---------------|---------|
| Hopfield | The Python class: train, latch, energy, weights, persistence. |
| dim | Hypercube dimension, 1 through 31. |
| N | Vertex count, N = 2ᵈⁱᵐ. The length of a cube. |
| distance | Hamming distance between neighbors, 1 through max(1, dim // 2). |
| degree | Neighbors per vertex, C(dim, distance). |
| exposure | The caller's pattern: 2ᵏ finite floats, k ≤ dim. |
| latch | The settled cube `latch` returns: the exposure pulled toward what the memory has stored. |
| sweep | One in-place rewrite of every even-popcount vertex in index order, then every odd-popcount vertex. |
| settle | Sweeps repeated until one moves every vertex by at most rest_tolerance, or sweep_budget is spent. |
| rest_tolerance | The largest move at one vertex that still counts as rest. Finite, 0 or more; default 0, bit-for-bit equality. Reaching it ends the settle with `at_rest` True. |
| sweep_budget | The most sweeps one settle will run, 1 or more. A cube still moving when the budget is spent stops there, with `at_rest` False. |
| weights | N · degree float32: row v holds v's weights in mask order, each edge in both endpoints' rows. |
