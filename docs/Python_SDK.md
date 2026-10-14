# HypercubeHopfield1984 Python SDK

HypercubeHopfield1984 is Hopfield's **graded associative memory**, the
model of his 1984 paper, built on a Boolean hypercube. One class,
`Hopfield`, owns the whole product: train, latch, energy, weights,
persistence.

It is a **shaper, not a recall**: in a latch the exposure is both the
starting state and a steady input current, so the output is the input
pulled part of the way toward what the memory has stored, never a stored
pattern looked up from a cue. For clean retrieval use
[HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield), the
modern Hopfield network in the same family.

C++ core and contracts: **[CPP_SDK.md](CPP_SDK.md)**.
The network, the energy, and the sweep: **[hopfield.md](hopfield.md)**.
PyPI-facing package story: **[python/README.md](../python/README.md)**.
Package version: single source `python/hypercube_hopfield1984/_version.py`
(`hypercube_hopfield1984.__version__` and wheel metadata both read it).

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [What a latch is](#what-a-latch-is)
- [API reference](#api-reference)
- [Input data layout](#input-data-layout)
- [Data types](#data-types)
- [Error handling](#error-handling)
- [Model persistence](#model-persistence)
- [Limitations](#limitations)
- [Dependencies](#dependencies)
- [Definitions](#definitions)

## Installation

### From PyPI (preferred)

Pre-built **wheels**, no compiler required:

```bash
pip install hypercube-hopfield1984
```

Import as `import hypercube_hopfield1984 as hh` (PyPI name
`hypercube-hopfield1984`). Wheels cover Python 3.10 through 3.14 on common
Windows (x64), Linux (x86_64, aarch64), and macOS (x86_64, arm64) builds.
NumPy is the only runtime dependency.

### From source (full repository)

Compile only from a **full clone** of HypercubeHopfield1984. The extension
compiles the C++ core (`Hopfield.cpp`) that sits **outside** the `python/`
package directory; a `python/`-only tree is not enough.

Requirements: Python 3.10+, C++23 compiler (GCC 13+, Clang 17+, MSVC
2022+), CMake 3.20+, scikit-build-core, pybind11, NumPy.

```bash
git clone https://github.com/dliptak001/HypercubeHopfield1984.git
cd HypercubeHopfield1984/python
pip install .
```

On Windows with MinGW (for example the CLion toolchain):

```powershell
pip install scikit-build-core pybind11 numpy
$env:PATH = "C:\path\to\mingw\bin;" + $env:PATH
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_MAKE_PROGRAM = "C:\path\to\ninja.exe"
$env:CC = "C:\path\to\mingw\bin\gcc.exe"
$env:CXX = "C:\path\to\mingw\bin\g++.exe"
pip install . --no-build-isolation
```

### Running tests

From the `python/` directory after install:

```bash
pip install ".[test]"
pytest tests/ -v --import-mode=importlib
```

Or from the repository root: `pytest python/tests/ -v --import-mode=importlib`.
Importlib mode avoids the source tree shadowing the installed `_core`
extension. Use the `pytest` entry point, not `python -m pytest`: the latter
puts the current directory on `sys.path`, and from `python/` the source
package (which has no compiled `_core`) then shadows the installed one.

### Examples

The [Quick start](#quick-start) below is enough after `pip install`. A
longer demo lives in the **git tree** under
[`python/examples/`](../python/examples/README.md); it is **not** part of
the wheel. From a clone, repository root:

```bash
pip install hypercube-hopfield1984   # or: pip install ./python
python python/examples/shaper.py
```

## Quick start

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
print(net.sweep_count, net.at_rest)         # a few sweeps, True
print(net.energy(patterns[0]), net.energy(net.cube, patterns[0]))
```

### The cycle

Every real run is these two calls:

```text
construct Hopfield once
for each pass:
    for each pattern:
        net.train(pattern)          (adds eta * e_v * e_u to every edge, clamps)
net.latch(exposure)                 (settle; returns the latch)
net.energy(net.cube, exposure)      (where the settle ended; optional)
```

`train` never sweeps and `latch` never touches the weights, so the two can
be interleaved in any order. Every array the package returns is a copy, so
`latch` results can be kept.

### Separate instances on threads

One instance is not thread-safe, but the extension releases the GIL during
`latch`, `train`, `energy`, and a weights load, so separate instances on
separate `threading.Thread` workers get real parallelism. The C++
benchmark runs one instance per grid cell on a worker pool; the same
pattern works here with one `Hopfield` per thread.

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
  **C(dim, distance)** neighbors; the graph is implicit in the indices,
  nothing is stored but the weights.
- Every rewrite is a single-vertex descent on Hopfield's energy, so the
  settle stops at a stationary point of it for this exposure as input.
- The exposure never stops pulling: a stored pattern comes back nearly
  as it went in, a damaged one is repaired toward its stored form, a
  novel one is moved toward the nearest thing the memory knows.
- Host packing (images, spectra, sensors into 2ᵏ floats) is the host's
  job. A short exposure repeats onto every copy of the low-k cube.

## API reference

### Constructor `Hopfield(dim, *, gain, sweep_budget, eta, **kwargs)`

All knobs are fixed at construction (the same contract as the C++
`HopfieldConfig`). Four are required and keyword-only after `dim`.

```python
import hypercube_hopfield1984 as hh

net = hh.Hopfield(
    dim=10,               # required; 1-31; N = 2^dim
    gain=0.03,            # required; finite, > 0
    sweep_budget=64,      # required; >= 1
    eta=0.003,            # required; finite, > 0
    rest_tolerance=0.01,  # finite, >= 0; default 0 = bit-for-bit rest
    input_gain=20.0,      # finite, >= 0; default 1; 0 = starting state only
    seed=0,               # starting-weight draw when init_scale > 0
    init_scale=0.0,       # [0, 1]; default 0 = every weight starts at 0
    distance=5,           # [1, max(1, dim // 2)]; default 1
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | `int` | required | Hypercube dimension **[1, 31]**. N = 2^dim. |
| `gain` | `float` | required | Scale on the local field inside tanh. Finite, > 0. Of order 1 / degree; the rule is 7.56 / degree. |
| `sweep_budget` | `int` | required | Most sweeps one settle will run. ≥ 1. |
| `eta` | `float` | required | Step one exposure adds to an edge weight. Finite, > 0. Keep eta · passes · √patterns well under 4. |
| `rest_tolerance` | `float` | `0.0` | Largest move at any vertex for a sweep to count as rest. Finite, ≥ 0. 0.01 is the measured value; 0 means bit-for-bit equality. |
| `input_gain` | `float` | `1.0` | Scale of the input current held on every vertex during a settle. Finite, ≥ 0. Of order degree / 10; the rule is degree / 12.6. |
| `seed` | `int` | `0` | Seeds the starting weights when `init_scale` > 0; unused otherwise. |
| `init_scale` | `float` | `0.0` | Scale of the starting weights, **[0, 1]**. 0 is Hopfield's model. |
| `distance` | `int` | `1` | Hamming distance between neighbors, **[1, max(1, dim // 2)]**. Degree C(dim, distance) is the capacity lever. |

**Sizing note:** the weight matrix is `4 · N · degree` bytes. dim 10 at
distance 5 is 1 MB; dim 12 at distance 6 is 15 MB. A configuration that
does not fit raises `MemoryError` with the byte count.

### Methods

| Method | Role |
|--------|------|
| `latch(exposure)` | Load the exposure as starting state and input current, settle, return the settled cube as a length-N float32 array (a copy). |
| `train(exposure)` | Add `eta * e_v * e_u` to every edge and clamp to [-1, 1]. No sweep; the cube is unchanged. |
| `energy(state, input=None)` | Hopfield's energy of `state` under the current weights with `input` as the current. `input` omitted means the state is its own input. `state` values must lie in [-1, 1]. |
| `save(path)` / `Hopfield.load(path)` | Pickle constructor config + weights. |

### Properties

| Property | Meaning |
|----------|---------|
| `dim`, `N`, `distance`, `degree` | Geometry: N = 2^dim, degree = C(dim, distance) |
| `gain`, `sweep_budget`, `eta`, `rest_tolerance`, `input_gain`, `seed`, `init_scale` | Config mirrors |
| `config` | The constructor arguments as a dict; `Hopfield(**net.config)` rebuilds an untrained twin |
| `num_weights` | N × degree |
| `weights` | All weights as a float32 array, row-major N × degree flattened. **Settable**: exact length, finite values in [-1, 1], the two rows of each edge equal |
| `latched` | True once `latch` has run |
| `cube` | The settled cube from the most recent `latch` (a copy). `RuntimeError` before the first `latch` |
| `sweep_count` | Sweeps the most recent `latch` ran, 1 through `sweep_budget`. `RuntimeError` before the first `latch` |
| `at_rest` | True if the tolerance stopped the last settle, False if the budget did. `RuntimeError` before the first `latch` |

## Input data layout

- **Exposures** are any array-like that ravels to length **2ᵏ, k ≤ dim**,
  every value finite. The low k bits of a vertex index name the value it
  carries, so a short exposure repeats on every copy of the low-k cube.
  Values outside [-1, 1] are accepted by `train` and `latch`; `energy`
  requires its state in [-1, 1], which every settled cube satisfies.
- **Weights** are N rows of degree floats, row v in mask order (masks are
  every dim-bit integer with `distance` bits set, ascending), each edge
  present in both endpoints' rows with the same value. `net.weights`
  returns them flattened; reshape to `(N, degree)` for a matrix view.
- **Host packing** (images, spectra, sensors into 2ᵏ floats) is outside
  this package.

## Data types

| Role | Preferred type | Notes |
|------|----------------|-------|
| Exposures / states / inputs / weights / latches | `float32` | Other dtypes and nested lists are converted via NumPy to contiguous float32 |
| `dim`, `sweep_budget`, `distance`, `seed` | `int` | |
| `energy` result | `float` | A Python float, computed in double |

## Error handling

Python-side checks raise `ValueError` with a short message (bad `dim`,
`sweep_budget`, `distance`, or `seed`; a weights array of the wrong
length). Native `std::invalid_argument`, from the C++ constructor's range
checks and from `latch` / `train` / `energy` / a weights load, maps to
`ValueError`; `cube`, `sweep_count`, and `at_rest` before the first `latch`
raise `RuntimeError`; a weight matrix that does not fit raises
`MemoryError` with the byte count.

Typical mistakes:

- Exposure length not a power of two, or longer than N
- A NaN or infinity in an exposure (rejected even under fast-math)
- A state outside [-1, 1] handed to `energy` (the input may be anything finite)
- `distance` above `dim // 2`: the degrees repeat beyond the middle shell
- Reading `cube` before the first `latch`; check `latched`
- Setting `weights` from another dim or distance: the length or the
  edge symmetry check fails
- A latch that returns the exposure unchanged: weights still near zero,
  or `gain * input_gain` near 1 so the current dominates
- Every settle hitting the budget: `rest_tolerance` too tight for graded
  creep; 0.01 is the measured value

## Model persistence

| Mechanism | What is stored | Latch record? |
|-----------|----------------|---------------|
| `save` / `pickle` | Constructor config + all weights | **No** (the loaded instance is unlatched; the cube, sweep count, and at-rest flag are not saved) |

Pickle version is 1 as of 1.0.0. Newer libraries reject unknown future
versions with an upgrade message.

```python
net.save("memory.pkl")
net2 = hh.Hopfield.load("memory.pkl")   # same ctor knobs + weights; unlatched
```

A pickle captures the whole product: the network reconstructs from the
constructor knobs and the weight array is loaded verbatim. `latch` on the
loaded instance reproduces the original bit for bit, and `train` continues
from the stored weights.

The raw weight array is also yours through `net.weights` (settable), so any
external format, `np.save`, HDF5, a database, works without pickle. Gain,
input gain, sweep budget, and rest tolerance do not touch the weights, so
one trained memory can be loaded into an instance built with different
values of them; only dim and distance must match.

**Security:** `load` uses `pickle.load`. Never load untrusted files.

## Limitations

- One `Hopfield` instance is **not thread-safe** for concurrent calls from
  multiple host threads. Separate instances on separate threads are fine:
  the extension releases the GIL during `latch`, `train`, `energy`, and a
  weights load.
- The settle is sequential by design: the in-place index order is what
  makes every rewrite a descent, so there is no parallel sweep.
- Per-sweep cost is `N · degree` multiply-adds plus a tanh per vertex;
  `train` and `energy` visit each of the `N · degree / 2` edges once.
- Every `weights` read copies `4 · N · degree` bytes; read it once and
  keep the array rather than reading it in a loop.
- There is no built-in noise, corruption, or packing helper; hosts add
  those around the calls (`python/examples/shaper.py` shows the damaged
  and novel cases).
- Native contracts and the C++ surface: **[CPP_SDK.md](CPP_SDK.md)**.

## Dependencies

| Layer | What |
|-------|------|
| Runtime | NumPy |
| Wheel install | No compiler |
| From-source build | Full repo clone, C++23, CMake ≥ 3.20, scikit-build-core, pybind11 |

The C++ core is compiled into the extension itself.

## Definitions

| Symbol / term | Meaning |
|---------------|---------|
| Hopfield | The Python class: train, latch, energy, weights, persistence. |
| dim | Hypercube dimension, 1 through 31. N = 2ᵈⁱᵐ vertices. |
| N | Vertex count, 2ᵈⁱᵐ. The length of a cube. |
| distance | Hamming distance between neighbors, 1 through max(1, dim // 2). 1 is the hypercube itself. |
| degree | Neighbors per vertex, C(dim, distance). |
| exposure, k | The caller's pattern: 2ᵏ finite floats, k from 0 through dim. A short one repeats onto every copy of the low-k cube. |
| cube | The network's state, one float32 per vertex. The latch is the settled cube. |
| gain | Scale on the local field inside tanh. |
| input_gain | Scale of the input current held on every vertex during a settle. |
| eta | The training step per exposure on an edge weight. |
| sweep | One in-place rewrite of every even-popcount vertex in index order, then every odd-popcount vertex. |
| settle | Sweeps repeated until one moves every vertex by at most rest_tolerance, or sweep_budget is spent. |
| rest_tolerance | The largest move at one vertex, the absolute difference between the new and old state, that still counts as rest. Finite, 0 or more; default 0, every written float must equal the one it replaced. A sweep that moves every vertex by at most this ends the settle with `at_rest` True. |
| sweep_budget | The most sweeps one settle will run, 1 or more. A cube still moving by more than `rest_tolerance` when the budget is spent stops there, with `at_rest` False and `sweep_count` equal to the budget. |
| weights | N · degree float32, row-major: row v holds v's weights in mask order, each edge in both endpoints' rows. |
| energy | Hopfield's energy of a cube under an input current. A settle is a descent on it. |
