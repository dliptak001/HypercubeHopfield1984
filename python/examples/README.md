# Python examples

Demo scripts for the `hypercube-hopfield1984` API. They call only the
installed package.

| Script | What it shows |
|--------|----------------|
| [`shaper.py`](shaper.py) | Store the fifty exemplars at the dim 10 operating point, then latch a stored, a damaged, and a novel pattern, with the latch energy as the familiarity readout |

## How to run

These files live in the GitHub repository. They are not part of `pip install`
— use the [Quick start](../README.md#quick-start) if you only want a snippet.

From a clone of HypercubeHopfield1984 (repository root), after installing
the package:

```bash
pip install hypercube-hopfield1984
# or from this tree:  pip install ./python
python python/examples/shaper.py
```

The script reads `data/exemplars.bin` from the clone. Without it the
patterns are synthesized from a fixed seed and the run says so.

## What these are not

- **Not** the C++ programs (`hopfield_example`, `hopfield_recall_benchmark`).
  Those live under [`examples/`](../../examples/).
- **Not** automated tests. Package tests are [`tests/test_basic.py`](../tests/test_basic.py).
- **Not** a benchmark. The numbers it prints match the README's operating
  point table on the bundled exemplars; do not cite them for other data.

## Going further

| Want… | See… |
|-------|------|
| Full Python API | [`docs/Python_SDK.md`](../../docs/Python_SDK.md) |
| C++ product guide | [`docs/CPP_SDK.md`](../../docs/CPP_SDK.md) |
| Package readme | [`README.md`](../README.md) |
| The network, the energy, and the sweep | [`docs/hopfield.md`](../../docs/hopfield.md) |
