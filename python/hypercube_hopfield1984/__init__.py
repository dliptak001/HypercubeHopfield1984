"""HypercubeHopfield1984: Hopfield's graded associative memory on a hypercube.

The model of J. J. Hopfield's 1984 paper on a Boolean hypercube: every
vertex holds a float state and every edge a float weight, both in [-1, 1].
``train`` adjusts the weights so a pattern becomes a low point of the
energy; ``latch`` loads a pattern as the starting state, holds it as the
input current, and settles. The result is the input pulled toward what the
memory has stored: a shaper, not a recall.

Quick start::

    import numpy as np
    import hypercube_hopfield1984 as hh

    net = hh.Hopfield(dim=10, gain=0.03, sweep_budget=64, eta=0.003,
                      rest_tolerance=0.01, input_gain=20.0, distance=5)
    for _ in range(50):
        net.train(pattern)          # 2^k float32 values, k <= dim
    latch = net.latch(pattern)      # (N,) float32, N = 2^dim
"""

from __future__ import annotations

import math
import numbers
import operator
import pathlib
import pickle

import numpy as np

from ._core import _Core
from ._version import __version__

__all__ = ["Hopfield", "__version__"]

# Valid hypercube dimensions (matches the C++ constructor's [1, 31] check).
_DIM_MIN = 1
_DIM_MAX = 31


def _to_float32(arr):
    """Flatten to a C-contiguous float32 vector.

    A contiguous float32 array of any shape passes through as a view with
    no copy; anything else is converted in one allocation.
    """
    return np.ascontiguousarray(arr, dtype=np.float32).ravel()


def _as_index(name, value):
    """An integer argument: any integral type (int, numpy ints) but not
    bool, float, or str."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer, got {value!r}") from None


def _as_float(name, value):
    """A real-number argument: int, float, or numpy numbers, not str."""
    if isinstance(value, numbers.Real):
        return float(value)
    raise ValueError(f"{name} must be a number, got {value!r}")


class Hopfield:
    """Hopfield's 1984 graded associative memory on a Boolean hypercube.

    A vertex is an integer from 0 to N - 1, N = 2^dim. Its neighbors are the
    vertices whose index differs in exactly ``distance`` bits, so each vertex
    has C(dim, distance) neighbors and every edge carries one weight read
    from both ends. A settle rewrites each vertex to
    ``tanh(gain * (sum of w * s_u + input_gain * e_v))``, one popcount class
    at a time and in place, until a sweep moves every vertex by at most
    ``rest_tolerance`` or ``sweep_budget`` sweeps have run.

    An exposure is any array of 2^k finite floats with k <= dim; a shorter
    one repeats onto every copy of the low-k cube. Exposures are converted to
    float32 and copied, so any array-like is accepted.

    Parameters
    ----------
    dim : int
        Hypercube dimension, 1 to 31. N = 2^dim vertices.
    gain : float
        Scale on the local field inside tanh. Finite, > 0. Of order
        1 / degree keeps a trained field in tanh's working range.
    sweep_budget : int
        Most sweeps one settle will run. >= 1.
    eta : float
        Step one exposure adds to an edge weight. Finite, > 0.
    rest_tolerance : float
        Largest move at any vertex for a sweep to count as rest. Finite,
        >= 0. Default 0: every written float must equal the old one.
    input_gain : float
        Scale of the input current ``input_gain * e_v`` held on every
        vertex during a settle. Finite, >= 0. Default 1. Of order
        degree / 10 balances the weight sum.
    seed : int
        Seeds the starting weights when ``init_scale`` > 0. Default 0.
    init_scale : float
        Scale of the starting weights, in [0, 1]. Default 0, Hopfield's
        model: every weight starts at 0.
    distance : int
        Hamming distance between neighbors, 1 to max(1, dim // 2).
        Default 1, the hypercube itself.

    Notes
    -----
    This class is **not thread-safe** for concurrent calls on one instance.
    Separate instances on separate threads are independent: the extension
    releases the GIL during ``latch``, ``train``, ``energy``, and a weights
    load.

    The weight matrix is 4 * N * degree bytes. dim 10 at distance 5 is
    1 MB; dim 12 at distance 6 is 15 MB; the top of the dim range does not
    fit in memory and the constructor raises ``MemoryError`` with the size.
    """

    def __init__(
        self,
        dim: int,
        *,
        gain: float,
        sweep_budget: int,
        eta: float,
        rest_tolerance: float = 0.0,
        input_gain: float = 1.0,
        seed: int = 0,
        init_scale: float = 0.0,
        distance: int = 1,
    ):
        dim = _as_index("dim", dim)
        if not (_DIM_MIN <= dim <= _DIM_MAX):
            raise ValueError(f"dim must be in [{_DIM_MIN}, {_DIM_MAX}], got {dim}")
        sweep_budget = _as_index("sweep_budget", sweep_budget)
        if sweep_budget < 1:
            raise ValueError(f"sweep_budget must be >= 1, got {sweep_budget}")
        distance = _as_index("distance", distance)
        if not (1 <= distance <= max(1, dim // 2)):
            raise ValueError(
                f"distance must be in [1, {max(1, dim // 2)}] for dim {dim}, got {distance}"
            )
        seed = _as_index("seed", seed)
        if not (0 <= seed < 2**64):
            raise ValueError(f"seed must be in [0, 2**64), got {seed}")
        self._ctor = {
            "dim": dim,
            "gain": _as_float("gain", gain),
            "sweep_budget": sweep_budget,
            "eta": _as_float("eta", eta),
            "rest_tolerance": _as_float("rest_tolerance", rest_tolerance),
            "input_gain": _as_float("input_gain", input_gain),
            "seed": seed,
            "init_scale": _as_float("init_scale", init_scale),
            "distance": distance,
        }
        try:
            self._core = _Core(**self._ctor)
        except MemoryError:
            n = 1 << dim
            degree = math.comb(dim, distance)
            raise MemoryError(
                f"the weight matrix for dim {dim} at distance {distance} is "
                f"{4 * n * degree:,} bytes (N = {n:,} x degree {degree:,} floats) "
                "and does not fit in memory"
            ) from None

    # ── Latch and train ──

    def latch(self, exposure) -> np.ndarray:
        """Load ``exposure`` as the starting state and the input current,
        settle, and return the settled cube.

        Parameters
        ----------
        exposure : array-like
            2^k finite values, k <= dim; any shape that ravels to that
            length. Converted to float32.

        Returns
        -------
        ndarray
            Length-N float32 cube (a copy). ``cube`` returns the same
            array until the next ``latch``.
        """
        return self._core.latch(_to_float32(exposure))

    def train(self, exposure) -> None:
        """Add ``eta * e_v * e_u`` to every edge and clamp to [-1, 1]. No
        sweep; the cube is unchanged."""
        self._core.train(_to_float32(exposure))

    def energy(self, state, input=None) -> float:
        """Hopfield's energy of ``state`` under the current weights with
        ``input`` applied as the current ``input_gain * input_v``.

        Both follow the exposure rules and need not have the same length.
        ``state`` values must lie in [-1, 1]. With ``input`` omitted the
        state is its own input: ``energy(e)`` is where a settle of ``e``
        starts and ``energy(net.cube, e)`` is where it ended.
        """
        s = _to_float32(state)
        i = s if input is None else _to_float32(input)
        return float(self._core.energy(s, i))

    # ── Latch record ──

    @property
    def latched(self) -> bool:
        """True once ``latch`` has run; ``cube``, ``sweep_count``, and
        ``at_rest`` are valid only then."""
        return bool(self._core.latched)

    @property
    def cube(self) -> np.ndarray:
        """The settled cube from the most recent ``latch`` (a copy).
        Raises ``RuntimeError`` before the first ``latch``."""
        return self._core.cube()

    @property
    def sweep_count(self) -> int:
        """Sweeps the most recent ``latch`` ran, 1 through ``sweep_budget``.
        Raises ``RuntimeError`` before the first ``latch``."""
        return int(self._core.sweep_count)

    @property
    def at_rest(self) -> bool:
        """True if the most recent ``latch`` stopped on ``rest_tolerance``,
        False if the budget stopped it while the cube was still moving.
        Raises ``RuntimeError`` before the first ``latch``."""
        return bool(self._core.at_rest)

    # ── Weights ──

    @property
    def weights(self) -> np.ndarray:
        """Copy of all weights, row-major N x degree, flattened: entry
        ``v * degree + k`` is the edge from v along mask k and equals the
        entry at ``(v ^ mask_k, k)``. Every value is in [-1, 1].
        **Settable**: exact length, finite values in [-1, 1], and the two
        rows of each edge must agree."""
        return self._core.weights()

    @weights.setter
    def weights(self, w) -> None:
        w = _to_float32(w)
        if w.size != self.num_weights:
            raise ValueError(
                f"weights length ({w.size}) must equal num_weights "
                f"({self.num_weights})"
            )
        self._core.load_weights(w)

    # ── Properties ──

    @property
    def dim(self) -> int:
        """Hypercube dimension."""
        return int(self._core.dim)

    @property
    def N(self) -> int:
        """Vertex count, 2^dim."""
        return int(self._core.N)

    @property
    def distance(self) -> int:
        """Hamming distance between neighbors."""
        return int(self._core.distance)

    @property
    def degree(self) -> int:
        """Neighbors per vertex, C(dim, distance)."""
        return int(self._core.degree)

    @property
    def num_weights(self) -> int:
        """N * degree."""
        return int(self._core.num_weights)

    @property
    def gain(self) -> float:
        return float(self._core.gain)

    @property
    def sweep_budget(self) -> int:
        return int(self._core.sweep_budget)

    @property
    def eta(self) -> float:
        return float(self._core.eta)

    @property
    def rest_tolerance(self) -> float:
        return float(self._core.rest_tolerance)

    @property
    def input_gain(self) -> float:
        return float(self._core.input_gain)

    @property
    def seed(self) -> int:
        return int(self._core.seed)

    @property
    def init_scale(self) -> float:
        return float(self._core.init_scale)

    @property
    def config(self) -> dict:
        """The constructor arguments as a dict; ``Hopfield(**net.config)``
        rebuilds an untrained twin."""
        return dict(self._ctor)

    def __repr__(self) -> str:
        return (
            f"Hopfield(dim={self.dim}, N={self.N}, distance={self.distance}, "
            f"degree={self.degree}, gain={self.gain:g}, input_gain={self.input_gain:g}, "
            f"eta={self.eta:g}, sweep_budget={self.sweep_budget}, "
            f"rest_tolerance={self.rest_tolerance:g}, weights={self.num_weights})"
        )

    # ── Persistence ──

    _PERSISTENCE_VERSION = 1

    def __getstate__(self) -> dict:
        """Serialize constructor config + weights. The latch record (cube,
        sweep count, at rest) is **not** saved."""
        return {
            "_version": self._PERSISTENCE_VERSION,
            "ctor": dict(self._ctor),
            "weights": self._core.weights(),
        }

    def __setstate__(self, state: dict) -> None:
        version = state.get("_version", 0)
        if version > self._PERSISTENCE_VERSION:
            raise ValueError(
                f"Model was saved with persistence version {version}, "
                f"but this version only supports up to "
                f"{self._PERSISTENCE_VERSION}. Upgrade hypercube-hopfield1984."
            )
        self.__init__(**dict(state["ctor"]))
        self._core.load_weights(_to_float32(state["weights"]))

    def save(self, path) -> None:
        """Save config + weights to a pickle file (the latch record is not
        saved)."""
        with open(pathlib.Path(path), "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path) -> "Hopfield":
        """Load a model saved by :meth:`save`.

        .. warning::

            Uses ``pickle.load``. Never load untrusted files.
        """
        with open(pathlib.Path(path), "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected Hopfield, got {type(obj).__name__}")
        return obj
