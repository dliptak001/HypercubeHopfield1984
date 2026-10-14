"""Contract tests for the hypercube_hopfield1984 package.

A port of test/smoketest.cpp: construction and argument checks, exposure
rules and tiling, latch and train contracts, determinism, the weights
round-trip, energy descent at every distance, pickle, and an independent
NumPy reference for one sweep and for the energy. Small cubes keep every
test fast; the goal is contract coverage, not benchmark scores.
"""

import math
import pickle

import numpy as np
import pytest

import hypercube_hopfield1984 as hh


def make_net(dim=6, distance=1, **kw):
    args = dict(dim=dim, gain=0.1, sweep_budget=32, rest_tolerance=0.001,
                eta=0.01, input_gain=5.0, seed=3, init_scale=0.0, distance=distance)
    args.update(kw)
    return hh.Hopfield(**args)


def random_pattern(n, rng):
    return rng.uniform(-1.0, 1.0, n).astype(np.float32)


# ── Construction / validation ──

def test_version():
    assert isinstance(hh.__version__, str) and hh.__version__
    assert hh.__version__ == hh._core.__version__


def test_sizes_and_accessors():
    net = make_net(dim=6, distance=2)
    assert net.dim == 6 and net.N == 64 and net.distance == 2 and net.degree == 15
    assert net.num_weights == 64 * 15
    assert net.weights.shape == (64 * 15,) and net.weights.dtype == np.float32
    assert not np.any(net.weights)
    assert net.gain == pytest.approx(0.1) and net.sweep_budget == 32
    assert net.eta == pytest.approx(0.01) and net.input_gain == pytest.approx(5.0)
    assert net.rest_tolerance == pytest.approx(0.001)
    assert net.seed == 3 and net.init_scale == 0.0
    assert net.config["dim"] == 6 and net.config["distance"] == 2


def test_dim_bounds():
    with pytest.raises(ValueError):
        make_net(dim=0)
    with pytest.raises(ValueError):
        make_net(dim=32)
    assert make_net(dim=1, distance=1).N == 2


def test_bad_config_throws():
    for kw in (
        dict(gain=0.0), dict(gain=float("inf")), dict(gain=float("nan")),
        dict(sweep_budget=0), dict(rest_tolerance=-1.0), dict(rest_tolerance=float("nan")),
        dict(eta=0.0), dict(input_gain=-0.5), dict(init_scale=1.5), dict(init_scale=-0.1),
        dict(distance=0), dict(distance=4), dict(seed=-1),
    ):
        with pytest.raises(ValueError):
            make_net(**kw)
    make_net(input_gain=0.0)  # allowed


@pytest.mark.parametrize("kind", [int, np.int64, np.uint8, np.int32])
def test_integer_arguments_accept_any_integral_type(kind):
    net = hh.Hopfield(dim=kind(6), gain=0.1, sweep_budget=kind(8), eta=0.01,
                      seed=kind(3), distance=kind(2))
    assert net.dim == 6 and net.sweep_budget == 8 and net.seed == 3 and net.distance == 2
    assert type(net.config["dim"]) is int


@pytest.mark.parametrize("kw", [
    dict(dim=True), dict(dim=6.0), dict(dim="6"),
    dict(sweep_budget=8.0), dict(distance=False), dict(seed=2**64), dict(seed=-1),
    dict(gain="0.1"), dict(eta=None),
])
def test_argument_types_rejected_with_value_error(kw):
    with pytest.raises(ValueError):
        make_net(**kw)


def test_numeric_knobs_accept_numpy_floats():
    net = make_net(gain=np.float32(0.1), eta=np.float64(0.01), input_gain=np.int64(5))
    assert net.gain == pytest.approx(0.1) and net.input_gain == 5.0


def test_seed_full_uint64_range():
    assert make_net(seed=2**64 - 1).seed == 2**64 - 1


def test_contiguous_float32_input_is_not_copied():
    from hypercube_hopfield1984 import _to_float32
    a = np.zeros((8, 8), dtype=np.float32)
    assert np.shares_memory(_to_float32(a), a)
    b = np.zeros(64, dtype=np.float64)
    assert not np.shares_memory(_to_float32(b), b)
    assert _to_float32([1, 2]).dtype == np.float32
    assert _to_float32(np.float32(1.5)).shape == (1,)


def test_degree_is_binomial():
    for dim in (6, 10):
        for d in range(1, dim // 2 + 1):
            assert make_net(dim=dim, distance=d).degree == math.comb(dim, d)


def test_latch_record_before_latch():
    net = make_net()
    assert not net.latched
    with pytest.raises(RuntimeError):
        net.cube
    with pytest.raises(RuntimeError):
        net.sweep_count
    with pytest.raises(RuntimeError):
        net.at_rest


# ── init_scale ──

def test_init_scale_seed_and_symmetry():
    a = make_net(dim=6, distance=3, init_scale=0.5, seed=42)
    b = make_net(dim=6, distance=3, init_scale=0.5, seed=42)
    c = make_net(dim=6, distance=3, init_scale=0.5, seed=43)
    np.testing.assert_array_equal(a.weights, b.weights)
    assert not np.array_equal(a.weights, c.weights)
    assert np.all(np.abs(a.weights) <= 0.5)
    fresh = make_net(dim=6, distance=3)
    fresh.weights = a.weights  # symmetric across every edge, else ValueError
    np.testing.assert_array_equal(fresh.weights, a.weights)


# ── Exposure rules ──

def test_exposure_rules():
    net = make_net()
    ok = np.full(64, 0.25, dtype=np.float32)
    assert net.latch(ok).shape == (64,)
    assert net.latch(ok[:1]).shape == (64,)
    assert net.latch(ok[:8]).shape == (64,)
    assert net.latch(ok.reshape(8, 8)).shape == (64,)
    with pytest.raises(ValueError):
        net.latch(ok[:0])
    with pytest.raises(ValueError):
        net.latch(ok[:3])
    with pytest.raises(ValueError):
        net.latch(np.zeros(128, dtype=np.float32))
    nf = ok.copy(); nf[5] = np.inf
    with pytest.raises(ValueError):
        net.latch(nf)
    with pytest.raises(ValueError):
        net.train(nf)
    wide = ok.copy(); wide[0] = 2.0
    net.latch(wide)                     # outside [-1, 1] allowed in latch
    with pytest.raises(ValueError):
        net.energy(wide)                # not as a state
    net.energy(ok, wide)                # allowed as an input


def test_exposure_any_dtype_and_list():
    net = make_net()
    assert net.latch([0.5] * 64).dtype == np.float32
    assert net.latch(np.ones(64, dtype=np.float64)).shape == (64,)


# ── Tiling ──

def test_short_exposure_tiles():
    net = make_net(input_gain=10.0, gain=0.1)
    e = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
    cube = net.latch(e)
    assert cube.shape == (64,)
    np.testing.assert_allclose(cube, cube[np.arange(64) & 3], atol=1e-6)
    full = e[np.arange(64) & 3]
    assert net.energy(e) == net.energy(full)
    assert net.energy(e) == net.energy(full, e) == net.energy(e, full)


# ── Latch and train contracts ──

def test_latch_and_train_contracts():
    rng = np.random.default_rng(11)
    net = make_net(dim=6, distance=3, sweep_budget=64)
    e = random_pattern(64, rng)
    for _ in range(20):
        net.train(e)
    w_before = net.weights
    assert not net.latched
    net.latch(e)
    assert net.latched
    np.testing.assert_array_equal(net.weights, w_before)
    assert 1 <= net.sweep_count <= 64
    assert net.at_rest
    cube_before = net.cube
    net.train(e)
    np.testing.assert_array_equal(net.cube, cube_before)
    assert np.all(np.abs(net.cube) < 1.0)

    budget1 = make_net(dim=6, distance=3, sweep_budget=1)
    budget1.weights = net.weights
    budget1.latch(random_pattern(64, rng))
    assert budget1.sweep_count == 1 and not budget1.at_rest

    e0, e1 = net.energy(e), net.energy(net.cube, e)
    assert e1 <= e0 + 1e-9

    heavy = make_net(dim=6, distance=3, eta=0.5)
    for _ in range(50):
        heavy.train(e)
    assert np.all(heavy.weights >= -1.0) and np.all(heavy.weights <= 1.0)

    assert net.latch(net.cube).shape == (64,)   # the own cube is a valid exposure


# ── Determinism and independence ──

def test_determinism():
    rng = np.random.default_rng(5)
    a, b = make_net(dim=6, distance=2), make_net(dim=6, distance=2)
    e = random_pattern(64, rng)
    for _ in range(10):
        a.train(e); b.train(e)
    np.testing.assert_array_equal(a.weights, b.weights)
    np.testing.assert_array_equal(a.latch(e), b.latch(e))
    assert a.sweep_count == b.sweep_count


def test_returned_arrays_are_copies():
    net = make_net()
    e = np.full(64, 0.3, dtype=np.float32)
    cube = net.latch(e)
    cube[:] = 0.0
    assert np.all(net.cube != 0.0)
    w = net.weights
    w[:] = 0.5
    assert not np.any(net.weights)


# ── Energy descent at every distance of two dims ──

@pytest.mark.parametrize("dim", [6, 10])
def test_energy_descends_across_sweeps(dim):
    rng = np.random.default_rng(189)
    start = random_pattern(1 << dim, rng)
    for d in range(1, dim // 2 + 1):
        prev = None
        for budget in range(1, 7):
            net = make_net(dim=dim, distance=d, init_scale=1.0, seed=189, sweep_budget=budget)
            net.latch(start)
            e = net.energy(net.cube, start)
            if prev is not None:
                assert e <= prev + 1e-6, f"dim {dim} d {d} budget {budget}"
            prev = e


# ── Weights round-trip ──

@pytest.mark.parametrize("d", [1, 2, 3])
def test_weights_roundtrip(d):
    rng = np.random.default_rng(d)
    trained = make_net(dim=6, distance=d)
    e = random_pattern(64, rng)
    for _ in range(20):
        trained.train(e)
    loaded = make_net(dim=6, distance=d, gain=0.5, input_gain=1.0)
    loaded.weights = trained.weights
    np.testing.assert_array_equal(loaded.weights, trained.weights)
    assert loaded.gain == pytest.approx(0.5)

    same = make_net(dim=6, distance=d)
    same.weights = trained.weights
    np.testing.assert_array_equal(trained.latch(e), same.latch(e))

    w = trained.weights
    with pytest.raises(ValueError):
        loaded.weights = w[:-1]
    with pytest.raises(ValueError):
        loaded.weights = np.append(w, 0.0)
    bad = w.copy(); bad[7] = 1.5
    with pytest.raises(ValueError):
        loaded.weights = bad
    bad = w.copy(); bad[7] = np.nan
    with pytest.raises(ValueError):
        loaded.weights = bad
    asym = w.copy(); asym[0] = 0.5 if asym[0] == 0.0 else -asym[0]
    with pytest.raises(ValueError):
        loaded.weights = asym
    np.testing.assert_array_equal(loaded.weights, w)   # rejected blobs leave weights alone

    if d < 3:
        wrong = make_net(dim=6, distance=d + 1)
        with pytest.raises(ValueError):
            wrong.weights = w

    sweeps = trained.sweep_count
    trained.weights = np.zeros(w.size, dtype=np.float32)
    assert trained.sweep_count == sweeps and trained.cube.shape == (64,)


# ── Persistence ──

def test_pickle_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    net = make_net(dim=6, distance=2, init_scale=0.3, seed=99, input_gain=7.0)
    e = random_pattern(64, rng)
    for _ in range(5):
        net.train(e)
    latch_before = net.latch(e)

    path = tmp_path / "model.pkl"
    net.save(path)
    loaded = hh.Hopfield.load(path)
    assert loaded.config == net.config
    np.testing.assert_array_equal(loaded.weights, net.weights)
    assert not loaded.latched
    np.testing.assert_array_equal(loaded.latch(e), latch_before)

    clone = pickle.loads(pickle.dumps(net))
    np.testing.assert_array_equal(clone.latch(e), latch_before)


def test_pickle_future_version_rejected():
    net = make_net()
    state = net.__getstate__()
    state["_version"] = 99
    fresh = hh.Hopfield.__new__(hh.Hopfield)
    with pytest.raises(ValueError, match="Upgrade"):
        fresh.__setstate__(state)


def test_repr():
    r = repr(make_net(dim=6, distance=2))
    assert r.startswith("Hopfield(dim=6") and "degree=15" in r


# ── NumPy reference: one sweep, and the energy ──

def _masks(dim, distance):
    return [m for m in range(1 << dim) if bin(m).count("1") == distance]


def _numpy_sweep(state, inp, w, dim, distance, gain, input_gain):
    """One sweep exactly as Hopfield.cpp does it: even-popcount vertices in
    index order, in place, then odd-popcount vertices the same way."""
    n = 1 << dim
    masks = _masks(dim, distance)
    s = state.astype(np.float64).copy()
    for even in (True, False):
        for v in range(n):
            if (bin(v).count("1") % 2 == 0) != even:
                continue
            h = input_gain * inp[v]
            for k, m in enumerate(masks):
                h += s[v ^ m] * w[v, k]
            s[v] = np.tanh(gain * h)
    return s


def _numpy_energy(state, inp, w, dim, distance, gain, input_gain):
    n = 1 << dim
    masks = _masks(dim, distance)
    e = 0.0
    for v in range(n):
        for k, m in enumerate(masks):
            top = 1 << (m.bit_length() - 1)
            if v & top:
                continue
            e -= w[v, k] * state[v] * state[v ^ m]
        e -= input_gain * inp[v] * state[v]
        x = float(state[v])
        if abs(x) == 1.0:
            e += math.log(2.0) / gain
        else:
            a, b = 1.0 + x, 1.0 - x
            e += 0.5 * (a * math.log(a) + b * math.log(b)) / gain
    return e


@pytest.mark.parametrize("distance", [1, 2])
def test_sweep_matches_numpy_reference(distance):
    # Distance 2 is the even case where the in-place index order matters.
    rng = np.random.default_rng(3)
    net = make_net(dim=4, distance=distance, init_scale=1.0, seed=11,
                   sweep_budget=1, gain=0.3, input_gain=2.0)
    e = random_pattern(16, rng)
    w = net.weights.astype(np.float64).reshape(16, net.degree)
    expected = _numpy_sweep(e, e.astype(np.float64), w, 4, distance, 0.3, 2.0)
    np.testing.assert_allclose(net.latch(e), expected, rtol=1e-5, atol=1e-6)


def test_energy_matches_numpy_reference():
    rng = np.random.default_rng(4)
    net = make_net(dim=4, distance=2, init_scale=1.0, seed=12, gain=0.3, input_gain=2.0)
    s = rng.uniform(-0.9, 0.9, 16).astype(np.float32)
    i = random_pattern(16, rng)
    w = net.weights.astype(np.float64).reshape(16, net.degree)
    expected = _numpy_energy(s.astype(np.float64), i.astype(np.float64), w, 4, 2, 0.3, 2.0)
    assert net.energy(s, i) == pytest.approx(expected, rel=1e-6, abs=1e-6)
