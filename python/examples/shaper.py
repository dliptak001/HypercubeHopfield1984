"""Smallest end-to-end hypercube_hopfield1984 run: store fifty patterns and
watch the network shape exposures.

Trains at the dim 10 operating point on the fifty exemplars bundled with
the repository (data/exemplars.bin, 1024 float32 each; synthesized from a
fixed seed when the file is not found), then reports three things: how a
stored pattern comes back, how a pattern with a quarter of its sites
randomized is repaired toward its stored form, and how far a never-stored
pattern is moved. It is a shaper, not a recall: the output is the input
bent toward what the memory has seen.
"""

import pathlib

import numpy as np

import hypercube_hopfield1984 as hh

DIM = 10
PASSES = 50
DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "exemplars.bin"

rng = np.random.default_rng(7)
net = hh.Hopfield(dim=DIM, gain=0.03, sweep_budget=64, rest_tolerance=0.01,
                  eta=0.003, input_gain=20.0, distance=5)
N = net.N

if DATA.is_file():
    patterns = np.fromfile(DATA, dtype=np.float32).reshape(-1, N)
    print(f"{len(patterns)} exemplars from {DATA}")
else:
    patterns = rng.uniform(-1.0, 1.0, (50, N)).astype(np.float32)
    print(f"{DATA} not found; {len(patterns)} patterns synthesized")


def cos(a, b):
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


for _ in range(PASSES):
    for p in patterns:
        net.train(p)

# The energy where each settle ends, net.energy(net.cube, exposure), is the
# familiarity readout: a stored pattern sits in the well its training dug,
# a stranger on ground the weights never shaped.
stored, stored_energy = [], []
for p in patterns:
    latch = net.latch(p)
    stored.append(cos(latch, p))
    stored_energy.append(net.energy(latch, p))

damaged_in, damaged_out, damaged_energy = [], [], []
for p in patterns:
    q = p.copy()
    hit = rng.random(N) < 0.25
    q[hit] = rng.uniform(-1.0, 1.0, hit.sum())
    latch = net.latch(q)
    damaged_in.append(cos(q, p))
    damaged_out.append(cos(latch, p))
    damaged_energy.append(net.energy(latch, q))

novel_self, novel_nearest, novel_energy = [], [], []
for _ in range(5):
    novel = rng.uniform(-1.0, 1.0, N).astype(np.float32)
    latch = net.latch(novel)
    novel_self.append(cos(latch, novel))
    novel_nearest.append(max(cos(latch, p) for p in patterns))
    novel_energy.append(net.energy(latch, novel))

print(net)
print()
print(f"stored pattern, cosine to itself after the settle:   {np.mean(stored):.3f}"
      f"   energy {np.mean(stored_energy):8.1f}")
print(f"damaged pattern, cosine to the stored one:            "
      f"{np.mean(damaged_in):.3f} -> {np.mean(damaged_out):.3f}"
      f"   energy {np.mean(damaged_energy):8.1f}")
print(f"novel pattern, cosine to itself / to nearest stored:  "
      f"{np.mean(novel_self):.3f} / {np.mean(novel_nearest):.3f}"
      f"   energy {np.mean(novel_energy):8.1f}")
print(f"last settle: {net.sweep_count} sweeps, at rest: {net.at_rest}")

print()
print("How to read this. Cosine measures how much the settle changed the input:")
print(f"a stored pattern ({np.mean(stored):.3f}) and a never-seen one ({np.mean(novel_self):.3f}) both")
print("come back almost unchanged, so cosine cannot tell them apart.")
print("Energy measures how well the weights fit what came out, and it can:")
print(f"stored {np.mean(stored_energy):,.0f}, damaged {np.mean(damaged_energy):,.0f}, "
      f"never seen {np.mean(novel_energy):,.0f}.")
print("Lower energy means a more familiar input.")
