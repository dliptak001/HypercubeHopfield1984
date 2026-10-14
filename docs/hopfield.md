# Sparse Hopfield network

`Hopfield` is an associative memory: the graded model of J. J. Hopfield's 1984 paper, "Neurons with graded response have collective computational properties like those of two-state neurons" (Proc. Natl. Acad. Sci. USA 81, 3088–3092), built on a hypercube. Every vertex holds a floating-point state and every edge a floating-point weight, both in [−1, 1]. The caller shows it a pattern, called an exposure. Training mode adjusts the weights so the pattern becomes a low point of the network's energy. In latch mode the pattern is both the starting state of the cube and a steady push on every vertex throughout the settle, Hopfield's input current; the weights pull the cube toward what they have stored, the pattern pulls it back toward itself, and the network returns the state where the two balance. A stored pattern comes back nearly as it went in; an unknown one is moved. It is a shaper, not a recall: the output is the input bent toward what the memory has seen, never a stored pattern looked up from a cue. The states stay floats throughout: a large gain pushes them toward the corners ±1, which is where this graded model turns into the classical binary one, but nothing is ever rounded.

## The hypercube graph

A vertex is an integer v from 0 to N − 1, where N = 2ᵈⁱᵐ and dim runs from 1 to 31. Two vertices are neighbors when their indices differ in exactly `distance` bits. Each such difference is a mask, an index with `distance` bits set, and stepping to a neighbor is one XOR: neighbor_j = v ^ mask_j. The masks are every dim-bit integer with `distance` bits set, kept in ascending order, so every vertex has the same degree, C(dim, distance). At distance 1 the masks are 1, 2, 4, …, and the graph is the ordinary hypercube; on dim 10 the degrees for distance 1 to 5 are 10, 45, 120, 210, 252. The distance runs from 1 to max(1, dim/2), the middle shell, above which the degrees repeat in reverse. Weight storage and the cost of a sweep grow with the degree.

A mask is never zero, so no vertex is its own neighbor and no field includes its own state. Each edge is one pair, read from either end; for bookkeeping its owner is the endpoint whose bit is clear at the highest set bit of the mask, and training and the energy touch each edge once, there. Flipping an odd number of bits changes the parity of the popcount, the number of 1-bits in the index. So at odd distance every edge joins an even-popcount vertex to an odd-popcount vertex, the graph splits into two classes with no edge inside either, and that split is what makes the sweep safe. At even distance the parity is preserved, an edge joins two vertices of the same class, and the sweep has to be a little more careful; both cases are covered below.

## How a vertex moves

The local field at v is the weighted sum of its neighbors' states plus the input current at v. The current is Hopfield's Iᵥ. Here it is the input gain times the exposure at v, and it stays applied for the whole settle:

hᵥ = Σ w · sᵤ  +  Iᵥ,   Iᵥ = input gain · eᵥ

The new state is the graded response:

sᵥ = tanh(gain · hᵥ)

tanh lands in (−1, 1). In Hopfield's paper the sigmoid has a steepness and the input has a resistance; here those are the one number `gain`, so there is no separate resistance and no separate input voltage. A small gain holds a vertex near 0. A large gain drives it to −1 or +1 by the sign of the field, which is the binary network's behavior, but the stored value is still the float tanh produced.

## The energy

The settle is a descent on Hopfield's energy function, written here with one weight per edge, the input term, and the [−1, 1] sigmoid. The first sum runs over edges, the other two over vertices:

E = − Σ w · sᵥ · sᵤ  −  Σ Iᵥ · sᵥ  +  Σ ∫₀ to sᵥ of (artanh / gain)

The integral is Hopfield's with the resistance folded into the gain; artanh is defined on (−1, 1), and at exactly ±1 the integral takes its limit, ln 2 / gain, so E is defined on the closed interval and nowhere outside it. Hold every vertex still except v. The slope of E in sᵥ is −hᵥ + artanh(sᵥ)/gain, which is zero exactly when sᵥ = tanh(gain · hᵥ), and since tanh is increasing that point is a minimum. So rewriting one vertex to its response, while the vertices it reads hold still, lowers E or leaves it alone, and a cube with every vertex at its response is a stationary point of E — the steady state of Hopfield's circuit equation for this exposure as input. That is what the settle finds. Two things make the slope −hᵥ: the weight on an edge is one value read from both ends, since unequal values in the two directions add a circulation the energy cannot see and the settle can orbit instead of descending; and every rewrite reads states that hold still while it writes, which the next section guarantees. `Energy(exposure)` is the energy a settle of that exposure starts from and `Energy(Cube(), exposure)` the energy it ends at; the settle itself never computes E.

## The sweep

Rewriting every vertex at once from a frozen copy of the cube is not a descent on E: each vertex's neighbors move under it, and at odd distance the two popcount classes simply trade answers, so a cube can swap with a second cube every step and never rest. The sweep instead rewrites one class at a time, in place. No two even-popcount vertices share an edge, so all of them can be rewritten together, in index order, each reading only odd states that are holding still; then the odd class is rewritten the same way from the even states just written. Each half is a descent, so the sweep is. At even distance two vertices of one class can share an edge, but the loop is unchanged: every rewrite is in place and reads the latest states, so each is a single-vertex descent taken in index order, and the only difference is that the order inside a class now matters. Either way the schedule is fixed and a settle is reproducible.

Sweeps repeat until one moves every vertex by at most the rest tolerance, or the sweep budget runs out. Graded states creep — each sweep lowers E by a shrinking amount, and the last bits of a float can keep changing long after the cube has stopped moving in any way that matters — so the tolerance is the cut-off, with zero meaning every written float must equal the one it replaced, and the budget is the stop for a cube still moving by more than that. `SweepCount` reports how many sweeps ran and `AtRest` whether the tolerance or the budget stopped them; a count equal to the budget does not decide that by itself, since the cube can come to rest on the last sweep allowed.

## Training

Training reads the exposure at the two ends of each edge and adds their product, scaled by η, to the weight on that edge, then keeps the weight inside [−1, 1]:

w ← w + η · eᵥ · eᵤ, clamped to [−1, 1]

Every edge takes its own product and both ends then read the updated weight; at distance 1, with the exposure 0.4 at vertex 0 and −0.7 at vertex 1, the edge between them changes by η · 0.4 · −0.7. A product is negative when the two values have opposite signs, so a weight can fall as well as rise, and one sitting on the clamp can be pulled back off it. While the addition stays inside the clamp, E at that exposure drops by η (eᵥ eᵤ)² on the edge: the exposure's well deepens, and so does the well of every cube that agrees with it on that pair. Repeating a similar exposure deepens the same well until the weights that carry the agreement reach the clamp and stop contributing, and the set of those wells is what the network has stored. Training never runs a sweep. Weights start at 0, which is Hopfield's model — the weights are exactly what training has added — unless the initial scale is above 0, in which case each edge starts at that scale times one `RngUniform11` draw from the engine the seed starts, taken at the owner with vertices in index order and masks in ascending order, so the same seed gives the same start.

## Latching

Latch mode loads the exposure as the starting state, holds it as the input current, and runs the settle. The result is the latch, the cube where the pull of the memory and the pull of the exposure balance: a stored exposure is already near its well and comes back nearly as it went in, while one the network has never seen is moved, and that difference is familiarity. The latch has length N, is valid until the next `Latch`, and leaves the weights untouched.

## Exposures shorter than the cube

An exposure has 2ᵏ values with k at most dim. The low k bits of a vertex index name the value that vertex carries, so when k is less than dim the pattern repeats on every copy, a copy being the vertices that share one setting of the bits above k, the extra axes; with dim 3 and k = 2, vertices 101 and 001 differ only in the extra axis and both carry value 1, and when k equals dim the exposure is the cube itself. Training treats copies alike, since matching edges in different copies receive the same addition, and an edge that crosses an extra axis joins two vertices carrying the same value, so its product is a square and training only ever pushes it up until it rests at 1.

## Weights in and out

`Weights` returns the live weight matrix, N rows of degree floats with row v holding v's weights in mask order and each edge appearing in both of its endpoints' rows. `LoadWeights` replaces the matrix from the same layout and rejects a blob with the wrong length, a value that is not finite or outside [−1, 1], or two rows that disagree on an edge, which is what a blob from another dim or distance looks like and what would break the descent. The class reads and writes no files; the caller does that. Gain, input gain, sweep budget, and rest tolerance do not touch the weights, so one trained memory can be loaded under any of them, and `LoadWeights` leaves the cube where it was, a settle of the old weights until the next `Latch`.

For a language binding the rules are short. The class never keeps a pointer into a caller's buffer, so a temporary conversion of the exposure is safe. The spans `Latch` and `Cube` return, and the reference `Weights` returns, point into the object and are overwritten in place by later calls, so a binding copies them out rather than exposing a view. All storage is allocated once, 4 · N · degree bytes of weights plus two cubes, and `Latch`, `Train`, `Energy`, and `LoadWeights` touch nothing but that storage, so they can run without an interpreter lock. One instance is not safe for concurrent calls.

## Operating points

| Product | What it controls | Target |
| --- | --- | --- |
| gain · input gain | How much of the exposure passes through. Near 1, blank weights return the exposure with its magnitudes; below 1 a shrunk copy; above about 1.5 a saturated one. | About 0.6, since for a stored pattern the memory supplies the rest of the magnitude. With gain of order 1 / degree that puts input gain of order degree / 10. |
| gain · degree | How hard the memory pulls. Of order 1 it shapes the cube; of order 100 it drives every vertex to ±1 and the exposure is lost. | Of order 1, so gain is of order 1 / degree. |
| η · T · √P / 4 | The crosstalk on a weight after T passes over P stored patterns, which is what reaches the clamp. | Well under 1, so the weights stay graded and the gain alone sets the memory's strength. |

Measured with 50 training passes. "Stored" is a stored pattern latched clean, as cosine to itself and RMS error; "repair" is the cosine to the pattern after an exposure with a quarter of its sites randomized settles, from 0.75 before.

| dim | distance | degree | η | gain | input gain | patterns | stored cos / RMS | repair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 5 | 252 | 0.003 | 0.030 | 20 | 50 | 0.987 / 0.14 | 0.85 |
| 11 | 5 | 462 | 0.003 | 0.0164 | 36.7 | 50 | 0.991 / 0.14 | 0.85 |
| 12 | 6 | 924 | 0.003 | 0.0082 | 73.3 | 50 | 0.992 / 0.14 | 0.85 |

The three rows are one rule: gain = 7.56 / degree, input gain = degree / 12.6. With these 1024-value exemplars a larger cube holds copies of the same pattern, so its extra neighbors carry nothing new and the numbers do not change; capacity depends on the pattern count, not the dimension. Storing 200 patterns instead of 50 costs about 0.02 of stored cosine at η 0.003 at every dim, and halving η recovers it. `hopfield_example` runs at the dim 10 point. The measurement is `examples/recall_benchmark.cpp`. All of these are fields of `HopfieldConfig`, fixed when the class is built.

## Symbols

| Symbol | Meaning |
| --- | --- |
| dim | Cube dimension, 1 through 31. N = 2ᵈⁱᵐ vertices. |
| v, u | Vertices: integer indices from 0 through N − 1. u is a neighbor of v. |
| distance | Hamming distance between neighbors, 1 through max(1, dim/2). Default 1, the hypercube. |
| mask, j | A mask is an index with exactly `distance` bits set; neighbor j of v is v XOR mask j. Masks are kept in ascending order, j from 0 through degree − 1. |
| degree | Neighbors per vertex, C(dim, distance). |
| edge | A neighbor pair. N·degree/2 of them. |
| owner | The endpoint of an edge whose bit is clear at the highest set bit of the edge's mask. Training and the energy visit each edge once, there. |
| popcount | The number of 1-bits in an index. At odd distance every edge joins an even-popcount vertex to an odd-popcount one. |
| cube | The network's state: one float per vertex, in [−1, 1] once settled. |
| exposure, k | The caller's pattern: 2ᵏ finite floats, k from 0 through dim. Vertex v carries the value at the position named by its low k bits. |
| copy, extra axis | The bits above k are the extra axes; a copy is the vertices that share one setting of them. The exposure repeats on every copy. |
| eᵥ, eᵤ | The exposure value carried by v, or u. |
| sᵥ, sᵤ | The state of v, or u. |
| Iᵥ | Input current at v: input gain · eᵥ, held fixed during a settle. |
| input gain | Scale of the input current, ≥ 0. Zero makes the exposure a starting state only. Default 1. |
| w | The weight on an edge, in [−1, 1], read from both ends. |
| hᵥ | Local field at v: Σ w · sᵤ over the neighbors, plus Iᵥ. |
| gain | Positive scale on hᵥ inside tanh; Hopfield's sigmoid steepness and input resistance as one number. |
| artanh | The inverse of tanh. |
| E | Hopfield's energy for a cube with states in [−1, 1], under a given input current. |
| sweep | One in-place rewrite of every even-popcount vertex in index order, then of every odd-popcount vertex. |
| settle | Sweeps repeated until one moves every vertex by at most the rest tolerance, or the sweep budget is spent. |
| rest tolerance | The largest move at one vertex that still counts as rest, ≥ 0. Zero means bit-for-bit equality. |
| sweep budget | The most sweeps a settle will run, ≥ 1. |
| `Latched`, `SweepCount`, `AtRest` | Whether a settle has run yet; how many sweeps the last one ran; and whether the tolerance rather than the budget stopped it. |
| Latch mode, latch | `Latch` loads the exposure as starting state and input current and runs the settle. The latch is the settled cube it returns: length N, valid until the next `Latch`. Weights unchanged. |
| Training mode | `Train` adds η · eᵥ · eᵤ to every edge, clamped to [−1, 1]. Cube unchanged. |
| η | The training step per exposure, > 0. |
| familiarity | How much of an exposure survives its own settle: a stored one comes back nearly unchanged, an unknown one is moved. |
| weights | The N · degree floats `Weights` returns and `LoadWeights` takes: row v holds v's weights in mask order, each edge in both endpoints' rows. |
| initial scale, seed | Weights start at 0 (initial scale 0, the default) or at initial scale · one `RngUniform11` draw per edge from the engine `seed` starts. `RngUniform11` is the project's uniform draw on [−1, 1), in `Rng.h`. |
| HopfieldConfig | The construction struct: dim, distance, gain, input gain, η, sweep budget, rest tolerance, initial scale, seed. |
| P, T | In the scaling rule: stored patterns, and training passes over them. |
| exemplar | One of the real stored patterns the diagnostics use, 1024 floats in [−1, 1]. |
