# exact-ancilla-count

**How many lines will a reversible embedding of this function need?**
Computed from the Walsh spectrum instead of the truth table, so it answers at
hundreds of inputs where enumeration stops at about ten.

> Reference implementation for
> M. A. Thornton, *Spectral Computation of the Line Requirement for Reversible
> Embedding*.

## This repository contains

- `python/` the reference implementation: the calculation front end and its CLI
  (`ancilla_v8.py`), the radix-q check, the figure script, the regression suite,
  and a local browser interface
- `c/` the C front end carrying the same ladder, with its Makefile and test suite
- `data/samples/` reference netlists shared by both implementations, with the
  canonical expected values as CSV, including six EPFL circuits with their BLIF
  twins under the suite's MIT license
- `LICENSE` (MIT)

Repository URL: `https://github.com/mitch-thornton/exact-ancilla-count`

## Quick start

```bash
git clone https://github.com/mitch-thornton/exact-ancilla-count.git
cd exact-ancilla-count

# the two worked examples of the paper (two-input AND, ISCAS-85 c17)
python3 python/ancilla_v8.py --demo

# analyze one netlist: class, garbage/line/ancilla counts or bounds, per-component audit
python3 python/ancilla_v8.py --netlist data/samples/block/block_d.v
python3 python/ancilla_v8.py --netlist data/samples/epfl/router.v

# build the C front end and run its tests, including Python against C
make -C c && make -C c test

# the odd-radix identities on random instances
python3 python/radix_check_v8.py

# local browser interface, then open http://localhost:8767
python3 python/ancilla_server_v8.py
```

## What it computes

For a switching function `f : GF(2)^n -> GF(2)^m`, let `N_dup` be the largest
number of input assignments sharing an output word. A reversible embedding needs
`g = ceil(log2 N_dup)` garbage outputs, has `L = m + g` lines, and has
`a = L - n` constant (ancilla) inputs. The tool reports all three.

`g` is a min-entropy of the output word, `g = ceil(n - H_inf)`. The collision
entropy brackets it from both sides, is exact at the lower end when every
nonempty fiber has the same size, and has an exact spectral form: a sum of
squared Walsh coefficients over **output parities**, so the sum runs over `2^m`
output subsets rather than `2^n` inputs. Those coefficients are available in
closed form for four classes:

| class | how the coefficients are obtained |
|---|---|
| affine | `g = n - rank(A)` exactly, by elimination over GF(2) |
| quadratic | symplectic rank for the magnitude, Arf invariant for the sign |
| cubic | directional-difference recursion onto quadratic inner problems |
| bounded control rank `R` | fix the `R` control variables, quadratic remainder; exact at **any** algebraic degree, in `2^m * 2^R * poly(n)` |

Whenever the `2^m` coefficients are available with their signs (the quadratic
rung through the Arf invariant, and the bounded-rank rung), inverting them
gives every fiber size and hence `N_dup` and `g` exactly, with no interval
(`dispatch.fibres_from_signed_spectrum`). The cubic rung returns the collision
probability alone and keeps the bracket.

If every nonempty parity is bent, `g = n - m + 1` with no summation. In an odd
prime radix the same identity holds with Chrestenson coefficients, and the
quadratic rung needs no sign computation there (`python/radix_check_v8.py`).

Outside those classes the problem is hard in general, and the tool says so. It
returns certified bounds and labels the component unclassified rather than
guessing. The lower bound comes from the affine hull of sampled output words
and is a Monte Carlo certificate (failure probability below `2^-10000` at the
default sample count); the upper bound `g <= n` is unconditional.

## Netlists other than the shipped ones

The front end reads structural and `assign`-style Verilog, BLIF, ISCAS `.isc`
and Espresso `.pla`, and every parse is validated before it is used. A netlist
whose outputs are undriven, or with gate inputs nothing drives, is rejected
rather than analyzed. Binary AIGER is refused on the Python side with a pointer
to the C front end, which does read it.

On a real circuit you will usually get bounds rather than an exact value. Real
design logic is high algebraic degree and lands in none of the structured
classes. Two bounds are computed for every input regardless of class, without
enumeration. On the EPFL `router` circuit that is `59 <= g <= 60` for 60 inputs
in well under a second, against `g >= 30` from pigeonhole alone.

One reported quantity is deliberately not a bound. The independence-based
estimate can exclude the true value. On the injective decoder `dec`, where the
true requirement is 0, it returns `[4,6]`. It is labeled an estimate wherever it
appears, and a regression test pins that counterexample.

```bash
python3 python/test_arbitrary_v8.py   # arbitrary-netlist checks
```

## What it shows

Exact line requirements at 512 inputs for affine specifications (CRC-32 over a
512-bit message, `g = 480`, `L = 512`, under a second) and at 72 inputs at
algebraic degree four (`g = 70` exactly, the output word is balanced), on
instances whose truth tables would have `2^512` and `2^72` rows. On every
component small enough to enumerate, the dispatched spectral answer equals the
enumerated one (21 of 21 in the classification run).

It is equally worth stating what it does not do. `c17` is five inputs and
enumerable, its true requirement is `g = 4`, and the tool declines to give an
exact value because a general five-input circuit is in none of the structured
classes. Control rank is detected from the primary inputs, so confinement that
exists only after a linear change of variables is missed. The affine lower bound
coincides with pigeonhole on `c17` and beats it on the EPFL circuits, so read
the reported `r_lin` rather than assuming either.

## Methods compared

There is no tool to race here. The comparison is against exhaustive enumeration,
which is the incumbent exact method and which does not return an answer at all
past roughly ten inputs. Where enumeration runs, the spectral route is checked
against it on every case; where it does not, the spectral route is the only
answer available. For the two-level route of Wille, Keszocze and Drechsler
(DATE 2011) the paper gives one concrete size: each output of CRC-32 over a
512-bit message is a parity of 239 to 271 message bits, so its sum-of-products
form has `2^238` to `2^270` terms per output, where the spectral route needs a
rank.

## How the implementations relate

Python is canonical, C is the performance path. Both carry the same ladder and
are held to agree: on the reference set the 20 cases both compute agree to within
`4.69e-14` against a declared tolerance of `1e-12`, 12 of them identical in
double precision. Any runtime quoted anywhere states its language; the times
printed by the Python CLI and the browser interface are reference-implementation
times and are not performance results.

The structure dispatcher is Python only. The C side is invoked per class.

## What is new in v13 versus v11

- The figure script embeds its text as TrueType fonts (`make_figs_v13.py`),
  so the PDF figures carry no Type 3 font. Nothing drawn has changed.
- Source headers carry the repository name as their project line.

## What is new in v11 versus v9

- Released under the MIT license alone; `LICENSE` is the whole license, and
  every source header says so.
- The repository has its name and URL.

## What is new in v9 versus v8.1

- The dispatcher enumerates an unclassified component with at most 14 inputs
  instead of returning bounds for it, and labels the result as enumerated. On
  the sample set this gives `c17`, `ctrl`, `dec` and the degree-4 anchor exact
  values; `router` still receives bounds. The tables are written by
  `emit_tables_v9.py`. The paper text was revised for readability; no spectral
  value changed.

## What is new in v8.1 versus v8

- The paper adds one remark: lines borrowed from a surrounding circuit, which
  enter with arbitrary values and are restored, leave the garbage count `g` and
  the ancilla count `a` unchanged. The code is identical to v8.

## What is new in v8 versus v7

- The quantity is now named correctly. The spectral count is the garbage
  requirement `g`; an embedding has `L = m + g` lines and `a = L - n` ancillae.
  Earlier versions called `g` "ancilla" and gave `max(n, m) + g` lines. The CLI,
  the browser interface and the C test runner print all three.
- The bent-parity corollary is corrected: `g = n - m + 1` exactly, checked by
  enumeration on Maiorana-McFarland functions.
- `g` is exact on the quadratic and bounded-rank rungs by inversion of the
  signed spectrum (a new proposition in the paper); the large bounded-rank
  rows are also confirmed by a balanced-output test (`g = n - m`).
- The bounded-rank generator draws distinct control monomials
  (`build_control_fn(..., distinct=True)`), and every row's algebraic degree is
  computed exactly. The earlier generator could cancel its top monomial, and the
  "degree 4" scale rows of v1 to v7 were in fact degree 3.
- Odd-radix (Chrestenson) form of the identity with a checked proposition for
  quadratic parities, `radix_check_v8.py`.
- CRC tap weights, the garbage/line/ancilla triple, and `priority` from the
  EPFL suite added; every number in the paper is generated from the results
  files.

## License

MIT; see `LICENSE`.

## Citation

```bibtex
@misc{Thornton_LineRequirement_2026,
  author = {Thornton, Mitchell A.},
  title  = {Spectral Computation of the Line Requirement for
            Reversible Embedding},
  year   = {2026},
  note   = {Reference implementation at
            https://github.com/mitch-thornton/exact-ancilla-count}
}
```

## Contact

Mitchell A. Thornton, Darwin Deason Institute for Cyber Security and Department
of Electrical and Computer Engineering, Southern Methodist University.
