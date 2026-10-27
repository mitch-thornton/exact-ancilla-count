#!/usr/bin/env python3
# file:        scripts/ancilla_v8.py
# project:     exact-ancilla-count (spectral computation of the line requirement for reversible embedding)
# brief:       Python reference implementation and CLI; regenerates every table of the paper
# purpose:     Single source of every computed value in the paper (results/ancilla_results_v8.json),
#              from which scripts/emit_tables_v8.py writes the LaTeX tables and macros
# description: Garbage requirement g = ceil(log2 N_dup), lines L = m + g, ancilla a = L - n, the
#              collision bracket, and the spectral ladder (affine, quadratic, cubic, bounded control
#              rank) with enumeration audits. v8 adds the balanced/regular detection that makes the
#              large bounded-rank rows exact, the corrected vectorial-bent corollary with an explicit
#              Maiorana-McFarland check, exact integer arithmetic for the bent values, CRC tap weights,
#              the garbage/ancilla/line triple in every report, exact degrees for every constructed
#              row, and exact g by inversion of the signed spectrum on the quadratic and
#              bounded-rank rungs.
# inputs:      data/samples/ (reference netlists); constructed specifications with fixed seeds
# outputs:     results/ancilla_results_v8.json (via --tables), console reports
# invocation:  python3 scripts/ancilla_v8.py --tables | --demo | --netlist FILE | --crosscheck | --check
#              (about 35 s for --tables on one core; run from the bundle root or anywhere)
# created:     bundle ancilla_spectral_bundle_v8, 2026-10-08
# modified:    bundle ancilla_spectral_bundle_v13, 2026-10-27
# history:     v1 2026-07-26 first driver (ancilla_v1.py); v1.1 arbitrary-netlist fixes;
#              v8 2026-10-08 renamed, garbage/ancilla/line accounting, exact bent values,
#              balanced detection, CRC tap weights, results moved to results/;
#              v11 2026-10-26 license line simplified to plain MIT, no code change;
#              v13 2026-10-27 project line of the header reworded, no code change
# author:      Mitchell A. Thornton <mitch@smu.edu>, ORCID 0000-0003-3559-9511
# SPDX-License-Identifier: MIT
# (c) 2026 Mitchell A. Thornton
"""ancilla_v8.py -- reference front end for the spectral line-requirement paper.

This is the Python reference implementation of every calculation reported in
"Spectral Computation of the Line Requirement for Reversible Embedding".
It is the calculation front end only. The synthesis back end of the parent tool
is not part of this bundle.

What it computes
----------------
For a switching function f : GF(2)^n -> GF(2)^m,

    N_dup(f) = max_y |f^{-1}(y)|         largest fibre
    g(f)     = ceil(log2 N_dup(f))       garbage requirement
    L(f)     = m + g(f)                  circuit lines of a minimal embedding
    a(f)     = L(f) - n                  constant (ancilla) inputs
    CP       = sum_y p(y)^2              collision probability
    g in [ceil(n - H_2), ceil(n - H_2/2)]   certified interval, H_2 = -log2 CP

Naming note. The library modules in this directory predate the v8 relabeling and
call the garbage requirement `v`. Every `v` field they return is the paper's g;
this driver adds the `g`, `ancilla` and `lines` fields alongside.

CP is obtained from the Walsh spectrum over output parities rather than by
enumeration, in closed form on four structural classes: affine, quadratic,
cubic, and bounded control rank. See the paper for the identities.

Commands
--------
    python3 scripts/ancilla_v8.py --demo
        the two worked examples of the paper (two-input AND, c17)

    python3 scripts/ancilla_v8.py --tables
        regenerate every table in the paper; writes results/ancilla_results_v8.json

    python3 scripts/ancilla_v8.py --netlist FILE [--trials N]
        classify one netlist and report g, L, a or the certified interval

    python3 scripts/ancilla_v8.py --crosscheck [--tol 1e-12]
        compare this implementation against the C front end in csrc/ on the
        reference sample set in data/samples/

    python3 scripts/ancilla_v8.py --check
        fast smoke path used by build.sh --check

Determinism: every randomized structural test is seeded. Pass --seed to change
the seed; the shipped results use --seed 0 with PYTHONHASHSEED=0.

Runtimes printed here are Python reference-implementation times. Per the project
synchronization rule they are not performance results.
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)


def _find(*candidates):
    """Resolve a directory in either the working-bundle or the repo-image layout."""
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


# Working bundle:  <bundle>/scripts/ (this file and the library), <bundle>/data/samples, <bundle>/csrc
# Repo image:      <repo>/python/  <repo>/data/samples  <repo>/c
LIBDIR = _find(os.path.join(HERE, "scripts"), HERE)
DATADIR = _find(os.path.join(HERE, "data", "samples"),
                os.path.join(PARENT, "data", "samples"))
CDIR = _find(os.path.join(HERE, "csrc"), os.path.join(PARENT, "csrc"),
             os.path.join(PARENT, "c"))
RESULTS_DEFAULT = os.path.join(PARENT, "results", "ancilla_results_v8.json")
sys.path.insert(0, LIBDIR)

import netlist                      # noqa: E402
import net_tags as nt               # noqa: E402
import symplectic                   # noqa: E402
import cubic_exact                  # noqa: E402
import bounded_rank                 # noqa: E402
import structured                   # noqa: E402
import order_profile                # noqa: E402
import trilinear                    # noqa: E402
import arf                          # noqa: E402
import dispatch as dispatcher       # noqa: E402
import embedcost                    # noqa: E402
import gf2subset                    # noqa: E402
import verilog_front_v1 as vfront   # noqa: E402


# --------------------------------------------------------------------------
# Core quantities. These are the definitions of Section III of the paper.
# --------------------------------------------------------------------------

def fibres(nl):
    """Exact fibre-size histogram by enumeration. Reference path only."""
    T = nt.NetTags(nl)
    words = Counter()
    for x in range(1 << T.n):
        assign = {pi: (x >> k) & 1 for k, pi in enumerate(T.inputs)}
        sv = netlist.simulate(T.nl, assign)
        w = 0
        for j, o in enumerate(T.outputs):
            if sv[o]:
                w |= (1 << j)
        words[w] += 1
    return words, T.n, T.m


def g_of_ndup(ndup):
    """g = ceil(log2 N_dup) in exact integer arithmetic (no floating point)."""
    return (ndup - 1).bit_length() if ndup > 1 else 0


def triple(n, m, g):
    """(garbage, lines, ancilla) = (g, m + g, m + g - n)."""
    return dict(g=g, lines=m + g, ancilla=m + g - n)


def g_exact(nl):
    """Exact garbage requirement by enumeration. Feasible only for small n."""
    words, n, m = fibres(nl)
    ndup = max(words.values())
    return g_of_ndup(ndup), ndup, n, m


def fibre_summary(words, n, m):
    """Ndup, g, whether all nonempty fibres are equal (regular), and whether the
    output is balanced (uniform on GF(2)^m, i.e. 2^m fibres of size 2^(n-m))."""
    sizes = sorted(words.values(), reverse=True)
    ndup = sizes[0]
    regular = len(set(sizes)) == 1
    balanced = regular and len(sizes) == (1 << m) and ndup == (1 << (n - m))
    return dict(ndup=ndup, fibres=len(sizes), regular=regular,
                balanced=balanced, **triple(n, m, g_of_ndup(ndup)))


def cp_from_fibres(words, n):
    p = [c / (1 << n) for c in words.values()]
    return sum(t * t for t in p)


def entropies(words, n):
    """(H_inf, H_2) of the output word, in bits."""
    p = [c / (1 << n) for c in words.values()]
    cp = sum(t * t for t in p)
    return -math.log2(max(p)), -math.log2(cp)


def v_bracket(cp, n):
    """Certified interval on v from CP. Proposition 2 of the paper."""
    h2 = -math.log2(cp)
    return math.ceil(n - h2), math.ceil(n - h2 / 2.0)


# --------------------------------------------------------------------------
# Worked examples of the paper
# --------------------------------------------------------------------------

def demo():
    print("Example 1: two-input AND")
    c = Counter()
    for a in (0, 1):
        for b in (0, 1):
            c[a & b] += 1
    n, m = 2, 1
    ndup = max(c.values())
    g = g_of_ndup(ndup)
    t = triple(n, m, g)
    hinf, h2 = entropies(c, n)
    cp = cp_from_fibres(c, n)
    print(f"  fibres {dict(c)}  N_dup={ndup}  garbage g={g}  "
          f"lines L=m+g={t['lines']}  ancilla a=L-n={t['ancilla']}")
    print(f"  H_inf={hinf:.4f} bits   CP={cp:.4f}   H_2={h2:.4f} bits")
    print(f"  check: ceil(n - H_inf) = {math.ceil(n - hinf)}")

    print("\nExample 2: ISCAS-85 c17")
    p = os.path.join(DATADIR, "c17.v")
    nl = netlist.load(p)
    words, n, m = fibres(nl)
    fs = fibre_summary(words, n, m)
    ndup, g = fs["ndup"], fs["g"]
    hinf, h2 = entropies(words, n)
    cp = cp_from_fibres(words, n)
    lo, hi = v_bracket(cp, n)
    print(f"  n={n} m={m}  fibre sizes {sorted(words.values(), reverse=True)}")
    print(f"  N_dup={ndup}  garbage g={g}  lines L={fs['lines']}  ancilla a={fs['ancilla']}")
    print(f"  CP={cp:.6f}  H_2={h2:.4f}  H_inf={hinf:.4f}")
    print(f"  bracket on log2 N_dup: [{n-h2:.4f}, {n-h2/2:.4f}]  "
          f"true {math.log2(ndup):.4f}")
    print(f"  reported interval on g: [{lo},{hi}]   true g = {g}")
    return dict(and_gate=dict(n=2, m=1, ndup=3, **triple(2, 1, g_of_ndup(3))),
                c17=dict(n=n, m=m, fibre_sizes=sorted(words.values(), reverse=True),
                         cp=cp, h2=h2, hinf=hinf, lo=lo, hi=hi, **fs))


# --------------------------------------------------------------------------
# Table regeneration
# --------------------------------------------------------------------------

def table_affine():
    """Affine specifications: exact g = n - rank, with lines and ancillae."""
    rows = []
    cases = [
        ("CRC-8, message 64", structured.build_crc(64, 0x07, 8, "CRC-8"), 56),
        ("CRC-16-CCITT, msg 128", structured.build_crc(128, 0x1021, 16, "CRC-16"), 112),
        ("CRC-32, message 256", structured.build_crc(256, 0x04C11DB7, 32, "CRC-32"), 224),
        ("CRC-32, message 512", structured.build_crc(512, 0x04C11DB7, 32, "CRC-32"), 480),
        ("linear map, rank 40", structured.build_linear(64, 64, 40, seed=1), 24),
        ("linear map, rank 100", structured.build_linear(128, 128, 100, seed=1), 28),
        ("linear map, rank 150", structured.build_linear(256, 200, 150, seed=1), 106),
        ("linear map, rank 256", structured.build_linear(512, 256, 256, seed=1), 256),
    ]
    for label, nl, true_v in cases:
        r = structured.measure(nl, true_v)
        r["label"] = label
        r["match"] = (true_v is None) or (r["v_measured"] == true_v)
        r.update(triple(r["n"], r["m"], r["v_measured"]))
        if label.startswith("CRC"):
            r["tap_weights"] = output_support_sizes(nl)
        rows.append(r)
        extra = (f"  taps {min(r['tap_weights'])}..{max(r['tap_weights'])}"
                 if "tap_weights" in r else "")
        print(f"  {label:24s} n={r['n']:4d} m={r['m']:4d} rank={r['r_lin']:4d} "
              f"g={r['g']:4d} L={r['lines']:4d} a={r['ancilla']:4d} true={true_v}  "
              f"{r['seconds']:.2f}s{extra}")
    return rows


def output_support_sizes(nl):
    """Number of inputs each output depends on, for an AFFINE netlist: output j
    depends on input i iff f(e_i)_j != f(0)_j. One simulation per input."""
    T = nt.NetTags(nl)
    def ev(xint):
        sv = netlist.simulate(nl, {p: (xint >> k) & 1 for k, p in enumerate(T.inputs)})
        return [sv[o] for o in T.outputs]
    f0 = ev(0)
    w = [0] * T.m
    for i in range(T.n):
        fi = ev(1 << i)
        for j in range(T.m):
            if fi[j] != f0[j]:
                w[j] += 1
    return w


def _quad_eval(nl):
    T = nt.NetTags(nl)

    def evalf(rint):
        sv = netlist.simulate(nl, {p: (rint >> k) & 1
                                   for k, p in enumerate(T.inputs)})
        return tuple(sv[o] for o in T.outputs)
    return evalf


def table_quadratic():
    """Spectral CP against enumeration on quadratics, with fibre regularity."""
    import order_profile as op
    rows = []
    cases = [("quadratic, 3 terms", op.build_quadratic(12, 8, terms=3, seed=2)),
             ("quadratic, 6 terms", op.build_quadratic(12, 8, terms=6, seed=2)),
             ("quadratic, 2 terms", op.build_quadratic(10, 6, terms=2, seed=4)),
             ("affine, degree 1", structured.build_linear(12, 8, 6, seed=1))]
    for label, nl in cases:
        t0 = time.perf_counter()
        n, m, Br, L = symplectic.extract_forms(nl)
        evalf = _quad_eval(nl)
        cp = symplectic.cp_symplectic(n, m, Br, L, evalf, evalf(0))
        dt = time.perf_counter() - t0
        cpb, vb = symplectic.brute_cp_v(nl)
        lo, hi = symplectic.v_bracket(cp, n)
        ok = abs(cp - cpb) < 1e-12
        fw, nn, mm = cubic_exact.truth_words(nl)
        fs = fibre_summary(Counter(fw), nn, mm)
        assert fs["g"] == vb
        # exact N_dup from the SIGNED spectrum (Arf sign + inversion), no enumeration
        t1 = time.perf_counter()
        nd_spec = dispatcher.exact_ndup_quadratic(nl, list(nl.inputs), list(nl.outputs), n, m)
        dt_inv = time.perf_counter() - t1
        g_spec = g_of_ndup(nd_spec)
        rows.append(dict(label=label, n=n, m=m, cp=cp, cp_brute=cpb,
                         match=ok, v_lo=lo, v_hi=hi, v_brute=vb, seconds=round(dt, 3),
                         ndup=fs["ndup"], regular=fs["regular"], balanced=fs["balanced"],
                         ndup_spectral=nd_spec, g_spectral=g_spec,
                         g_spectral_match=(nd_spec == fs["ndup"]),
                         inversion_seconds=round(dt_inv, 3),
                         lo_exact=(lo == vb), **triple(n, m, vb)))
        print(f"  {label:20s} n={n:3d} m={m:3d} cp={cp:.12e} "
              f"brute={cpb:.12e} match={'yes' if ok else 'NO'} g={vb} "
              f"bracket=[{lo},{hi}] inversion N_dup={nd_spec} ({'=' if nd_spec == fs['ndup'] else '!='} enum) "
              f"regular={fs['regular']} balanced={fs['balanced']}")
    # low-order truncation, quoted in the paper text
    nl = op.build_quadratic(20, 12, terms=3, seed=7)
    n, m, Br, L = symplectic.extract_forms(nl)
    evalf = _quad_eval(nl)
    full = symplectic.cp_symplectic(n, m, Br, L, evalf, evalf(0))
    trunc = {K: symplectic.cp_symplectic(n, m, Br, L, evalf, evalf(0), K=K)
             for K in (0, 1, 2, 3)}
    print(f"  truncation n=20 m=12: full={full:.6e} " +
          " ".join(f"K={K}:{v:.6e}" for K, v in trunc.items()))
    rows.append(dict(label="truncation n=20 m=12", n=n, m=m, cp=full,
                     truncated={str(k): v for k, v in trunc.items()},
                     frac_K0=trunc[0] / full))
    return rows


def table_bounded_rank():
    """Control-variable reduction; the scale rows are tested for balance."""
    rows = []
    # distinct=True (v8): the control monomials of each degree are distinct, so the
    # whole function has algebraic degree exactly ctrl_deg. The v1 to v7 generator
    # could draw the same monomial twice and cancel it, and with R = ctrl_deg = 4 it
    # always did, so the earlier scale rows had true degree 3 while the table said 4.
    for (n, m, R, cd) in [(12, 3, 4, 3), (12, 3, 4, 4), (14, 3, 5, 4), (14, 4, 4, 3)]:
        nl = bounded_rank.build_control_fn(n, m, R, cd, seed=n + cd, distinct=True)
        import cubic as cubicmod
        deg = max(cubicmod.anf_degree(nl, j) for j in range(m))
        rdeg = max(bounded_rank.control_restricted_degree(nl, R))
        assert rdeg == deg, f"restricted degree {rdeg} != ANF degree {deg} at n={n}"
        t0 = time.perf_counter()
        cp = bounded_rank.cp_bounded(nl, R)
        dt = time.perf_counter() - t0
        fw, nn, mm = cubic_exact.truth_words(nl)
        cpb = cubic_exact.brute_cp(fw, nn)
        ok = abs(cp - cpb) < 1e-12
        fs = fibre_summary(Counter(fw), nn, mm)
        lo, hi = v_bracket(cp, n)
        Ws = nonempty_parity_walsh(nl, R)
        fib = dispatcher.fibres_from_signed_spectrum([1 << n] + Ws, m)
        nd_spec = max(fib)
        assert nd_spec == fs["ndup"], f"inversion N_dup {nd_spec} != enumerated {fs['ndup']}"
        rows.append(dict(n=n, m=m, R=R, ctrl_deg=cd, anf_deg=deg, restricted_deg=rdeg,
                         cp=cp, cp_brute=cpb, match=ok, seconds=round(dt, 2),
                         scale=False, v_lo=lo, v_hi=hi, ndup=fs["ndup"],
                         ndup_spectral=nd_spec, g_spectral=g_of_ndup(nd_spec),
                         g_spectral_match=True,
                         regular=fs["regular"], balanced=fs["balanced"],
                         lo_exact=(lo == fs["g"]), **triple(n, m, fs["g"])))
        print(f"  n={n:3d} m={m:2d} R={R} anf_deg={deg} cp={cp:.12e} "
              f"brute={cpb:.12e} match={'yes' if ok else 'NO'} g={fs['g']} "
              f"bracket=[{lo},{hi}] balanced={fs['balanced']} {dt:.2f}s")
    for (n, m, R, cd) in [(40, 3, 4, 4), (56, 3, 4, 4), (72, 2, 4, 4)]:
        nl = bounded_rank.build_control_fn(n, m, R, cd, seed=7, distinct=True)
        t0 = time.perf_counter()
        cp = bounded_rank.cp_bounded(nl, R)
        dt = time.perf_counter() - t0
        lo, hi = v_bracket(cp, n)
        # exact whole-function degree, 2^R simulations per output (see bounded_rank)
        rdeg = max(bounded_rank.control_restricted_degree(nl, R))
        deg = rdeg if rdeg >= 3 else 2
        # Exact integer Walsh coefficients of every nonempty parity. If all vanish the
        # output word is uniform on GF(2)^m, every fibre has 2^(n-m) elements, and
        # g = n - m exactly (Proposition "balanced specifications" of the paper).
        Ws = nonempty_parity_walsh(nl, R)
        balanced = all(w == 0 for w in Ws)
        # exact fibre sizes by inversion of the signed spectrum (always available on
        # this rung), hence exact N_dup and g with no interval; the balanced test is
        # an independent check of the same value
        fib = dispatcher.fibres_from_signed_spectrum([1 << n] + Ws, m)
        nd_spec = max(fib)
        g_spec = g_of_ndup(nd_spec)
        if balanced:
            assert nd_spec == 1 << (n - m)
        rec = dict(n=n, m=m, R=R, ctrl_deg=cd, anf_deg=deg, restricted_deg=rdeg,
                   cp=cp, cp_brute=None,
                   match=None, v_lo=lo, v_hi=hi, seconds=round(dt, 1),
                   scale=True, balanced=balanced, exact=True,
                   ndup=nd_spec, ndup_spectral=nd_spec, g_spectral=g_spec,
                   max_abs_nonempty_W=max(abs(w) for w in Ws), **triple(n, m, g_spec))
        rows.append(rec)
        tag = f"g={g_spec} exact by inversion" + (" (balanced)" if balanced else "")
        print(f"  n={n:3d} m={m:2d} R={R} deg={deg} cp={cp:.4e} {tag}  {dt:.1f}s"
              f"   (enumeration needs 2^{n} rows)")
    return rows


def nonempty_parity_walsh(nl, R):
    """Integer W_S for every nonempty output subset S, via the control-coset
    factorization of bounded_rank.W_T_bounded. Exact integers, no floating point."""
    T = nt.NetTags(nl)
    def f_eval(x):
        sv = netlist.simulate(nl, {p: (x >> k) & 1 for k, p in enumerate(T.inputs)})
        return tuple(sv[o] for o in T.outputs)
    out = []
    for Tset in range(1, 1 << T.m):
        bits = [j for j in range(T.m) if (Tset >> j) & 1]
        out.append(bounded_rank.W_T_bounded(f_eval, bits, T.n, R))
    return out


def table_cubic():
    """Cubic route, including the degree-4 negative control."""
    import cubic as cubicmod
    rows = []
    cases = [(f"cubic, seed {s}", cubicmod.build_deg(9, 3, 3, terms=4, seed=s),
              "exact") for s in (11, 22, 33)]
    cases.append(("structured cubic",
                  cubicmod.build_structured_cubic(10, 4, seed=5), "exact"))
    cases.append(("degree 4 (control)",
                  cubicmod.build_deg(9, 3, 4, terms=3, seed=7), "mismatch"))
    for label, nl, expect in cases:
        fw, n, m = cubic_exact.truth_words(nl)
        deg = max(cubicmod.anf_degree(nl, j) for j in range(m))   # exact, by Moebius transform
        assert deg == (3 if expect == "exact" else 4), f"{label}: algebraic degree is {deg}"
        cp = cubic_exact.cp_cubic_exact(fw, n, m)
        cpb = cubic_exact.brute_cp(fw, n)
        ok = abs(cp - cpb) < 1e-12
        fs = fibre_summary(Counter(fw), n, m)
        lo, hi = v_bracket(cp, n)
        rows.append(dict(label=label, n=n, m=m, anf_deg=deg, cp=cp, cp_brute=cpb,
                         match=ok, expect=expect, ndup=fs["ndup"], regular=fs["regular"],
                         balanced=fs["balanced"], v_lo=lo, v_hi=hi,
                         lo_exact=(lo == fs["g"]), **triple(n, m, fs["g"])))
        note = "" if expect == "exact" else "   <- negative control, expected"
        print(f"  {label:20s} n={n:3d} m={m:3d} cp={cp:.12e} "
              f"brute={cpb:.12e} agree={'yes' if ok else 'no'}{note}")
    return rows


def table_order_profile():
    """Order-profile diagnostic quoted in the paper text."""
    import order_profile as op
    rows = []
    cases = [("affine full-rank", structured.build_linear(12, 8, 8, seed=1)),
             ("affine rank 5", structured.build_linear(12, 8, 5, seed=1)),
             ("quadratic 3-term", op.build_quadratic(12, 8, terms=3, seed=2)),
             ("quadratic 6-term", op.build_quadratic(12, 8, terms=6, seed=2))]
    for label, nl in cases:
        prof = op.order_profile(nl)
        rows.append(dict(label=label, profile=prof))
        print(f"  {label:20s} {prof}")
    return rows


def _gf2k_mul(a, b, k, poly):
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if (a >> k) & 1:
            a ^= poly
    return r


_IRRED = {2: 0b111, 3: 0b1011, 4: 0b10011, 5: 0b100101, 6: 0b1000011, 8: 0b100011011}


def vectorial_bent_check(n, m):
    """Enumerate the Maiorana-McFarland vectorial bent function
    F(x, y) = x * y in GF(2^(n/2)), keeping the first m coordinates (m <= n/2),
    and return CP, N_dup, g and whether every nonempty parity is bent."""
    k = n // 2
    poly = _IRRED[k]
    q = 1 << n
    words = Counter()
    sign = []                      # per nonempty parity, running Walsh sums
    masks = list(range(1, 1 << m))
    W = {u: 0 for u in masks}
    for xy in range(q):
        x, y = xy & ((1 << k) - 1), xy >> k
        f = _gf2k_mul(x, y, k, poly) & ((1 << m) - 1)
        words[f] += 1
        for u in masks:
            W[u] += 1 if bin(u & f).count("1") % 2 == 0 else -1
    # zero-frequency coefficient only; bentness of every parity needs all a:
    # check |W(u.F + a.x)| = 2^(n/2) for all a by a fast Walsh transform per u
    allbent = True
    for u in masks:
        tt = [0] * q
        for xy in range(q):
            x, y = xy & ((1 << k) - 1), xy >> k
            f = _gf2k_mul(x, y, k, poly) & ((1 << m) - 1)
            tt[xy] = 1 - 2 * (bin(u & f).count("1") % 2)
        h = 1
        while h < q:
            for i in range(0, q, 2 * h):
                for j in range(i, i + h):
                    a, b = tt[j], tt[j + h]
                    tt[j], tt[j + h] = a + b, a - b
            h *= 2
        if any(abs(c) != (1 << k) for c in tt):
            allbent = False
            break
    fs = fibre_summary(words, n, m)
    cp = cp_from_fibres(words, n)
    cp_formula = 2.0 ** (-m) + (1 - 2.0 ** (-m)) * 2.0 ** (-n)
    ndup_upper = (1 << (n - m)) + (1 << k) - (1 << (k - m))
    return dict(n=n, m=m, all_parities_bent=allbent, cp_enum=cp, cp_formula=cp_formula,
                cp_match=abs(cp - cp_formula) < 1e-15, ndup=fs["ndup"],
                ndup_upper=ndup_upper, g_enum=fs["g"], g_formula=n - m + 1,
                g_match=(fs["g"] == n - m + 1), **triple(n, m, fs["g"]))


def table_bent():
    """Vectorial bent parities. Corollary of the paper: if every nonempty output
    parity is bent (so m <= n/2) then CP = 2^-m + (1 - 2^-m) 2^-n and
    g = n - m + 1 exactly. Checked here by enumeration on Maiorana-McFarland
    constructions, and the closed form tabulated in exact integer arithmetic."""
    rows = []
    for (n, m) in [(4, 1), (4, 2), (8, 1), (8, 2), (8, 4), (12, 3), (12, 6)]:
        r = vectorial_bent_check(n, m)
        rows.append(dict(kind="enumerated", **r))
        print(f"  MM bent n={n:2d} m={m}: all parities bent={r['all_parities_bent']} "
              f"CP enum={r['cp_enum']:.10f} formula={r['cp_formula']:.10f} "
              f"N_dup={r['ndup']} (upper {r['ndup_upper']}) g={r['g_enum']} "
              f"formula n-m+1={r['g_formula']} match={r['g_match']}")
    for (n, m) in [(16, 8), (32, 16), (64, 32), (128, 64), (256, 128), (256, 1)]:
        g = n - m + 1
        rows.append(dict(kind="closed_form", n=n, m=m, g_formula=g, **triple(n, m, g)))
        print(f"  closed form n={n:3d} m={m:3d}: g = n - m + 1 = {g}, L = {m+g}")
    return rows


def table_dispatch():
    """Automatic classification over the sample set."""
    rows = []
    samples = DATADIR
    anchors = [structured.build_crc(64, 0x07, 8, "CRC-8"),
               structured.build_linear(64, 32, 20, seed=2)]
    for nl in anchors:
        rows.append(dispatcher.dispatch(nl))
    for pat in ["*.v", "*.isc", "epfl/*.v", "quad/*.v", "cubic/*.v",
                "brank/*.v", "block/*.v"]:
        for p in sorted(glob.glob(os.path.join(samples, pat))):
            try:
                nl = vfront.load_any(p)
                rows.append(dispatcher.dispatch(nl))
            except Exception as exc:                       # noqa: BLE001
                print(f"  {os.path.basename(p):22s} SKIP ({type(exc).__name__})")
    return rows


def table_bounds():
    """What an arbitrary circuit receives: the sound bounds and the labelled
    estimate on the EPFL circuits quoted in the paper, plus c17."""
    rows = []
    for rel in ("c17.v", "epfl/router.v", "epfl/priority.v", "epfl/dec.v"):
        p = os.path.join(DATADIR, rel)
        if not os.path.exists(p):
            print(f"  {rel}: absent, skipped")
            continue
        t0 = time.perf_counter()
        out = analyze(p, verbose=False)
        dt = time.perf_counter() - t0
        rec = dict(name=out["name"], n=out["n"], m=out["m"], gates=out["gates"],
                   analyze_seconds=round(dt, 2),
                   classes=out["classes"], exact_g=out["exact_v"],
                   lower=out["lower"], upper=out["upper"], r_lin=out["r_lin"],
                   pigeonhole=out["pigeonhole"],
                   estimate=(list(out["estimate"]["second_order"]) if out["estimate"] else None),
                   seconds=out["seconds"])
        if out["n"] <= 12:
            g, ndup, _, _ = g_exact(vfront.load_any(p))
            rec.update(g_enum=g, ndup_enum=ndup)
        rows.append(rec)
        print(f"  {rec['name']:10s} n={rec['n']:3d} m={rec['m']:3d} r_lin={rec['r_lin']:3d} "
              f"bounds {rec['lower']}<=g<={rec['upper']} (pigeonhole {rec['pigeonhole']}) "
              f"estimate {rec['estimate']} enum g={rec.get('g_enum')}")
    return rows


def run_tables(outfile):
    import platform
    res = {"meta": dict(driver="scripts/ancilla_v8.py", bundle="ancilla_spectral_bundle_v8",
                        python=platform.python_version(), machine=platform.machine(),
                        system=platform.system(), pythonhashseed=os.environ.get("PYTHONHASHSEED"),
                        timings="Python reference-implementation wall clock, single core")}
    print("Worked examples")
    res["examples"] = demo()
    print("\nTable: affine specifications")
    res["affine"] = table_affine()
    print("\nTable: quadratic specifications against enumeration")
    res["quadratic"] = table_quadratic()
    print("\nTable: bounded control rank")
    res["bounded_rank"] = table_bounded_rank()
    print("\nTable: cubic route and degree boundary")
    res["cubic"] = table_cubic()
    print("\nArf-signed Walsh coefficient against direct summation")
    bad = arf.validate(trials=400)
    res["arf"] = dict(disagree=bad, trials=400)
    print(f"  {400-bad}/400 exact")
    print("\nOrder profile")
    res["order_profile"] = table_order_profile()
    print("\nVectorial bent parities (corollary check)")
    res["bent"] = table_bent()
    print("\nTable: automatic classification")
    res["dispatch"] = table_dispatch()
    print("\nSound bounds on arbitrary circuits")
    res["bounds"] = table_bounds()
    os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
    with open(outfile, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwrote {outfile}")
    return res


# --------------------------------------------------------------------------
# Single netlist: what an arbitrary circuit actually gets back
# --------------------------------------------------------------------------

def sound_bounds(nl, seed=0):
    """Bounds on v that hold for ANY circuit, computed without enumeration.

    Lower: the affine-rank bound v >= n - r_lin, where r_lin is the linear
    dimension of the image. It subsumes pigeonhole v >= n - m because
    r_lin <= m, and on real circuits it is often much stronger.
    Upper: v <= n, trivially, since there are only 2^n input assignments.

    The upper bound is unconditional. The lower bound is a Monte Carlo certificate:
    r_lin is the affine-hull dimension of 64*max(256, n+64) sampled output words, the
    sampled hull can only be too small, and the paper (Section VI-E) bounds the
    failure probability below 2^-10000. Unlike the independence estimate below, neither can exclude
    the true value.
    """
    import random
    T = nt.NetTags(nl)
    words = gf2subset.sample_words(T, max(256, T.n + 64), random.Random(seed))
    r_lin = gf2subset.affine_rank(words)
    return dict(n=T.n, m=T.m, r_lin=r_lin,
                pigeonhole=max(0, T.n - T.m),
                lower=max(0, T.n - r_lin), upper=T.n)


def independence_estimate(nl, seed=0, passes=256):
    """First and second order estimate of v under an output-bit independence
    assumption.

    NOT A BOUND. It can and does exclude the true value: on the EPFL decoder
    `dec` it reports v in [4,6] where the true requirement is 0, because the
    outputs are one-hot and maximally dependent. It is reported because it is
    informative on unstructured circuits, and it is labelled everywhere it is
    reported. See test_arbitrary_v1.py, which pins this counterexample.
    """
    T = nt.NetTags(nl)
    try:
        e = embedcost.estimate(T, passes=passes, seed=seed)
    except Exception:                                        # noqa: BLE001
        return None
    return dict(first_order=tuple(e["v_indep_bracket"]),
                second_order=tuple(e["v_2nd_bracket"]),
                cp_indep=e["cp_indep"], cp_2nd=e["cp_2nd"])


def analyze(path, trials=48, verbose=True, seed=0, estimate=True):
    """Full analysis of one netlist, in any supported format.

    Returns a record whose contract does not depend on whether a structural
    class was found:

        exact_v      the ancilla requirement, or None if none was established
        lower/upper  bounds that always hold
        estimate     labelled, and never a bound
    """
    nl = vfront.load_any(path)
    rec = dispatcher.dispatch(nl, trials=trials, verbose=False)
    b = sound_bounds(nl, seed=seed)
    est = independence_estimate(nl, seed=seed) if estimate else None
    cls = "+".join(sorted({r["cls"] for r in rec.get("blocks", [])})) or "none"
    out = dict(name=nl.name, n=b["n"], m=b["m"], gates=nl.n_gates,
               components=rec.get("components"), classes=cls,
               exact_v=rec.get("v_exact_total"),
               lower=b["lower"], upper=b["upper"],
               pigeonhole=b["pigeonhole"], r_lin=b["r_lin"],
               estimate=est, blocks=rec.get("blocks", []),
               seconds=rec.get("seconds"))
    if verbose:
        print(f"  circuit        {out['name']}  n={out['n']} m={out['m']} "
              f"gates={out['gates']}")
        print(f"  classes        {cls}  ({out['components']} component(s))")
        if out["exact_v"] is not None:
            t = triple(out["n"], out["m"], out["exact_v"])
            out.update(t)
            print(f"  GARBAGE g      {t['g']}   (exact)")
            print(f"  lines L=m+g    {t['lines']}      ancilla a=L-n  {t['ancilla']}")
        else:
            print(f"  GARBAGE g      not established exactly; "
                  f"sound bounds {out['lower']} <= g <= {out['upper']}")
            print(f"  lines L=m+g    {out['m']+out['lower']} <= L <= {out['m']+out['upper']}")
        print(f"  sound bounds   affine-rank lower {out['lower']} "
              f"(r_lin={out['r_lin']}, pigeonhole would give {out['pigeonhole']}), "
              f"upper {out['upper']}")
        if est:
            print(f"  estimate       g ~ {est['second_order']}  "
                  f"[NOT A BOUND: independence assumption; on a one-hot "
                  f"decoder it excludes the true value]")
    return out


# --------------------------------------------------------------------------
# Cross-language consistency against the C front end
# --------------------------------------------------------------------------

C_BINS = {"quad": "vsim_quad", "cubic": "vsim_cubic", "brank": "vsim_brank"}


def _c_cp(binary, path, extra=None):
    exe = os.path.join(CDIR, binary)
    if not os.path.exists(exe):
        return None
    cmd = [exe, path] + ([str(extra)] if extra is not None else [])
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600).stdout
    for tok in out.split():
        if tok.startswith(("cp_symp=", "cp_cubic=", "cp_bounded=")):
            return float(tok.split("=", 1)[1])
    return None


_XCHECK = {"max_dev": 0.0, "exact": 0, "cases": 0}


def crosscheck(tol=1e-12, outfile=None):
    """Compare Python and C on the shipped reference sample set. Writes
    results/crosscheck_v8.json (max deviation, cases, exact agreements)."""
    _XCHECK.update(max_dev=0.0, exact=0, cases=0)
    samples = DATADIR
    n_ok = n_bad = n_skip = 0
    print(f"Cross-language consistency, tolerance {tol:g}")
    for p in sorted(glob.glob(os.path.join(samples, "quad", "*.v"))):
        nl = vfront.load_any(p)
        n, m, Br, L = symplectic.extract_forms(nl)
        evalf = _quad_eval(nl)
        py = symplectic.cp_symplectic(n, m, Br, L, evalf, evalf(0))
        c = _c_cp(C_BINS["quad"], p)
        n_ok, n_bad, n_skip = _report(p, py, c, tol, n_ok, n_bad, n_skip)
    for p in sorted(glob.glob(os.path.join(samples, "cubic", "*.v"))):
        if "deg4" in os.path.basename(p):
            continue                       # negative control, both sides wrong
        nl = vfront.load_any(p)
        fw, n, m = cubic_exact.truth_words(nl)
        py = cubic_exact.cp_cubic_exact(fw, n, m)
        c = _c_cp(C_BINS["cubic"], p)
        n_ok, n_bad, n_skip = _report(p, py, c, tol, n_ok, n_bad, n_skip)
    for p in sorted(glob.glob(os.path.join(samples, "brank", "*.v"))):
        base = os.path.basename(p)
        R = int(base.split("_R")[1].split("_")[0]) if "_R" in base else 4
        nl = vfront.load_any(p)
        py = bounded_rank.cp_bounded(nl, R)
        c = _c_cp(C_BINS["brank"], p, R)
        n_ok, n_bad, n_skip = _report(p, py, c, tol, n_ok, n_bad, n_skip)
    print(f"\n  {n_ok} agree, {n_bad} disagree, {n_skip} skipped")
    print(f"  max |python - c| = {_XCHECK['max_dev']:.3e} over {_XCHECK['cases']} cases, "
          f"{_XCHECK['exact']} identical in double precision")
    outfile = outfile or os.path.join(os.path.dirname(RESULTS_DEFAULT), "crosscheck_v8.json")
    if n_skip == 0 and _XCHECK["cases"] > 0:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        with open(outfile, "w") as fh:
            json.dump(dict(tol=tol, agree=n_ok, disagree=n_bad, skipped=n_skip,
                           max_dev=_XCHECK["max_dev"], exact=_XCHECK["exact"],
                           cases=_XCHECK["cases"]), fh, indent=1)
        print(f"  wrote {outfile}")
    return 0 if n_bad == 0 else 1


def _report(path, py, c, tol, n_ok, n_bad, n_skip):
    name = os.path.basename(path)
    if c is None:
        print(f"  {name:24s} SKIP (C binary absent; run make -C csrc)")
        return n_ok, n_bad, n_skip + 1
    d = abs(py - c)
    ok = d <= tol
    _XCHECK["cases"] += 1
    _XCHECK["max_dev"] = max(_XCHECK["max_dev"], d)
    _XCHECK["exact"] += (d == 0.0)
    print(f"  {name:24s} py={py:.12e} c={c:.12e} |d|={d:.2e} "
          f"{'OK' if ok else 'MISMATCH'}")
    return (n_ok + 1, n_bad, n_skip) if ok else (n_ok, n_bad + 1, n_skip)


# --------------------------------------------------------------------------

def smoke():
    """Fast path for build.sh --check."""
    print("[1/4] worked examples")
    demo()
    print("\n[2/4] quadratic spectral route against enumeration")
    p = os.path.join(DATADIR, "quad", "quad_10_6_t2.v")
    nl = vfront.load_any(p)
    n, m, Br, L = symplectic.extract_forms(nl)
    evalf = _quad_eval(nl)
    cp = symplectic.cp_symplectic(n, m, Br, L, evalf, evalf(0))
    cpb, vb = symplectic.brute_cp_v(nl)
    assert abs(cp - cpb) < 1e-12, (cp, cpb)
    print(f"  cp={cp:.12e} == brute {cpb:.12e}   v={vb}   OK")
    print("\n[3/4] classification of one netlist")
    analyze(os.path.join(DATADIR, "quad", "quad_12_8_t3.v"))
    print("\n[4/4] cross-language check on the quadratic reference set")
    rc = 0
    for q in sorted(glob.glob(os.path.join(DATADIR, "quad", "*.v")))[:3]:
        nl = vfront.load_any(q)
        n, m, Br, L = symplectic.extract_forms(nl)
        evalf = _quad_eval(nl)
        py = symplectic.cp_symplectic(n, m, Br, L, evalf, evalf(0))
        c = _c_cp("vsim_quad", q)
        if c is None:
            print(f"  {os.path.basename(q):24s} SKIP (build csrc first)")
        else:
            d = abs(py - c)
            print(f"  {os.path.basename(q):24s} |py - c| = {d:.2e} "
                  f"{'OK' if d <= 1e-12 else 'MISMATCH'}")
            rc |= 0 if d <= 1e-12 else 1
    print("\nsmoke check complete")
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--netlist")
    ap.add_argument("--crosscheck", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=48)
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--precision", type=int, default=12)
    ap.add_argument("--out", default=RESULTS_DEFAULT)
    a = ap.parse_args(argv)

    if a.check:
        return smoke()
    if a.demo:
        demo()
        return 0
    if a.crosscheck:
        return crosscheck(a.tol)
    if a.netlist:
        analyze(a.netlist, trials=a.trials)
        return 0
    if a.tables:
        run_tables(a.out)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
