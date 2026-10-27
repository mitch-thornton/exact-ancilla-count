# file:        scripts/bounded_rank.py
# project:     exact-ancilla-count (spectral computation of the line requirement for reversible embedding)
# brief:       Control-variable reduction (bounded control rank), exact at any algebraic degree
# purpose:     W_S as a sum over the 2^R control assignments of Arf-signed quadratic Walsh coefficients; generator of the control functions; exact degree from the control restriction
# inputs:      netlists or in-memory specifications passed by the caller
# outputs:     Python values returned to the caller (no files)
# invocation:  library module; imported by ancilla_v8.py and dispatch.py
# created:     bundle ancilla_spectral_bundle_v1, 2026-07-26 (extracted from the parent reversible-synthesis tool)
# modified:    bundle ancilla_spectral_bundle_v13, 2026-10-27
# history:     v1 2026-07-26 extracted from the parent tool; v8 2026-10-08 distinct=True option in build_control_fn and control_restricted_degree added;
#              v11 2026-10-26 license line simplified to plain MIT, no code change;
#              v13 2026-10-27 project line of the header reworded, no code change
# author:      Mitchell A. Thornton <mitch@smu.edu>, ORCID 0000-0003-3559-9511
# SPDX-License-Identifier: MIT
# (c) 2026 Mitchell A. Thornton
"""Bounded-rank scalable exact embedding cost, for ANY algebraic degree.

If a function's non-quadratic part is confined to R "control" linear forms (WLOG the
first R inputs), then fixing those R variables leaves a QUADRATIC in the remaining n-R.
Hence the (cubic/quartic/...) Walsh coefficient factors:
    W_T = sum_{y in GF(2)^R} signed_walsh( free -> f_T(y, free) ),
each inner term an EXACT quadratic Walsh (Arf).  So
    CP = 2^{-m} sum_T (W_T / 2^n)^2
is computed in 2^m * 2^R * poly(n) time -- poly in n, exponential only in the rank R and
output count m.  This makes bounded-trilinear-rank cubics AND bounded-rank quartics (and
higher) exactly and scalably solvable, generalizing the bent special case.

Contrast: a GENERIC degree-r function has no such structure; the derivative recursion
still gives exactness but nests r-2 direction sums, costing 2^{(r-2)n} -- the nesting
cost explodes with degree. Structure (bounded rank) is what keeps the ladder scalable.
"""
import sys, os, math, random, json, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import netlist
from netlist import Gate, Netlist
import net_tags as nt, arf, cubic as cubicmod, cubic_exact


def build_control_fn(n, m, R, ctrl_degree, seed=0, qterms=6, distinct=False):
    """f_j = Q_j(x) XOR H_j(x_0..x_{R-1}), Q_j quadratic in all n inputs, H_j an
    arbitrary function of the R controls with monomials up to ctrl_degree.

    distinct=False (v1 behavior, kept for the shipped reference netlists): the two
    control monomials of each degree are drawn independently, so they can coincide
    and cancel. With R == ctrl_degree they always do, which is why the v1 to v7
    scale rows had true degree 3 and not 4.
    distinct=True (v8 driver): the monomials of each degree are distinct subsets of
    the controls (one monomial when only one subset exists), so every output has
    algebraic degree exactly ctrl_degree whenever ctrl_degree >= 3.
    """
    rng = random.Random(seed); ins = [f"x{i}" for i in range(n)]; gates = []; outs = []; gc = [0]
    ctrls = ins[:R]
    def andk(vs):
        cur = vs[0]
        for v in vs[1:]:
            nm = f"a{gc[0]}"; gc[0] += 1; gates.append(Gate(nm, "AND", [cur, v])); cur = nm
        return cur
    def xorg(a, b):
        nm = f"c{gc[0]}"; gc[0] += 1; gates.append(Gate(nm, "XOR", [a, b])); return nm
    for j in range(m):
        acc = None
        # quadratic part over all n
        for i in range(n):
            if rng.random() < 0.3:
                acc = ins[i] if acc is None else xorg(acc, ins[i])
        for _ in range(qterms):
            a, b = rng.sample(ins, 2); t = andk([a, b])
            acc = t if acc is None else xorg(acc, t)
        # control part H_j: monomials of degree 2..ctrl_degree in the R controls
        for deg in range(2, ctrl_degree + 1):
            if R < deg:
                continue
            if distinct:
                subsets = list(itertools.combinations(ctrls, deg))
                chosen = rng.sample(subsets, min(2, len(subsets)))
            else:
                chosen = [tuple(rng.sample(ctrls, deg)) for _ in range(2)]
            for vs in chosen:
                t = andk(list(vs))
                acc = t if acc is None else xorg(acc, t)
        nm = f"y{j}"
        gates.append(Gate(nm, "BUF", [acc]) if acc else Gate(nm, "CONST0", [])); outs.append(nm)
    return Netlist(f"ctrl_n{n}_R{R}_d{ctrl_degree}", ins, outs, gates)


def W_T_bounded(f_eval, T_bits, n, R):
    """Exact Walsh coefficient of f_T via the control-coset factorization."""
    nf = n - R
    Wt = 0
    for y in range(1 << R):
        def evalq(free, _y=y):
            x = _y | (free << R)
            fv = f_eval(x)
            b = 0
            for j in T_bits:
                b ^= fv[j]
            return b
        q0 = evalq(0); qe = [evalq(1 << a) for a in range(nf)]
        Brows = [0] * nf
        for a in range(nf):
            for b in range(a + 1, nf):
                if q0 ^ qe[a] ^ qe[b] ^ evalq((1 << a) | (1 << b)):
                    Brows[a] |= (1 << b); Brows[b] |= (1 << a)
        Wt += arf.signed_walsh(evalq, Brows, nf)
    return Wt


def control_restricted_degree(nl, R):
    """Exact algebraic degree of every output of a bounded-control-rank function.

    For f_j = Q_j(x) + H_j(u) with Q_j quadratic and H_j a function of the R control
    inputs u, every monomial of degree three or more lives in H_j, so it survives
    when the n - R free inputs are set to zero. The Moebius transform of f_j(u, 0)
    over the 2^R control assignments therefore gives deg f_j exactly whenever that
    degree is at least 3. A value of at most 2 means f_j has degree at most 2 (the
    quadratic part of Q_j may still live in the free inputs, so the caller decides
    between 2 and lower). Costs 2^R simulations per output, independent of n.
    Returns the raw restricted degree of each output.
    """
    T = nt.NetTags(nl)
    degs = []
    for j in range(len(T.outputs)):
        a = []
        for u in range(1 << R):
            assign = {p: 0 for p in T.inputs}
            for k in range(R):
                assign[T.inputs[k]] = (u >> k) & 1
            a.append(netlist.simulate(nl, assign)[T.outputs[j]])
        N = 1 << R; step = 1
        while step < N:
            for i in range(0, N, 2 * step):
                for jj in range(i, i + step):
                    a[jj + step] ^= a[jj]
            step *= 2
        degs.append(max((bin(x).count("1") for x in range(N) if a[x]), default=0))
    return degs


def cp_bounded(nl, R):
    T = nt.NetTags(nl); n = T.n; m = T.m
    def f_eval(x):
        sv = netlist.simulate(nl, {p: (x >> k) & 1 for k, p in enumerate(T.inputs)})
        return tuple(sv[o] for o in T.outputs)
    cp = 0.0
    for Tset in range(1 << m):
        T_bits = [j for j in range(m) if (Tset >> j) & 1]
        Wt = W_T_bounded(f_eval, T_bits, n, R)
        cp += (Wt / (1 << n)) ** 2
    return cp * (2 ** (-m))


def main():
    print("BOUNDED-RANK exact CP for arbitrary degree (control part in R inputs)\n")
    print(f"{'case':<26}{'n':>4}{'m':>4}{'R':>3}{'deg':>5}{'CP_bounded':>14}{'CP_brute':>14}{'match':>7}{'s':>7}")
    rows = []
    for (n, m, R, cd) in [(12, 3, 4, 3), (12, 3, 4, 4), (14, 3, 5, 4), (14, 4, 4, 3)]:
        nl = build_control_fn(n, m, R, cd, seed=n + cd)
        deg = max(cubicmod.anf_degree(nl, j) for j in range(m))
        t0 = time.perf_counter(); cpb = cp_bounded(nl, R); dt = time.perf_counter() - t0
        fw, nn, mm = cubic_exact.truth_words(nl); cpt = cubic_exact.brute_cp(fw, nn)
        ok = "yes" if abs(cpb - cpt) < 1e-12 else "NO"
        rows.append(dict(n=n, m=m, R=R, ctrl_deg=cd, anf_deg=deg, cp_bounded=cpb, cp_brute=cpt, match=ok))
        print(f"{('control deg '+str(cd)+' (anf '+str(deg)+')'):<26}{n:>4}{m:>4}{R:>3}{deg:>5}"
              f"{cpb:>14.6e}{cpt:>14.6e}{ok:>7}{dt:>7.2f}")

    print("\nSCALE (brute = 2^n impossible): bounded-rank CP at large n, small R\n")
    for (n, m, R, cd) in [(40, 3, 4, 4), (56, 3, 4, 4), (72, 2, 4, 4)]:
        nl = build_control_fn(n, m, R, cd, seed=7)
        deg = max(cubicmod.anf_degree(nl, j) for j in range(m)) if n <= 14 else cd
        t0 = time.perf_counter(); cpb = cp_bounded(nl, R); dt = time.perf_counter() - t0
        v_lo = math.ceil(max(0, n + math.log2(cpb))) if cpb > 0 else 0
        print(f"  n={n:>3} m={m} R={R} ctrl_deg={cd}: CP={cpb:.4e}  v_lo={v_lo}  [{dt:.1f}s]  (brute needs 2^{n})")
    json.dump(rows, open("results/bounded_rank.json", "w"), indent=1)


if __name__ == "__main__":
    main()
