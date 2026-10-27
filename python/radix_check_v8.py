#!/usr/bin/env python3
# file:        scripts/radix_check_v8.py
# project:     exact-ancilla-count (spectral computation of the line requirement for reversible embedding)
# brief:       Numerical check of the radix-q (odd prime) extension stated in the paper
# purpose:     Verifies, on random instances, (i) the Chrestenson form of the collision
#              probability over F_q^m, (ii) the magnitude of the zero-frequency Chrestenson
#              coefficient of a quadratic parity (q^(n - rank/2) when the linear part lies in the
#              column space of the symmetric form, else 0), and (iii) the radix-q bracket and the
#              line/garbage/ancilla accounting against enumeration.
# inputs:      none (seeded random instances)
# outputs:     results/radix_check_v8.json (full run only; --check prints and writes nothing)
# invocation:  python3 scripts/radix_check_v8.py [--check]    (about 10 s; --check about 1 s)
# created:     bundle ancilla_spectral_bundle_v8, 2026-10-08
# modified:    bundle ancilla_spectral_bundle_v13, 2026-10-27
# history:     2026-10-08 created for v8 (new Section on radix-q specifications);
#              v11 2026-10-26 license line simplified to plain MIT, no code change;
#              v13 2026-10-27 project line of the header reworded, no code change
# author:      Mitchell A. Thornton <mitch@smu.edu>, ORCID 0000-0003-3559-9511
# SPDX-License-Identifier: MIT
# (c) 2026 Mitchell A. Thornton
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, math, itertools, random, cmath
from collections import Counter

CHECK = "--check" in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "radix_check_v8.json")
SEED = 20261008
rng = random.Random(SEED)
PRIMES = (3, 5, 7)


def rank_mod(rows, q):
    """Rank over F_q of a list of row vectors, and the reduced rows."""
    M = [list(r) for r in rows]
    r = 0
    ncol = len(M[0]) if M else 0
    for c in range(ncol):
        piv = next((i for i in range(r, len(M)) if M[i][c] % q), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = pow(M[r][c], q - 2, q)
        M[r] = [(x * inv) % q for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] % q:
                f = M[i][c]
                M[i] = [(x - f * y) % q for x, y in zip(M[i], M[r])]
        r += 1
    return r


def in_row_space(rows, v, q):
    return rank_mod(rows + [v], q) == rank_mod(rows, q)


def chrest0(vals, q):
    """Zero-frequency Chrestenson coefficient sum_x omega^{vals(x)}, as a complex number."""
    w = cmath.exp(2j * math.pi / q)
    return sum(w ** (t % q) for t in vals)


def random_quadratic(n, q, force_in_rowspace):
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            A[i][j] = rng.randrange(q)
            A[j][i] = A[i][j]
    if force_in_rowspace:
        z = [rng.randrange(q) for _ in range(n)]
        b = [sum(A[i][k] * z[k] for k in range(n)) % q for i in range(n)]
    else:
        b = [rng.randrange(q) for _ in range(n)]
    c = rng.randrange(q)
    return A, b, c


def quad_eval(A, b, c, x, q):
    n = len(x)
    s = c
    for i in range(n):
        s += b[i] * x[i]
        for j in range(n):
            s += A[i][j] * x[i] * x[j]
    return s % q


def check_quadratic_magnitude(trials):
    bad = 0
    for t in range(trials):
        q = rng.choice(PRIMES)
        n = rng.randrange(2, 5)
        A, b, c = random_quadratic(n, q, force_in_rowspace=(t % 3 == 0))
        pts = itertools.product(range(q), repeat=n)
        val = abs(chrest0([quad_eval(A, b, c, x, q) for x in pts], q))
        rk = rank_mod(A, q)
        pred = q ** (n - rk / 2) if in_row_space(A, b, q) else 0.0
        if abs(val - pred) > 1e-7:
            bad += 1
    return bad


def check_cp_identity(trials):
    """CP = q^-m sum_u |E omega^{u.f}|^2 for random multi-output f over F_q."""
    bad = 0
    worst = 0.0
    for _ in range(trials):
        q = rng.choice(PRIMES)
        n = rng.randrange(2, 4)
        m = rng.randrange(1, 3)
        pts = list(itertools.product(range(q), repeat=n))
        f = {x: tuple(rng.randrange(q) for _ in range(m)) for x in pts}
        cnt = Counter(f.values())
        cp = sum(v * v for v in cnt.values()) / q ** (2 * n)
        acc = 0.0
        for u in itertools.product(range(q), repeat=m):
            s = chrest0([sum(ui * fi for ui, fi in zip(u, f[x])) for x in pts], q)
            acc += abs(s) ** 2
        cp_spec = acc / q ** (m + 2 * n)
        d = abs(cp - cp_spec)
        worst = max(worst, d)
        if d > 1e-12:
            bad += 1
    return bad, worst


def check_bracket(trials):
    """g = ceil(log_q N_dup) lies in [n - H_2, n - H_2/2] (logs base q), and
    L = m + g >= n so a = L - n >= 0, on random radix-q functions."""
    bad = 0
    for _ in range(trials):
        q = rng.choice(PRIMES)
        n = rng.randrange(2, 4)
        m = rng.randrange(1, 3)
        pts = list(itertools.product(range(q), repeat=n))
        cnt = Counter(tuple(rng.randrange(q) for _ in range(m)) for _ in pts)
        ndup = max(cnt.values())
        g = math.ceil(math.log(ndup, q) - 1e-12) if ndup > 1 else 0
        cp = sum(v * v for v in cnt.values()) / q ** (2 * n)
        h2 = -math.log(cp, q)
        lo, hi = n - h2, n - h2 / 2
        if not (lo - 1e-9 <= math.log(ndup, q) <= hi + 1e-9) or m + g < n:
            bad += 1
    return bad


def main():
    T = 30 if CHECK else 300
    bad_q = check_quadratic_magnitude(T)
    bad_cp, worst = check_cp_identity(T // 3)
    bad_br = check_bracket(T // 3)
    res = dict(seed=SEED, primes=list(PRIMES), check_mode=CHECK,
               quadratic_trials=T, quadratic_mismatches=bad_q,
               cp_identity_trials=T // 3, cp_identity_mismatches=bad_cp, cp_identity_worst=worst,
               bracket_trials=T // 3, bracket_violations=bad_br)
    if not CHECK:   # the smoke path must not overwrite the shipped full-run results
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(res, open(OUT, "w"), indent=1)
    print(f"radix_check_v8: quadratic magnitude {T - bad_q}/{T} exact, "
          f"CP identity {T//3 - bad_cp}/{T//3} (worst {worst:.1e}), "
          f"bracket {T//3 - bad_br}/{T//3}  (q in {PRIMES})")
    assert bad_q == 0 and bad_cp == 0 and bad_br == 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
