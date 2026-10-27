# file:        scripts/cubic_exact.py
# project:     exact-ancilla-count (spectral computation of the line requirement for reversible embedding)
# brief:       Exact collision probability of a cubic map by the direction sum
# purpose:     D_d f is quadratic for cubic f; averages the Arf-signed zero-probability over all 2^n directions and provides the enumeration reference brute_cp
# inputs:      netlists or in-memory specifications passed by the caller
# outputs:     Python values returned to the caller (no files)
# invocation:  library module
# created:     bundle ancilla_spectral_bundle_v1, 2026-07-26 (extracted from the parent reversible-synthesis tool)
# modified:    bundle ancilla_spectral_bundle_v13, 2026-10-27
# history:     v1 2026-07-26 extracted from the parent tool; carried unchanged into v8 2026-10-08;
#              v11 2026-10-26 license line simplified to plain MIT, no code change;
#              v13 2026-10-27 project line of the header reworded, no code change
# author:      Mitchell A. Thornton <mitch@smu.edu>, ORCID 0000-0003-3559-9511
# SPDX-License-Identifier: MIT
# (c) 2026 Mitchell A. Thornton
"""Exact CP for CUBIC functions via the derivative recursion + Arf-signed quadratic
point-count.  D_d f is quadratic; g(d) = P_x[D_d f(x)=0] is its zero-probability,
computed EXACTLY (no x-enumeration) by the signed-Walsh parity-subset sum:
    g(d) = 2^{-m-n} sum_{T subseteq [m]} W_{q_T},   q_T = XOR_{j in T} D_d f_j,
each W by arf.signed_walsh.  Then CP(f) = mean_d g(d).  Enumerating d is exact; sampling
d (with each g(d) exact) is an unbiased estimator whose variance is far below raw
collision sampling, because g(d) carries no x-sampling floor.
"""
import sys, os, math, random, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import netlist
import net_tags as nt, arf, cubic as cubicmod


def truth_words(nl):
    T = nt.NetTags(nl); n = T.n; m = T.m
    fw = []
    for x in range(1 << n):
        sv = netlist.simulate(nl, {p: (x >> k) & 1 for k, p in enumerate(T.inputs)})
        w = 0
        for j, o in enumerate(T.outputs):
            if sv[o]: w |= (1 << j)
        fw.append(w)
    return fw, n, m


def _forms_Dd(fw, n, m, d):
    def Dd(z): return fw[z] ^ fw[z ^ d]
    d0 = Dd(0); de = [Dd(1 << a) for a in range(n)]
    Brows = [[0] * n for _ in range(m)]
    for a in range(n):
        for b in range(a + 1, n):
            dab = Dd((1 << a) | (1 << b))
            for j in range(m):
                if ((d0 >> j) ^ (de[a] >> j) ^ (de[b] >> j) ^ (dab >> j)) & 1:
                    Brows[j][a] |= (1 << b); Brows[j][b] |= (1 << a)
    return Brows, Dd


def g_of_d(fw, n, m, d):
    Brows, Dd = _forms_Dd(fw, n, m, d)
    total = 0
    for Tset in range(1 << m):
        BT = [0] * n
        for j in range(m):
            if (Tset >> j) & 1:
                bj = Brows[j]
                for a in range(n):
                    BT[a] ^= bj[a]
        def qT(x, _T=Tset):
            return (Dd(x) & _T).bit_count() & 1
        total += arf.signed_walsh(qT, BT, n)
    return total / ((1 << m) * (1 << n))


def cp_cubic_exact(fw, n, m):
    return sum(g_of_d(fw, n, m, d) for d in range(1 << n)) / (1 << n)


def cp_cubic_sampled(fw, n, m, D, seed=0):
    rng = random.Random(seed)
    ds = [rng.getrandbits(n) for _ in range(D)]
    gs = [g_of_d(fw, n, m, d) for d in ds]
    return sum(gs) / len(gs), gs


def brute_cp(fw, n):
    from collections import Counter
    h = Counter(fw); N = 1 << n
    return sum(c * c for c in h.values()) / (N * N)


def main():
    print("EXACT CP for cubic functions via derivative recursion + Arf-signed count\n")
    print(f"{'cubic':<20}{'n':>4}{'m':>4}{'CP_exact(dirs)':>16}{'CP_brute':>14}{'match':>7}{'s':>7}")
    rows = []
    for seed in [11, 22, 33]:
        nl = cubicmod.build_deg(9, 3, 3, terms=4, seed=seed)
        fw, n, m = truth_words(nl)
        t0 = time.perf_counter(); cpe = cp_cubic_exact(fw, n, m); dt = time.perf_counter() - t0
        cpb = brute_cp(fw, n); ok = "yes" if abs(cpe - cpb) < 1e-12 else "NO"
        rows.append(dict(seed=seed, n=n, m=m, cp_exact=cpe, cp_brute=cpb, match=ok))
        print(f"{'cubic seed='+str(seed):<20}{n:>4}{m:>4}{cpe:>16.8e}{cpb:>14.8e}{ok:>7}{dt:>7.2f}")

    # also a structured cubic (with collapse) and a degree-4 (should FAIL: derivative is cubic)
    nl = cubicmod.build_structured_cubic(10, 4, seed=5)
    fw, n, m = truth_words(nl); cpe = cp_cubic_exact(fw, n, m); cpb = brute_cp(fw, n)
    print(f"{'structured cubic':<20}{n:>4}{m:>4}{cpe:>16.8e}{cpb:>14.8e}"
          f"{('yes' if abs(cpe-cpb)<1e-12 else 'NO'):>7}")
    nl4 = cubicmod.build_deg(9, 3, 4, terms=3, seed=7)
    fw4, n4, m4 = truth_words(nl4); cpe4 = cp_cubic_exact(fw4, n4, m4); cpb4 = brute_cp(fw4, n4)
    print(f"{'degree-4 (control)':<20}{n4:>4}{m4:>4}{cpe4:>16.8e}{cpb4:>14.8e}"
          f"{('yes' if abs(cpe4-cpb4)<1e-12 else 'NO'):>7}   <- expected NO (D_d is cubic, not quadratic)")

    # estimator: sample directions (exact g per direction) vs raw collision sampling
    print("\nVariance: sample D directions (exact g(d)) vs raw collision pairs, cubic n=12 m=4")
    nl = cubicmod.build_deg(12, 4, 3, terms=5, seed=42)
    fw, n, m = truth_words(nl); cp_true = brute_cp(fw, n)
    N = 1 << n
    for D in [16, 64, 256]:
        ests = []
        for rep in range(20):
            e, _ = cp_cubic_sampled(fw, n, m, D, seed=rep)
            ests.append(e)
        import statistics
        mean = statistics.mean(ests); rel = statistics.pstdev(ests) / cp_true
        # raw collision sampling with same # of function-pair evaluations (~D * 2^n / 2^n... use D*something)
        Sraw = D * 200
        raw_ests = []
        for rep in range(20):
            rng = random.Random(1000 + rep)
            c = sum(1 for _ in range(Sraw) if fw[rng.getrandbits(n)] == fw[rng.getrandbits(n)])
            raw_ests.append(c / Sraw)
        rel_raw = statistics.pstdev(raw_ests) / cp_true
        print(f"  D={D:>4}: direction-est mean={mean:.4e} relstd={rel:6.3f} | "
              f"raw({Sraw} pairs) relstd={rel_raw:6.3f}   (CP_true={cp_true:.4e})")
    json.dump(rows, open("results/cubic_exact.json", "w"), indent=1)


if __name__ == "__main__":
    main()
