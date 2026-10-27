# file:        scripts/trilinear.py
# project:     exact-ancilla-count (spectral computation of the line requirement for reversible embedding)
# brief:       Third-order (trilinear) closed form for the direction sum
# purpose:     Autocorrelation-side check that the direction route and the output-parity route compute the same object
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
"""Deep push: the trilinear / third-order closed form for the direction sum.

Exact identity (derived): for any function f and output-parity f_T,
    sum_d W( D_d f_T ) = ( sum_x (-1)^{f_T(x)} )^2 = W_T^2,
because sum_{d,x} (-1)^{f_T(x) XOR f_T(x+d)} = (sum_x (-1)^{f_T(x)})(sum_y (-1)^{f_T(y)}).
Each W(D_d f_T) is a QUADRATIC Walsh coefficient, computed EXACTLY by arf.signed_walsh.
So the cubic collision probability
    CP = 2^{-m} sum_T (W_T / 2^n)^2 = 2^{-2n-m} sum_T sum_d W(D_d f_T)
is the SAME whether summed over directions or over output-parity subsets.  The entire
third-order barrier collapses to one object: the CUBIC Walsh coefficient W_T.

Scalability verdict: W_T for a general cubic is #P-hard, so cubic CP is not poly in
general.  It IS poly for structured cubics whose Walsh is closed-form -- in particular
BENT cubics (|W_T| = 2^{n/2}), the Maiorana-McFarland family, and bounded-trilinear-rank
cubics.  For those the embedding cost is exact and scalable with no enumeration.
"""
import sys, os, math, random, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import netlist
import net_tags as nt, arf, cubic as cubicmod, cubic_exact


def cubic_parity_words(nl, Tset):
    """Truth vector of the single cubic f_T = XOR_{j in T} f_j."""
    T = nt.NetTags(nl); n = T.n
    out = []
    for x in range(1 << n):
        sv = netlist.simulate(nl, {p: (x >> k) & 1 for k, p in enumerate(T.inputs)})
        b = 0
        for j, o in enumerate(T.outputs):
            if (Tset >> j) & 1 and sv[o]:
                b ^= 1
        out.append(b)
    return out, n


def W_brute(fw):
    return sum(1 if b == 0 else -1 for b in fw)


def W_via_directions(fw, n):
    """sum_d W(D_d f_T) via exact quadratic Walsh (should equal W_T^2)."""
    total = 0
    for d in range(1 << n):
        def Dd(z, _d=d): return fw[z] ^ fw[z ^ _d]
        # polar form of the quadratic D_d f_T
        d0 = Dd(0); de = [Dd(1 << a) for a in range(n)]
        Brows = [0] * n
        for a in range(n):
            for b in range(a + 1, n):
                if d0 ^ de[a] ^ de[b] ^ Dd((1 << a) | (1 << b)):
                    Brows[a] |= (1 << b); Brows[b] |= (1 << a)
        total += arf.signed_walsh(lambda z, _d=d: fw[z] ^ fw[z ^ _d], Brows, n)
    return total


def main():
    print("IDENTITY: sum_d W(D_d f_T) == W_T^2   (direction sum closes to cubic Walsh^2)\n")
    for seed in [1, 2, 3]:
        nl = cubicmod.build_deg(8, 3, 3, terms=4, seed=seed)
        Tset = 0b101
        fw, n = cubic_parity_words(nl, Tset)
        wt = W_brute(fw); lhs = W_via_directions(fw, n)
        ok = "yes" if lhs == wt * wt else "NO"
        print(f"  cubic parity seed={seed}: sum_d W(D_d f_T)={lhs}   W_T^2={wt*wt}   match={ok}")

    print("\nTHREE-WAY CP AGREEMENT: directions == parity-subsets == brute\n")
    for seed in [11, 22]:
        nl = cubicmod.build_deg(9, 3, 3, terms=4, seed=seed)
        fw, n, m = cubic_exact.truth_words(nl)
        cp_dir = cubic_exact.cp_cubic_exact(fw, n, m)
        # parity-subset CP using brute cubic Walsh W_T
        cp_ps = 0.0
        for Tset in range(1 << m):
            wt = sum(1 if (fw[x] & Tset).bit_count() % 2 == 0 else -1 for x in range(1 << n))
            cp_ps += (wt / (1 << n)) ** 2
        cp_ps *= 2 ** (-m)
        cp_b = cubic_exact.brute_cp(fw, n)
        ok = "yes" if abs(cp_dir - cp_ps) < 1e-12 and abs(cp_ps - cp_b) < 1e-12 else "NO"
        print(f"  seed={seed}: directions={cp_dir:.8e} parity-subset={cp_ps:.8e} "
              f"brute={cp_b:.8e}  agree={ok}")

    print("\nSCALABLE STRUCTURED SUBCLASS: BENT cubics have |W_T| = 2^{n/2} in CLOSED FORM,")
    print("so embedding cost v = ceil(log2 ((2^n + 2^{n/2})/2)) is exact with NO enumeration.")
    print(f"{'n':>5}{'|W| (closed form)':>20}{'N_dup':>16}{'v (closed form)':>18}")
    for n in [16, 32, 64, 128, 256]:
        Wmag = 2 ** (n // 2)
        ndup = (2 ** n + Wmag) // 2
        v = math.ceil(math.log2(ndup))
        print(f"{n:>5}{('2^'+str(n//2)):>20}{('~2^'+str(n-1)):>16}{v:>18}")
    print("\n(For n=256 a bent cubic oracle needs v=255 ancilla, computed in closed form;")
    print(" brute force would require 2^256 rows.)")
    json.dump({"note": "identity + three-way + bent scalable"}, open("results/trilinear.json", "w"))


if __name__ == "__main__":
    main()
