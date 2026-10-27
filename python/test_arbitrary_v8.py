#!/usr/bin/env python3
# file:        scripts/test_arbitrary_v8.py
# project:     exact-ancilla-count (spectral computation of the line requirement for reversible embedding)
# brief:       Regression suite for arbitrary netlists
# purpose:     Parser agreement on the EPFL twins, degenerate-parse refusal, sound bounds on every input, the dec counterexample for the independence estimate, and the router bound
# inputs:      netlists or in-memory specifications passed by the caller
# outputs:     console report
# invocation:  python3 scripts/test_arbitrary_v8.py  (about 20 s); called by build.sh
# created:     bundle ancilla_spectral_bundle_v1.1, 2026-07-26 (as test_arbitrary_v1.py)
# modified:    bundle ancilla_spectral_bundle_v13, 2026-10-27
# history:     v1.1 2026-07-26; v8 2026-10-08 renamed, imports ancilla_v8, garbage/line/ancilla vocabulary;
#              v11 2026-10-26 license line simplified to plain MIT, no code change;
#              v13 2026-10-27 project line of the header reworded, no code change
# author:      Mitchell A. Thornton <mitch@smu.edu>, ORCID 0000-0003-3559-9511
# SPDX-License-Identifier: MIT
# (c) 2026 Mitchell A. Thornton
"""test_arbitrary_v8.py -- the regression suite for arbitrary netlists.

    python3 scripts/test_arbitrary_v8.py

This suite exists because of a defect found after the first bundle was assembled.
The Python Verilog reader understood only structural ISCAS-style instantiations.
Given `assign`-style Verilog, which is what ABC and the EPFL suite emit, it
returned a netlist with zero or almost zero gates and no error, and the analysis
then reported a confident and wrong ancilla requirement. Of 21 real Verilog files
it handled exactly one. Every check below is a guard against that class of
failure returning.

Checks
------
  1  assign-style Verilog parses, and the gate count is not degenerate
  2  the Verilog and BLIF readers agree functionally on the same circuit
  3  the degenerate-parse guard fires rather than returning a broken netlist
  4  sequential input is refused, and the refusal is not triggered by a port
     whose name merely contains a keyword
  5  the sound bounds contain the enumerated truth on every small circuit
  6  the independence estimate is NOT a bound, pinned by a counterexample
  7  the affine-rank lower bound really can beat pigeonhole
  8  every shipped sample loads and analyzes without an exception
"""
import glob
import math
import os
import random
import sys
import tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import ancilla_v8 as core                    # noqa: E402
import netlist                               # noqa: E402
import net_tags as nt                        # noqa: E402
import verilog_front_v1 as vf                # noqa: E402

FAILED = []
EPFL = os.path.join(core.DATADIR, "epfl")


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILED.append(f"{name}: {detail}")
    return ok


def true_v(nl):
    T = nt.NetTags(nl)
    words = Counter()
    for x in range(1 << T.n):
        a = {pi: (x >> k) & 1 for k, pi in enumerate(T.inputs)}
        sv = netlist.simulate(nl, a)
        words[tuple(sv[o] for o in T.outputs)] += 1
    nd = max(words.values())
    return (math.ceil(math.log2(nd)) if nd > 1 else 0), nd


# ---------------------------------------------------------------- 1

def t1_assign_parses():
    print("\n[1] assign-style Verilog parses to a non-degenerate netlist")
    expect = {"ctrl": 585, "dec": 624, "int2float": 805,
              "cavlc": 2293, "router": 801}
    for name, gates in sorted(expect.items()):
        p = os.path.join(EPFL, name + ".v")
        if not os.path.exists(p):
            check(f"{name}.v present", False, "sample missing")
            continue
        try:
            nl = vf.load_verilog(p)
            check(f"{name}.v", nl.n_gates == gates,
                  f"got {nl.n_gates} gates, expected {gates}")
        except Exception as exc:                             # noqa: BLE001
            check(f"{name}.v", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- 2

def t2_verilog_matches_blif():
    print("\n[2] the Verilog and BLIF readers agree functionally")

    def norm(x):
        return x.replace("\\", "").replace("[", "_").replace("]", "_") \
                .replace("ESC_", "").strip("_")

    for p in sorted(glob.glob(os.path.join(EPFL, "*.v"))):
        b = p[:-2] + ".blif"
        if not os.path.exists(b):
            continue
        nv, nb = vf.load_verilog(p), netlist.parse_blif(b)
        mv = {norm(i): i for i in nv.inputs}
        mb = {norm(i): i for i in nb.inputs}
        if set(mv) != set(mb):
            check(os.path.basename(p), False, "port names differ between formats")
            continue
        rng = random.Random(0)
        ok = True
        for _ in range(32):
            bits = {k: rng.getrandbits(1) for k in mv}
            sv = netlist.simulate(nv, {mv[k]: v for k, v in bits.items()})
            sb = netlist.simulate(nb, {mb[k]: v for k, v in bits.items()})
            if tuple(sv[o] for o in nv.outputs) != tuple(sb[o] for o in nb.outputs):
                ok = False
                break
        check(os.path.basename(p), ok, "" if ok else "outputs differ on a random vector")


# ---------------------------------------------------------------- 3

def t3_guard_fires():
    print("\n[3] the degenerate-parse guard fires instead of returning a broken netlist")
    # A file in the old dialect-gap shape: declarations only, no logic. The old
    # reader returned this happily as a circuit of constants.
    src = """module top ( a, b, z );
      input a, b;
      output z;
      wire n1, n2;
    endmodule
    """
    with tempfile.NamedTemporaryFile("w", suffix=".v", delete=False) as fh:
        fh.write(src)
        p = fh.name
    try:
        vf.load_verilog(p)
        check("declarations-only file is rejected", False,
              "parser returned a netlist instead of raising")
    except vf.DegenerateParse as exc:
        check("declarations-only file is rejected", True, str(exc)[:60])
    except Exception as exc:                                 # noqa: BLE001
        check("declarations-only file is rejected", False,
              f"raised {type(exc).__name__}, wanted DegenerateParse")
    finally:
        os.unlink(p)

    # An output that no gate drives must also be caught.
    src2 = """module top ( a, b, z, q );
      input a, b;
      output z, q;
      assign z = a & b;
    endmodule
    """
    with tempfile.NamedTemporaryFile("w", suffix=".v", delete=False) as fh:
        fh.write(src2)
        p2 = fh.name
    try:
        vf.load_verilog(p2)
        check("undriven output is rejected", False, "parser returned a netlist")
    except vf.DegenerateParse:
        check("undriven output is rejected", True)
    except Exception as exc:                                 # noqa: BLE001
        check("undriven output is rejected", False, type(exc).__name__)
    finally:
        os.unlink(p2)


# ---------------------------------------------------------------- 4

def t4_sequential_refused():
    print("\n[4] sequential input refused; keyword inside a port name does not trip it")
    src = """module top ( clk, a, z );
      input clk, a;
      output z;
      reg z;
      always @(posedge clk) z <= a;
    endmodule
    """
    with tempfile.NamedTemporaryFile("w", suffix=".v", delete=False) as fh:
        fh.write(src)
        p = fh.name
    try:
        vf.load_verilog(p)
        check("sequential file refused", False, "parser accepted it")
    except vf.VerilogUnsupported:
        check("sequential file refused", True)
    except Exception as exc:                                 # noqa: BLE001
        check("sequential file refused", False, type(exc).__name__)
    finally:
        os.unlink(p)

    # `ctrl` has a port literally named reg_write. It must still parse.
    p = os.path.join(EPFL, "ctrl.v")
    if os.path.exists(p):
        try:
            nl = vf.load_verilog(p)
            check("port named reg_write does not trip the guard", nl.n_gates > 500,
                  f"{nl.n_gates} gates")
        except Exception as exc:                             # noqa: BLE001
            check("port named reg_write does not trip the guard", False,
                  f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- 5

def t5_bounds_contain_truth():
    print("\n[5] the sound bounds contain the enumerated truth")
    for p in sorted(glob.glob(os.path.join(EPFL, "*.v"))) + \
            [os.path.join(core.DATADIR, "c17.v")]:
        try:
            nl = vf.load_any(p)
        except Exception:                                    # noqa: BLE001
            continue
        T = nt.NetTags(nl)
        if T.n > 12:
            continue
        v, _nd = true_v(nl)
        b = core.sound_bounds(nl)
        ok = b["lower"] <= v <= b["upper"]
        check(f"{os.path.basename(p)} bounds [{b['lower']},{b['upper']}] "
              f"contain v={v}", ok)


# ---------------------------------------------------------------- 6

def t6_estimate_is_not_a_bound():
    print("\n[6] the independence estimate is NOT a bound (pinned counterexample)")
    p = os.path.join(EPFL, "dec.v")
    if not os.path.exists(p):
        check("dec.v counterexample present", False, "sample missing")
        return
    nl = vf.load_verilog(p)
    v, nd = true_v(nl)
    est = core.independence_estimate(nl)
    lo, hi = est["second_order"]
    check(f"dec is injective, N_dup={nd}, true v={v}", v == 0 and nd == 1)
    check(f"estimate {lo,hi} EXCLUDES the true v={v}, as documented",
          not (lo <= v <= hi),
          "if this now passes, the estimate changed and the label may need review")
    b = core.sound_bounds(nl)
    check(f"the sound bounds [{b['lower']},{b['upper']}] still contain it",
          b["lower"] <= v <= b["upper"])


# ---------------------------------------------------------------- 7

def t7_affine_beats_pigeonhole():
    print("\n[7] the affine-rank lower bound can beat pigeonhole on real circuits")
    p = os.path.join(EPFL, "router.v")
    if not os.path.exists(p):
        check("router.v present", False, "sample missing")
        return
    nl = vf.load_verilog(p)
    b = core.sound_bounds(nl)
    check(f"router: affine lower {b['lower']} > pigeonhole {b['pigeonhole']}",
          b["lower"] > b["pigeonhole"],
          f"r_lin={b['r_lin']} of m={b['m']}")


# ---------------------------------------------------------------- 8

def t8_all_samples_analyze():
    print("\n[8] every shipped sample loads and analyzes without an exception")
    pats = ["*.v", "*.isc", "*.pla", "epfl/*.v", "epfl/*.blif",
            "quad/*.v", "cubic/*.v", "brank/*.v", "block/*.v"]
    n_ok = n_bad = 0
    for pat in pats:
        for p in sorted(glob.glob(os.path.join(core.DATADIR, pat))):
            try:
                core.analyze(p, verbose=False, estimate=False)
                n_ok += 1
            except Exception as exc:                         # noqa: BLE001
                n_bad += 1
                check(os.path.basename(p), False, f"{type(exc).__name__}: {exc}")
    check(f"{n_ok} samples analyzed, {n_bad} failed", n_bad == 0)

    print("\n[9] unsupported formats are refused with a usable message")
    aig = os.path.join(core.DATADIR, "ctrl.aig")
    if os.path.exists(aig):
        try:
            vf.load_any(aig)
            check("binary AIGER refused", False, "it was accepted")
        except vf.VerilogUnsupported as exc:
            check("binary AIGER refused with a pointer to the C front end",
                  "csrc/vsim" in str(exc), str(exc)[:70])
        except Exception as exc:                             # noqa: BLE001
            check("binary AIGER refused", False,
                  f"raised {type(exc).__name__}, wanted VerilogUnsupported")


def main():
    print("Regression suite for arbitrary netlists")
    for fn in (t1_assign_parses, t2_verilog_matches_blif, t3_guard_fires,
               t4_sequential_refused, t5_bounds_contain_truth,
               t6_estimate_is_not_a_bound, t7_affine_beats_pigeonhole,
               t8_all_samples_analyze):
        fn()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILURE(S):")
        for f in FAILED:
            print("  " + f)
        return 1
    print("ALL ARBITRARY-NETLIST CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
