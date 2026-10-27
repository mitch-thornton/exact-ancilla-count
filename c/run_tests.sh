#!/usr/bin/env bash
# file:        csrc/run_tests.sh
# project:     exact-ancilla-count (spectral computation of the line requirement for reversible embedding)
# brief:       C test suite with the cross-language step
# purpose:     Parser agreement, spectrum sanity, C against its own enumeration on the reference set, and C against the Python reference (ancilla_v8.py --crosscheck) at 1e-12
# inputs:      csrc/*.c, data/samples/
# outputs:     console report (and the five binaries for the Makefile)
# invocation:  bash csrc/run_tests.sh  (after make -C csrc); called by build.sh
# created:     bundle ancilla_spectral_bundle_v1, 2026-07-26
# modified:    bundle ancilla_spectral_bundle_v13, 2026-10-27
# history:     v1 2026-07-26; v8 2026-10-08 looks for ../scripts/ancilla_v8.py or ../python/ancilla_v8.py;
#              v11 2026-10-26 license line simplified to plain MIT, no code change;
#              v13 2026-10-27 project line of the header reworded, no code change
# author:      Mitchell A. Thornton <mitch@smu.edu>, ORCID 0000-0003-3559-9511
# SPDX-License-Identifier: MIT
# (c) 2026 Mitchell A. Thornton
# run_tests.sh -- checks for the ancilla-requirement C front end.
#
#   [1-3] C-only: parser agreement and spectrum sanity.
#   [4-7] C against its own brute-force enumeration on the reference set.
#   [8]   cross-language: C against the Python reference, tolerance 1e-12.
#
# Step 8 is skipped with a notice if python3 is unavailable; steps 1-7 need no
# Python at all. Exits non-zero on any failure.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
SAMP="../data/samples"
TOL="1e-12"

for b in vsim vsim_quad vsim_cubic vsim_brank vsim_block; do
  [ -x "./$b" ] || { echo "build first: make"; exit 1; }
done

fail=0
pass(){ echo "  PASS: $1"; }
bad(){  echo "  FAIL: $1"; fail=1; }

echo "[1] c17: the .v and .isc parsers must yield the identical function"
A=$(./vsim $SAMP/c17.v   --truth | awk '{print $NF}')
B=$(./vsim $SAMP/c17.isc --truth | awk '{print $NF}')
[ -n "$A" ] && [ "$A" = "$B" ] && pass "c17.v == c17.isc ($A)" || bad "c17.v ($A) != c17.isc ($B)"

echo "[2] ctrl: the Verilog and binary-AIGER parsers must agree"
A=$(./vsim $SAMP/ctrl.v   --truth | awk '{print $NF}')
B=$(./vsim $SAMP/ctrl.aig --truth | awk '{print $NF}')
[ -n "$A" ] && [ "$A" = "$B" ] && pass "ctrl.v == ctrl.aig ($A)" || bad "ctrl.v ($A) != ctrl.aig ($B)"

echo "[3] PLA parse plus spectrum: XOR cone affine (deg 1), AND cone deg 2"
S=$(./vsim $SAMP/xa.pla --spectrum)
echo "$S" | grep -q "xr : support=2 degree=1 affine=yes" && pass "xor cone affine deg 1" || bad "xor cone spectrum wrong"
echo "$S" | grep -q "an : support=2 degree=2 affine=no"  && pass "and cone deg 2"        || bad "and cone spectrum wrong"

echo "[4] quadratic ladder: spectral CP must equal enumeration"
for f in $SAMP/quad/*.v; do
  O=$(./vsim_quad "$f")
  echo "$O" | grep -q "match=yes" && pass "$(basename "$f")" || bad "$(basename "$f"): $O"
done

echo "[5] cubic ladder: spectral CP must equal enumeration (deg4 is a negative control)"
for f in $SAMP/cubic/*.v; do
  O=$(./vsim_cubic "$f")
  if [[ "$(basename "$f")" == deg4* ]]; then
    echo "$O" | grep -Eq "match=(no|deg>3)" && pass "$(basename "$f") disagrees as expected" \
      || bad "$(basename "$f") should NOT match: $O"
  else
    echo "$O" | grep -q "match=yes" && pass "$(basename "$f")" || bad "$(basename "$f"): $O"
  fi
done

echo "[6] bounded control rank: spectral CP must equal enumeration where enumeration runs"
for f in $SAMP/brank/*.v; do
  R=$(basename "$f" | sed -n 's/.*_R\([0-9]\+\)_.*/\1/p'); R=${R:-4}
  O=$(./vsim_brank "$f" "$R")
  if echo "$O" | grep -q "match=n/a"; then
    echo "$O" | grep -q "v_bracket=" && pass "$(basename "$f") (beyond enumeration; interval reported)" \
      || bad "$(basename "$f") reported no interval: $O"
  else
    echo "$O" | grep -q "match=yes" && pass "$(basename "$f")" || bad "$(basename "$f"): $O"
  fi
done

echo "[7] block decomposition: bracket must contain the enumerated value"
for f in $SAMP/block/*.v; do
  O=$(./vsim_block "$f")
  echo "$O" | grep -q "bracket=in" && pass "$(basename "$f")" || bad "$(basename "$f"): $O"
done

echo "[8] cross-language: C against the Python reference, tolerance $TOL"
if command -v python3 >/dev/null 2>&1; then
  DRV=""
  for cand in ../scripts/ancilla_v8.py ../python/ancilla_v8.py; do
    [ -f "$cand" ] && { DRV="$cand"; break; }
  done
  if [ -z "$DRV" ]; then
    echo "  SKIP: reference driver ancilla_v8.py not found next to this tree"
    rc=0
  else
    PYTHONHASHSEED=0 python3 "$DRV" --crosscheck --tol "$TOL" > /tmp/_xl.$$ 2>&1
    rc=$?
  fi
  if [ -n "$DRV" ]; then
    grep -E "MISMATCH|disagree" /tmp/_xl.$$ | tail -3
    tail -1 /tmp/_xl.$$
    [ $rc -eq 0 ] && pass "Python and C agree to $TOL on the reference set" \
                  || bad "cross-language mismatch (see output above)"
    rm -f /tmp/_xl.$$
  fi
else
  echo "  SKIP: python3 not found; steps 1-7 already passed without it"
fi

echo
[ $fail -eq 0 ] && echo "ALL C TESTS PASSED" || echo "FAILURES PRESENT"
exit $fail
