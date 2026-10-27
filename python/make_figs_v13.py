#!/usr/bin/env python3
# file:        scripts/make_figs_v13.py
# project:     exact-ancilla-count (spectral computation of the line requirement for reversible embedding)
# brief:       Regenerates both figures of the bundle from results/ancilla_results_v8.json
# purpose:     Fig. 1 (reach) and the order-profile figure, each with its CSV of plotted values
# inputs:      results/ancilla_results_v8.json
# outputs:     figures/fig_reach.{pdf,png,csv}, figures/fig_order.{pdf,png,csv}
# invocation:  python3 scripts/make_figs_v13.py [--quick] [--stacked] [--outdir figures]
# created:     bundle ancilla_spectral_bundle_v1, 2026-07-26 (as make_figs_v1.py)
# modified:    bundle ancilla_spectral_bundle_v13, 2026-10-27
# history:     v7 2026-08-12 annotation and axis changes; v8 2026-10-08 renamed, reads results/,
#              legends moved below the axes, quadratic points use measured times,
#              Fig. 1 panels side by side (--stacked keeps the v7 arrangement);
#              v11 2026-10-26 license line simplified to plain MIT, no code change;
#              v13 2026-10-27 renamed; fonts embedded as TrueType (Type 42) instead of Type 3
#              and the family pinned to DejaVu Sans, so the PDFs carry no Type 3 font; the
#              project line of the header reworded; no change to what is drawn
# author:      Mitchell A. Thornton <mitch@smu.edu>, ORCID 0000-0003-3559-9511
# SPDX-License-Identifier: MIT
# (c) 2026 Mitchell A. Thornton
"""make_figs_v13.py -- regenerate both figures from results/ancilla_results_v8.json.

    python3 scripts/make_figs_v13.py [--quick] [--stacked] [--outdir figures]

v13 embeds the figure text as TrueType (Type 42) fonts in the PDFs, pinned to DejaVu
Sans, which ships with matplotlib; earlier versions let matplotlib write Type 3 fonts,
which publisher PDF checks query. Nothing drawn has changed.

Reads results/ancilla_results_v8.json (produced by `python3 scripts/ancilla_v8.py --tables`) and
writes, into figures/:

    fig_reach.{pdf,png}    where the spectral route runs and enumeration does not
    fig_order.{pdf,png}    order profile as a tractability diagnostic

Both figures also emit their plotted values as CSV alongside the image, so the
figure content is checkable without rerunning matplotlib.

--quick is accepted for the smoke path of build.sh and changes nothing, since
both figures are drawn from the results file in under a second. --stacked keeps
the v7 arrangement of Fig. 1 (panels one above the other) for comparison.
"""
import argparse
import csv
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
# TrueType (Type 42) embedding and a font that ships with matplotlib, so the PDF figures
# carry no Type 3 font and look the same on every machine.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results", "ancilla_results_v8.json")

INK = "#1a1a2e"
ACC = "#0f4c81"
WARM = "#b45309"
MUTED = "#6b7280"


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(True, which="major", axis="y", color="#e5e7eb", lw=0.6)
    ax.set_axisbelow(True)


def fig_reach(res, outdir, stacked=False):
    """Rows the method answers, against the size of the truth table it avoids.

    Default layout (v8): the two panels side by side at full column width, so
    the figure is 1.65 in tall on the page and every label prints at its
    native size.  --stacked restores the v7 arrangement (panels one above the
    other, shared x axis) for comparison.
    """
    pts = []
    for r in res.get("affine", []):
        pts.append(("affine", r["n"], r["seconds"]))
    for r in res.get("bounded_rank", []):
        pts.append(("bounded control rank", r["n"], r["seconds"]))
    for r in res.get("quadratic", []):
        if "v_brute" in r and "seconds" in r:
            pts.append(("quadratic", r["n"], r["seconds"]))

    if stacked:
        fig, (axa, axb) = plt.subplots(2, 1, figsize=(3.4, 3.55), dpi=200,
                                       sharex=True,
                                       gridspec_kw={"height_ratios": [1, 1.15]})
    else:
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(3.5, 1.65), dpi=200)
    ns = sorted({p[1] for p in pts})
    xs = list(range(2, max(ns) + 1))
    axa.plot(xs, xs, color=MUTED, lw=1.1, ls="--")
    axa.set_ylabel("truth-table rows,\n$\\log_2$", fontsize=7.5, color=INK)
    if stacked:
        axa.text(0.60, 0.45, "enumeration wall", transform=axa.transAxes,
                 fontsize=6.4, color=MUTED, rotation=27)
        axa.annotate("practical limit\nof enumeration", xy=(10, 20),
                     xytext=(0.10, 0.62), textcoords="axes fraction",
                     fontsize=6.0, color=WARM, ha="left",
                     arrowprops=dict(arrowstyle="->", color=WARM, lw=0.7))
    else:
        axa.text(0.56, 0.30, "enumeration\nwall", transform=axa.transAxes,
                 fontsize=6.0, color=MUTED, ha="left")
        axa.annotate("practical limit\nof enumeration", xy=(10, 20),
                     xytext=(0.10, 0.74), textcoords="axes fraction",
                     fontsize=6.0, color=WARM, ha="left",
                     arrowprops=dict(arrowstyle="->", color=WARM, lw=0.7))
    axa.axvline(10, color=WARM, lw=0.8, ls=":")
    axa.tick_params(labelbottom=True)
    axa.set_xlabel("number of inputs $n$", fontsize=8, color=INK)
    _style(axa)

    styles = {"affine": (ACC, "o"), "bounded control rank": (WARM, "s"),
              "quadratic": ("#1a7f37", "^")}
    seen = set()
    for cls, n, secs in pts:
        col, mk = styles[cls]
        axb.semilogy([n], [max(secs, 1e-3)], mk, color=col, ms=4.5, alpha=0.9,
                     label=cls if cls not in seen else None)
        seen.add(cls)
    axb.axvline(10, color=WARM, lw=0.8, ls=":")
    axb.set_xlabel("number of inputs $n$", fontsize=8, color=INK)
    axb.set_ylabel("this work, seconds\n(Python reference)", fontsize=7.5, color=INK)
    # legend below the axes, outside the plot box
    if stacked:
        axb.legend(fontsize=6.2, frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, -0.42), ncol=3, handletextpad=0.3,
                   columnspacing=1.0)
    _style(axb)
    if stacked:
        fig.tight_layout(pad=0.4)
    else:
        # one legend row under both panels, outside the plot boxes
        hs, ls = axb.get_legend_handles_labels()
        fig.legend(hs, ls, fontsize=6.2, frameon=False, loc="lower center",
                   bbox_to_anchor=(0.5, -0.01), ncol=3, handletextpad=0.3,
                   columnspacing=1.2)
        fig.tight_layout(pad=0.4, w_pad=1.4, rect=(0, 0.10, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_reach.{ext}"))
    plt.close(fig)

    with open(os.path.join(outdir, "fig_reach.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "n", "seconds_python", "enumeration_rows_2^n"])
        for cls, n, secs in sorted(pts, key=lambda t: (t[0], t[1])):
            w.writerow([cls, n, secs, f"2^{n}"])
    return len(pts)


def fig_order(res, outdir):
    """Cumulative collision mass by spectral order: the tractability diagnostic."""
    prof = res.get("order_profile", [])
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=200)
    cols = [ACC, WARM, "#1a7f37", "#7c3aed", MUTED]
    rows_csv = []
    for i, entry in enumerate(prof):
        cum = entry["profile"]["cum_frac"]
        ks = list(range(len(cum)))
        ax.plot(ks, [100.0 * c for c in cum], marker="o", ms=3, lw=1.2,
                color=cols[i % len(cols)], label=entry["label"])
        rows_csv.append([entry["label"]] + [round(100.0 * c, 3) for c in cum])
    ax.axhline(90.0, color="#9ca3af", lw=0.8, ls=":")
    ax.text(0.60, 84.0, "90 percent of the mass", fontsize=6, color=MUTED)
    ax.set_xlabel("spectral order $k$ (output-subset size)", fontsize=8, color=INK)
    ax.set_ylabel("cumulative percent of $\\mathrm{CP}$", fontsize=8, color=INK)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=6.2, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2)
    _style(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_order.{ext}"))
    plt.close(fig)

    with open(os.path.join(outdir, "fig_order.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        maxk = max(len(r) - 1 for r in rows_csv) if rows_csv else 0
        w.writerow(["function"] + [f"cum_pct_k{k}" for k in range(maxk)])
        w.writerows(rows_csv)
    return len(prof)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "figures"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stacked", action="store_true",
                    help="v7 layout for Fig. 1 (panels stacked) instead of side by side")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    if not os.path.exists(RESULTS):
        print(f"missing {RESULTS}; run: python3 scripts/ancilla_v8.py --tables",
              file=sys.stderr)
        return 1
    res = json.load(open(RESULTS))
    n1 = fig_reach(res, a.outdir, stacked=a.stacked)
    n2 = fig_order(res, a.outdir)
    print(f"wrote figures/fig_reach.{{pdf,png,csv}} ({n1} points)")
    print(f"wrote figures/fig_order.{{pdf,png,csv}} ({n2} series)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
