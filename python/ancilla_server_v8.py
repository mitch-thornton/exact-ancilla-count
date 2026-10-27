#!/usr/bin/env python3
# file:        scripts/ancilla_server_v8.py
# project:     exact-ancilla-count (spectral computation of the line requirement for reversible embedding)
# brief:       Local browser front end for the line-requirement tool
# purpose:     Serves ancilla.html and runs the analysis on an uploaded netlist, reporting g, L and a with the class and audit (browser for presentation, Python for computation)
# inputs:      netlists or in-memory specifications passed by the caller
# outputs:     console report
# invocation:  python3 scripts/ancilla_server_v8.py  then open http://localhost:8767
# created:     bundle ancilla_spectral_bundle_v1, 2026-07-26 (as ancilla_server_v1.py)
# modified:    bundle ancilla_spectral_bundle_v13, 2026-10-27
# history:     v1 2026-07-26; v8 2026-10-08 renamed, imports ancilla_v8, reports garbage g, lines L and ancilla a;
#              v11 2026-10-26 license line simplified to plain MIT, no code change;
#              v13 2026-10-27 project line of the header reworded, no code change
# author:      Mitchell A. Thornton <mitch@smu.edu>, ORCID 0000-0003-3559-9511
# SPDX-License-Identifier: MIT
# (c) 2026 Mitchell A. Thornton
"""ancilla_server_v8.py -- local web front end for the line-requirement tool.

    python3 scripts/ancilla_server_v8.py [port]

Then open http://localhost:8767 in a browser. Do NOT open ancilla.html directly
as a file; the page must be served so its requests reach this process. Binds
127.0.0.1 only. Dependencies: python3 standard library only.

Endpoints
---------
    GET  /            ancilla.html
    GET  /samples     {"samples": [...]}   the bundled reference netlists
    POST /analyze     {"sample": name} or {"filename":..., "content":...}
                      -> the analysis record consumed by the page

This follows the project convention for local user interfaces: the browser does
presentation, the Python reference implementation does the computation. Times
reported are Python reference times, not performance results.
"""
import json
import math
import os
import sys
import tempfile
import time
import traceback
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ancilla_v8 as core                    # noqa: E402
import dispatch as dispatcher                # noqa: E402
import net_tags as nt                        # noqa: E402
import netlist                               # noqa: E402
import gf2subset                             # noqa: E402

import ancilla_v8 as _core_paths          # noqa: E402
sys.path.insert(0, _core_paths.LIBDIR)

HTML_PATH = os.path.join(HERE, "ancilla.html")
SAMPLE_ROOT = _core_paths.DATADIR
DEFAULT_PORT = 8767


def list_samples():
    out = []
    for root, _dirs, files in os.walk(SAMPLE_ROOT):
        for f in sorted(files):
            if f.endswith((".v", ".isc", ".pla", ".aig")):
                out.append(os.path.relpath(os.path.join(root, f), SAMPLE_ROOT))
    return sorted(out)


import verilog_front_v1 as vfront            # noqa: E402


def _load(path):
    """Every format the Python front end supports, with the guard applied."""
    return vfront.load_any(path)


def analyze_path(path, trials=48):
    t0 = time.perf_counter()
    nl = _load(path)
    rec = dispatcher.dispatch(nl, trials=trials, verbose=False)
    T = nt.NetTags(nl)

    comps = []
    for c in rec.get("blocks", []):
        chk = c.get("brute_check")
        comps.append(dict(
            n=c.get("inputs"), m=c.get("outputs"), cls=c.get("cls", "?"),
            v=c.get("v"), audit=(None if chk is None else (chk == "OK"))))
    classes = "+".join(sorted({c["cls"] for c in comps})) or "none"

    # Certified interval and the affine lower bound, always reported.
    cp = h2 = v_lo = v_hi = None
    try:
        import random
        words = gf2subset.sample_words(T, max(256, T.n + 64), random.Random(0))
        r_lin = gf2subset.affine_rank(words)
        affine_bound = T.n - r_lin
    except Exception:                                       # noqa: BLE001
        affine_bound = None
    if T.n <= 20:
        hist, n, _m = core.fibres(nl)
        cp = core.cp_from_fibres(hist, n)
        h2 = -math.log2(cp)
        v_lo, v_hi = core.v_bracket(cp, n)

    src = ""
    try:
        with open(path, "r", errors="replace") as fh:
            src = fh.read(20000)
    except Exception:                                       # noqa: BLE001
        pass

    b = core.sound_bounds(nl)
    est = core.independence_estimate(nl)
    return dict(
        name=rec.get("name", os.path.basename(path)),
        n=T.n, m=T.m, gates=nl.n_gates,
        sound_lower=b["lower"], sound_upper=b["upper"],
        pigeonhole=b["pigeonhole"], r_lin=b["r_lin"],
        estimate=(list(est["second_order"]) if est else None),
        classes=classes,
        components=comps,
        v_exact=rec.get("v_exact_total"),
        v_lo=v_lo, v_hi=v_hi, cp=cp, h2=h2,
        affine_bound=affine_bound,
        trials=trials,
        seconds=round(time.perf_counter() - t0, 3),
        source=src,
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "ancilla/8.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path).path
        if p in ("/", "/index.html", "/ancilla.html"):
            with open(HTML_PATH, "rb") as fh:
                return self._send(200, fh.read(), "text/html; charset=utf-8")
        if p == "/samples":
            return self._send(200, json.dumps({"samples": list_samples()}))
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if urlparse(self.path).path != "/analyze":
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
            if req.get("sample"):
                path = os.path.join(SAMPLE_ROOT, req["sample"])
                if not os.path.abspath(path).startswith(os.path.abspath(SAMPLE_ROOT)):
                    return self._send(400, json.dumps({"error": "bad sample path"}))
                out = analyze_path(path)
            else:
                name = req.get("filename", "upload.v")
                suffix = os.path.splitext(name)[1] or ".v"
                with tempfile.NamedTemporaryFile("w", suffix=suffix,
                                                 delete=False) as fh:
                    fh.write(req.get("content", ""))
                    tmp = fh.name
                try:
                    out = analyze_path(tmp)
                    out["name"] = name
                finally:
                    os.unlink(tmp)
            return self._send(200, json.dumps(out))
        except Exception as exc:                            # noqa: BLE001
            return self._send(200, json.dumps(
                {"error": f"{type(exc).__name__}: {exc}\n\n"
                          + traceback.format_exc(limit=4)}))


def main(argv):
    port = int(argv[1]) if len(argv) > 1 else DEFAULT_PORT
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"ancilla front end on http://localhost:{port}   (ctrl-C to stop)")
    print(f"  samples: {len(list_samples())} netlists under data/samples/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
