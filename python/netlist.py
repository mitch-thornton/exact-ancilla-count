# file:        scripts/netlist.py
# project:     exact-ancilla-count (spectral computation of the line requirement for reversible embedding)
# brief:       Gate-level netlist representation with ISCAS structural Verilog and BLIF parsers
# purpose:     Netlist and Gate classes, simulation, and the two original parsers; the dialect-complete readers are in verilog_front_v1.py
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
"""Gate-level netlist IR + parsers for ISCAS85 structural Verilog and BLIF (EPFL)."""
import re
from collections import defaultdict


class Gate:
    __slots__ = ("out", "func", "ins", "cubes")

    def __init__(self, out, func, ins, cubes=None):
        self.out = out          # output net name
        self.func = func        # AND OR NAND NOR XOR XNOR NOT BUF CONST0 CONST1 LUT
        self.ins = ins          # list of input net names
        self.cubes = cubes      # for LUT: list of (cube_str, out_val) rows; cube chars 0/1/-

    def __repr__(self):
        return f"{self.out} = {self.func}({', '.join(self.ins)})"


class Netlist:
    def __init__(self, name, inputs, outputs, gates):
        self.name = name
        self.inputs = inputs      # list of net names
        self.outputs = outputs    # list of net names
        self.gates = gates        # list of Gate, arbitrary order
        self._topo = None

    @property
    def n_gates(self):
        return len(self.gates)

    def topo_gates(self):
        """Gates in topological order (inputs first)."""
        if self._topo is not None:
            return self._topo
        gate_of = {g.out: g for g in self.gates}
        state = {}   # net -> 0 visiting, 1 done
        order = []
        for net in self.inputs:
            state[net] = 1
        # iterative DFS
        for g in self.gates:
            if state.get(g.out) == 1:
                continue
            stack = [(g.out, 0)]
            while stack:
                net, idx = stack.pop()
                if state.get(net) == 1:
                    continue
                gg = gate_of.get(net)
                if gg is None:      # dangling input treated as PI
                    state[net] = 1
                    continue
                if idx == 0:
                    state[net] = 0
                deps = gg.ins
                if idx < len(deps):
                    stack.append((net, idx + 1))
                    d = deps[idx]
                    if state.get(d) != 1:
                        if state.get(d) == 0:
                            raise ValueError(f"combinational loop at {d}")
                        stack.append((d, 0))
                else:
                    state[net] = 1
                    order.append(gg)
        self._topo = order
        return order

    def levelize(self):
        """net -> level (PIs at 0); gate level = 1 + max(input levels)."""
        lvl = {net: 0 for net in self.inputs}
        for g in self.topo_gates():
            lvl[g.out] = 1 + max((lvl.get(i, 0) for i in g.ins), default=0)
        return lvl

    def stats(self):
        lvl = self.levelize()
        return dict(name=self.name, inputs=len(self.inputs), outputs=len(self.outputs),
                    gates=self.n_gates, depth=max((lvl[o] for o in self.outputs), default=0))


# ---------------- ISCAS85 structural Verilog ----------------

_VPRIMS = {"and", "or", "nand", "nor", "xor", "xnor", "not", "buf"}


def parse_iscas_verilog(path, name=None):
    text = open(path).read()
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    stmts = [s.strip() for s in text.replace("\n", " ").split(";")]
    inputs, outputs, gates = [], [], []
    for s in stmts:
        if not s:
            continue
        m = re.match(r"^(input|output|wire)\s+(.*)$", s)
        if m:
            kind, rest = m.group(1), m.group(2)
            nets = [n.strip() for n in rest.split(",") if n.strip()]
            if kind == "input":
                inputs += nets
            elif kind == "output":
                outputs += nets
            continue
        m = re.match(r"^(\w+)\s+\w+\s*\(([^)]*)\)$", s)
        if m and m.group(1).lower() in _VPRIMS:
            func = m.group(1).lower()
            ports = [p.strip() for p in m.group(2).split(",")]
            out, ins = ports[0], ports[1:]
            fmap = {"and": "AND", "or": "OR", "nand": "NAND", "nor": "NOR",
                    "xor": "XOR", "xnor": "XNOR", "not": "NOT", "buf": "BUF"}
            gates.append(Gate(out, fmap[func], ins))
    return Netlist(name or path.split("/")[-1].split(".")[0], inputs, outputs, gates)


# ---------------- BLIF ----------------

def parse_blif(path, name=None):
    lines = []
    for raw in open(path):
        raw = raw.split("#")[0].rstrip()
        if not raw:
            continue
        if lines and lines[-1].endswith("\\"):
            lines[-1] = lines[-1][:-1] + " " + raw.strip()
        else:
            lines.append(raw)
    inputs, outputs, gates = [], [], []
    i = 0
    while i < len(lines):
        ln = lines[i]
        tok = ln.split()
        if tok[0] == ".inputs":
            inputs += tok[1:]
        elif tok[0] == ".outputs":
            outputs += tok[1:]
        elif tok[0] == ".names":
            nets = tok[1:]
            out, ins = nets[-1], nets[:-1]
            cubes = []
            j = i + 1
            while j < len(lines) and not lines[j].startswith("."):
                parts = lines[j].split()
                if len(parts) == 2:
                    cubes.append((parts[0], parts[1]))
                elif len(parts) == 1 and not ins:   # constant
                    cubes.append(("", parts[0]))
                j += 1
            i = j - 1
            if not ins:
                val = cubes[0][1] if cubes else "0"
                gates.append(Gate(out, "CONST1" if val == "1" else "CONST0", []))
            else:
                gates.append(Gate(out, "LUT", ins, cubes))
        elif tok[0] in (".model", ".end", ".latch"):
            if tok[0] == ".latch":
                raise ValueError("sequential BLIF not supported")
        i += 1
    return Netlist(name or path.split("/")[-1].split(".")[0], inputs, outputs, gates)


def load(path):
    if path.endswith(".blif"):
        return parse_blif(path)
    return parse_iscas_verilog(path)


# ---------------- reference simulation ----------------

def simulate(nl, assignment):
    """assignment: dict input->0/1. Returns dict of all net values."""
    val = dict(assignment)
    for g in nl.topo_gates():
        vs = [val[i] for i in g.ins]
        f = g.func
        if f == "AND":
            v = int(all(vs))
        elif f == "OR":
            v = int(any(vs))
        elif f == "NAND":
            v = int(not all(vs))
        elif f == "NOR":
            v = int(not any(vs))
        elif f == "XOR":
            v = sum(vs) % 2
        elif f == "XNOR":
            v = (sum(vs) + 1) % 2
        elif f == "NOT":
            v = 1 - vs[0]
        elif f == "BUF":
            v = vs[0]
        elif f == "CONST0":
            v = 0
        elif f == "CONST1":
            v = 1
        elif f == "LUT":
            pol = int(g.cubes[0][1]) if g.cubes else 1   # cover polarity (uniform per BLIF)
            v = 1 - pol
            for cube, ov in g.cubes:
                if all(c == "-" or int(c) == vs[k] for k, c in enumerate(cube)):
                    v = pol
                    break
        else:
            raise ValueError(f)
        val[g.out] = v
    return val
