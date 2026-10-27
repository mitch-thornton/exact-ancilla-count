# file:        csrc/Makefile
# project:     exact-ancilla-count (spectral computation of the line requirement for reversible embedding)
# brief:       Builds the five C tools of the front end
# purpose:     vsim, vsim_quad, vsim_cubic, vsim_brank, vsim_block from the sources in this directory; make test runs run_tests.sh
# inputs:      csrc/*.c, data/samples/
# outputs:     console report (and the five binaries for the Makefile)
# invocation:  make -C csrc [test|clean]
# created:     bundle ancilla_spectral_bundle_v1, 2026-07-26
# modified:    bundle ancilla_spectral_bundle_v13, 2026-10-27
# history:     v1 2026-07-26; carried unchanged into v8 2026-10-08 (header added);
#              v11 2026-10-26 license line simplified to plain MIT, no code change;
#              v13 2026-10-27 project line of the header reworded, no code change
# author:      Mitchell A. Thornton <mitch@smu.edu>, ORCID 0000-0003-3559-9511
# SPDX-License-Identifier: MIT
# (c) 2026 Mitchell A. Thornton
# ============================================================================
# Makefile for the ancilla-requirement C front end.
#
# This is the calculation front end only: netlist parsing, the VSIM net-tag
# structure, the Walsh/spectral kernel, and the exact ladder (affine, quadratic,
# cubic, bounded control rank, block decomposition). The synthesis back end of
# the parent tool is deliberately not included.
#
#     make            build vsim, vsim_quad, vsim_cubic, vsim_brank, vsim_block
#     make test       build, then run the C-only and cross-language checks
#     make clean
# ============================================================================

CC      ?= cc
STD      = -std=c99 -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE
WARN     = -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation
OPT      = -O3
LDLIBS   = -lm

UNAME_M := $(shell uname -m 2>/dev/null)
ifeq ($(UNAME_M),x86_64)
  MARCH := $(shell $(CC) -march=native -E -x c /dev/null >/dev/null 2>&1 && echo -march=native || echo -mtune=generic)
else ifeq ($(UNAME_M),aarch64)
  MARCH := $(shell $(CC) -march=native -E -x c /dev/null >/dev/null 2>&1 && echo -march=native || echo -mcpu=generic)
else ifeq ($(UNAME_M),arm64)
  MARCH := $(shell $(CC) -march=native -E -x c /dev/null >/dev/null 2>&1 && echo -march=native || echo -mcpu=generic)
else
  MARCH :=
endif

CFLAGS  = $(STD) $(WARN) $(OPT) $(MARCH) $(EXTRA_CFLAGS)

CORE = vsim.c vsim_tags.c vsim_spectral.c vsim_quad.c vsim_cubic.c \
       vsim_brank.c vsim_block.c \
       parse_verilog.c parse_isc.c parse_pla.c parse_aig.c
OBJ  = $(CORE:.c=.o)
HDR  = vsim.h
BINS = vsim vsim_quad vsim_cubic vsim_brank vsim_block

all: $(BINS)

%.o: %.c $(HDR)
	$(CC) $(CFLAGS) -c $< -o $@

vsim: $(OBJ) vsim_main.o
	$(CC) $(CFLAGS) -o $@ $(OBJ) vsim_main.o $(LDLIBS)

vsim_quad: $(OBJ) vsim_quad_main.o
	$(CC) $(CFLAGS) -o $@ $(OBJ) vsim_quad_main.o $(LDLIBS)

vsim_cubic: $(OBJ) vsim_cubic_main.o
	$(CC) $(CFLAGS) -o $@ $(OBJ) vsim_cubic_main.o $(LDLIBS)

vsim_brank: $(OBJ) vsim_brank_main.o
	$(CC) $(CFLAGS) -o $@ $(OBJ) vsim_brank_main.o $(LDLIBS)

vsim_block: $(OBJ) vsim_block_main.o
	$(CC) $(CFLAGS) -o $@ $(OBJ) vsim_block_main.o $(LDLIBS)

test: all
	bash run_tests.sh

clean:
	rm -f *.o $(BINS)

.PHONY: all test clean
