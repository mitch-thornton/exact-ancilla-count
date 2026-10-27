/*
 * file:        csrc/vsim_block_main.c
 * project:     exact-ancilla-count (spectral computation of the line requirement for reversible embedding)
 * brief:       Driver: vsim_block <file> [small_limit]
 * purpose:     C performance path of the ladder; held to agree with the Python reference to 1e-12 (csrc/run_tests.sh step 8)
 * inputs:      netlist files under data/samples/ (see each driver's usage line)
 * outputs:     console report
 * invocation:  make -C csrc && make -C csrc test
 * created:     bundle ancilla_spectral_bundle_v1, 2026-07-26 (extracted from the parent tool)
 * modified:    bundle ancilla_spectral_bundle_v13, 2026-10-27
 * history:     v1 2026-07-26; carried unchanged into v8 2026-10-08 (header added);
 *              v11 2026-10-26 license line simplified to plain MIT, no code change;
 *              v13 2026-10-27 project line of the header reworded, no code change
 * author:      Mitchell A. Thornton <mitch@smu.edu>, ORCID 0000-0003-3559-9511
 * SPDX-License-Identifier: MIT
 * (c) 2026 Mitchell A. Thornton
 */
/* ============================================================================
 * vsim_block_main.c -- driver for block-separable support factoring.
 *   vsim_block <file> [small_limit]
 * Reports the embedding-cost bracket obtained by factoring over support-disjoint
 * output cones, the number of components, and the free-input count. For n <= 22
 * it also runs brute force and checks the exact bracket contains it. Output line
 * starts with "BLOCK".
 * ==========================================================================*/
#include "vsim.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv){
    if (argc<2){ fprintf(stderr,"usage: %s <netlist> [small_limit]\n", argv[0]); return 1; }
    int small = (argc>2)? atoi(argv[2]) : 18;
    Vsim *v=vsim_load(argv[1]);
    if (!v){ fprintf(stderr,"parse failed %s\n",argv[1]); return 2; }
    if (vsim_finalize(v)!=0){ vsim_free(v); return 3; }
    int n=v->n_in, m=v->n_out;

    VsimBlock b=vsim_embed_block(v,small);

    int vb=-1; const char *chk="n/a";
    if (n<=22){
        uint64_t *fw; int nn,mm;
        if (vsim_build_truth(v,&fw,&nn,&mm)==0){ vsim_cp_brute_fw(fw,nn,&vb); free(fw);
            /* the (possibly exact) bracket must contain the true v */
            chk = (vb>=b.lo && vb<=b.hi)? "in" : "OUT";
        }
    }
    printf("BLOCK circuit=%s n=%d m=%d comps=%d free=%d v=[%d,%d] exact=%s v_brute=%d bracket=%s\n",
           v->name,n,m,b.n_comps,b.n_free,b.lo,b.hi, b.exact?"yes":"no", vb, chk);
    vsim_free(v);
    return (strcmp(chk,"OUT")==0)? 10 : 0;
}
