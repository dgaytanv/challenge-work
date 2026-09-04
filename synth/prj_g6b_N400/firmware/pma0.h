#ifndef PMA0_H_
#define PMA0_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"


// Prototype of top level function for C-synthesis
void pma0(
    x_feat_t x_feat[400*14], mask_add_t mask_add[400*32],
    result_t layer37_out[6]
);

// hls-fpga-machine-learning insert emulator-defines


#endif
