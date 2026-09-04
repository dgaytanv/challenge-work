#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<15,8> x_feat_t;
typedef ap_fixed<14,8> phi_0_iq_t;
typedef ap_fixed<21,8> phi_0_accum_t;
typedef ap_fixed<14,7> phi_0_t;
typedef ap_fixed<5,1> phi_0_weight_t;
typedef ap_fixed<13,0> phi_0_bias_t;
typedef ap_fixed<13,7> nrm0_iq_t;
typedef ap_fixed<15,8> nrm0_t;
typedef ap_ufixed<6,2> nrm0_scale_t;
typedef ap_fixed<16,3> nrm0_bias_t;
typedef ap_fixed<14,8> phi_act0_iq_t;
typedef ap_ufixed<6,2> phi_act0_table_t;
typedef ap_ufixed<6,2,AP_RND,AP_WRAP,0> phi_act0_t;
typedef ap_fixed<22,6> phi_1_accum_t;
typedef ap_fixed<13,6> phi_1_t;
typedef ap_fixed<5,1> phi_1_weight_t;
typedef ap_fixed<14,-2> phi_1_bias_t;
typedef ap_fixed<12,6> nrm1_iq_t;
typedef ap_fixed<14,7> nrm1_t;
typedef ap_ufixed<6,2> nrm1_scale_t;
typedef ap_fixed<15,3> nrm1_bias_t;
typedef ap_fixed<13,7> phi_act1_iq_t;
typedef ap_ufixed<6,2> phi_act1_table_t;
typedef ap_ufixed<6,2,AP_RND,AP_SAT_SYM,0> phi_act1_t;
typedef ap_ufixed<8,2> score_iq_t;
typedef ap_fixed<22,8> score_accum_t;
typedef ap_fixed<13,6> score_t;
typedef ap_fixed<6,2> score_weight_t;
typedef ap_fixed<13,-1> score_bias_t;
typedef ap_fixed<14,8,AP_RND,AP_WRAP,0> mask_add_t;
typedef ap_fixed<12,6> quantizer_t;
typedef ap_fixed<15,9> mask_add_op_t;
typedef ap_ufixed<6,2,AP_RND_CONV,AP_SAT,0> softmax_exp_table_t;
typedef ap_ufixed<6,2,AP_RND_CONV,AP_SAT,0> softmax_inv_table_t;
typedef ap_ufixed<14,8,AP_RND,AP_WRAP,0> softmax_inv_inp_t;
typedef ap_ufixed<13,7,AP_RND,AP_WRAP,0> softmax_inp_norm_t;
typedef ap_ufixed<20,8> softmax_accum_t;
typedef ap_ufixed<8,1> softmax_t;
typedef ap_fixed<18,8> softmax_table_t;
typedef ap_ufixed<8,2> v_iq_t;
typedef ap_fixed<24,6> v_accum_t;
typedef ap_fixed<12,5> v_t;
typedef ap_fixed<5,1> v_weight_t;
typedef ap_fixed<16,-2> v_bias_t;
typedef ap_ufixed<7,1> quantizer_2_t;
typedef ap_fixed<11,5> quantizer_3_t;
typedef ap_fixed<22,10> combine_accum_t;
typedef ap_fixed<12,5> combine_t;
typedef ap_fixed<11,5> out_proj_iq_t;
typedef ap_fixed<21,7> out_proj_accum_t;
typedef ap_fixed<13,6> out_proj_t;
typedef ap_fixed<5,1> out_proj_weight_t;
typedef ap_fixed<13,-1> out_proj_bias_t;
typedef ap_fixed<12,6> norm_pooled_iq_t;
typedef ap_fixed<11,4> norm_pooled_t;
typedef ap_ufixed<6,2> norm_pooled_scale_t;
typedef ap_fixed<14,3> norm_pooled_bias_t;
typedef ap_fixed<10,4> bottleneck_iq_t;
typedef ap_fixed<19,9> bottleneck_accum_t;
typedef ap_fixed<19,9> result_t;
typedef ap_fixed<5,1> bottleneck_weight_t;
typedef ap_fixed<8,-1> bottleneck_bias_t;
typedef ap_uint<1> layer37_index;

// hls-fpga-machine-learning insert emulator-defines


#endif
