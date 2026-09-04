#include <iostream>

#include "pma0.h"
#include "parameters.h"


void pma0(
    x_feat_t x_feat[400*14], mask_add_t mask_add[400*32],
    result_t layer37_out[6]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=x_feat complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=mask_add complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer37_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=x_feat,mask_add,layer37_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<phi_0_weight_t, 1792>(w38, "w38.txt");
        nnet::load_weights_from_txt<phi_0_bias_t, 128>(b38, "b38.txt");
        nnet::load_weights_from_txt<nrm0_scale_t, 128>(s5, "s5.txt");
        nnet::load_weights_from_txt<nrm0_bias_t, 128>(b5, "b5.txt");
        nnet::load_weights_from_txt<phi_act0_table_t, 16384>(table7, "table7.txt");
        nnet::load_weights_from_txt<phi_1_weight_t, 16384>(w39, "w39.txt");
        nnet::load_weights_from_txt<phi_1_bias_t, 128>(b39, "b39.txt");
        nnet::load_weights_from_txt<nrm1_scale_t, 128>(s12, "s12.txt");
        nnet::load_weights_from_txt<nrm1_bias_t, 128>(b12, "b12.txt");
        nnet::load_weights_from_txt<phi_act1_table_t, 8192>(table14, "table14.txt");
        nnet::load_weights_from_txt<score_weight_t, 4096>(w40, "w40.txt");
        nnet::load_weights_from_txt<score_bias_t, 32>(b40, "b40.txt");
        nnet::load_weights_from_txt<v_weight_t, 16384>(w41, "w41.txt");
        nnet::load_weights_from_txt<v_bias_t, 128>(b41, "b41.txt");
        nnet::load_weights_from_txt<out_proj_weight_t, 16384>(w42, "w42.txt");
        nnet::load_weights_from_txt<out_proj_bias_t, 128>(b42, "b42.txt");
        nnet::load_weights_from_txt<norm_pooled_scale_t, 512>(s35, "s35.txt");
        nnet::load_weights_from_txt<norm_pooled_bias_t, 512>(b35, "b35.txt");
        nnet::load_weights_from_txt<bottleneck_weight_t, 3072>(w37, "w37.txt");
        nnet::load_weights_from_txt<bottleneck_bias_t, 6>(b37, "b37.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    phi_0_iq_t layer2_out[400*14];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    phi_0_t layer38_out[400*128];
    #pragma HLS ARRAY_PARTITION variable=layer38_out complete dim=0

    nrm0_iq_t layer4_out[400*128];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    nrm0_t layer5_out[400*128];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    phi_act0_iq_t layer6_out[400*128];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    phi_act0_t layer7_out[400*128];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0

    phi_1_t layer39_out[400*128];
    #pragma HLS ARRAY_PARTITION variable=layer39_out complete dim=0

    nrm1_iq_t layer11_out[400*128];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0

    nrm1_t layer12_out[400*128];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0

    phi_act1_iq_t layer13_out[400*128];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0

    phi_act1_t layer14_out[400*128];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0

    score_iq_t layer16_out[400*128];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0

    score_t layer40_out[400*32];
    #pragma HLS ARRAY_PARTITION variable=layer40_out complete dim=0

    quantizer_t layer19_out[400*32];
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0

    mask_add_op_t layer21_out[400*32];
    #pragma HLS ARRAY_PARTITION variable=layer21_out complete dim=0

    softmax_t layer22_out[400*32];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0

    v_iq_t layer23_out[400*128];
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0

    v_t layer41_out[400*128];
    #pragma HLS ARRAY_PARTITION variable=layer41_out complete dim=0

    auto& layer25_out = layer22_out;
    auto& layer26_out = layer41_out;
    quantizer_2_t layer27_out[400*8*4];
    #pragma HLS ARRAY_PARTITION variable=layer27_out complete dim=0

    quantizer_3_t layer28_out[400*8*16];
    #pragma HLS ARRAY_PARTITION variable=layer28_out complete dim=0

    combine_t layer29_out[4*8*16];
    #pragma HLS ARRAY_PARTITION variable=layer29_out complete dim=0

    auto& layer30_out = layer29_out;
    out_proj_iq_t layer31_out[4*128];
    #pragma HLS ARRAY_PARTITION variable=layer31_out complete dim=0

    out_proj_t layer42_out[4*128];
    #pragma HLS ARRAY_PARTITION variable=layer42_out complete dim=0

    auto& layer33_out = layer42_out;
    norm_pooled_iq_t layer34_out[512];
    #pragma HLS ARRAY_PARTITION variable=layer34_out complete dim=0

    norm_pooled_t layer35_out[512];
    #pragma HLS ARRAY_PARTITION variable=layer35_out complete dim=0

    bottleneck_iq_t layer36_out[512];
    #pragma HLS ARRAY_PARTITION variable=layer36_out complete dim=0

    nnet::phi_0_iq<x_feat_t, phi_0_iq_t>(x_feat, layer2_out); // phi_0_iq

    nnet::pointwise_conv_1d_cl<phi_0_iq_t, phi_0_t, config43>(layer2_out, layer38_out, w38, b38); // phi_0

    nnet::nrm0_iq<phi_0_t, nrm0_iq_t>(layer38_out, layer4_out); // nrm0_iq

    nnet::normalize<nrm0_iq_t, nrm0_t, config5>(layer4_out, layer5_out, s5, b5); // nrm0

    nnet::phi_act0_iq<nrm0_t, phi_act0_iq_t>(layer5_out, layer6_out); // phi_act0_iq

    nnet::unary_lut<phi_act0_iq_t, phi_act0_t, unary_lut_config7>(layer6_out, layer7_out, table7); // phi_act0

    nnet::pointwise_conv_1d_cl<phi_act0_t, phi_1_t, config44>(layer7_out, layer39_out, w39, b39); // phi_1

    nnet::nrm1_iq<phi_1_t, nrm1_iq_t>(layer39_out, layer11_out); // nrm1_iq

    nnet::normalize<nrm1_iq_t, nrm1_t, config12>(layer11_out, layer12_out, s12, b12); // nrm1

    nnet::phi_act1_iq<nrm1_t, phi_act1_iq_t>(layer12_out, layer13_out); // phi_act1_iq

    nnet::unary_lut<phi_act1_iq_t, phi_act1_t, unary_lut_config14>(layer13_out, layer14_out, table14); // phi_act1

    nnet::score_iq<phi_act1_t, score_iq_t>(layer14_out, layer16_out); // score_iq

    nnet::pointwise_conv_1d_cl<score_iq_t, score_t, config45>(layer16_out, layer40_out, w40, b40); // score

    nnet::quantizer<score_t, quantizer_t>(layer40_out, layer19_out); // quantizer

    nnet::add<quantizer_t, mask_add_t, mask_add_op_t, config21>(layer19_out, mask_add, layer21_out); // mask_add_op

    nnet::softmax_multidim<mask_add_op_t, softmax_t, softmax_config22>(layer21_out, layer22_out); // softmax

    nnet::v_iq<phi_act1_t, v_iq_t>(layer14_out, layer23_out); // v_iq

    nnet::pointwise_conv_1d_cl<v_iq_t, v_t, config46>(layer23_out, layer41_out, w41, b41); // v

    nnet::quantizer_2<softmax_t, quantizer_2_t>(layer25_out, layer27_out); // quantizer_2

    nnet::quantizer_3<v_t, quantizer_3_t>(layer26_out, layer28_out); // quantizer_3

    nnet::einsum<quantizer_2_t, quantizer_3_t, combine_t, config29>(layer27_out, layer28_out, layer29_out); // combine

    nnet::out_proj_iq<combine_t, out_proj_iq_t>(layer30_out, layer31_out); // out_proj_iq

    nnet::pointwise_conv_1d_cl<out_proj_iq_t, out_proj_t, config47>(layer31_out, layer42_out, w42, b42); // out_proj

    nnet::norm_pooled_iq<out_proj_t, norm_pooled_iq_t>(layer33_out, layer34_out); // norm_pooled_iq

    nnet::normalize<norm_pooled_iq_t, norm_pooled_t, config35>(layer34_out, layer35_out, s35, b35); // norm_pooled

    nnet::bottleneck_iq<norm_pooled_t, bottleneck_iq_t>(layer35_out, layer36_out); // bottleneck_iq

    nnet::dense<bottleneck_iq_t, result_t, config37>(layer36_out, layer37_out, w37, b37); // bottleneck

}

