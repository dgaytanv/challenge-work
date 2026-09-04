#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
#include "nnet_utils/nnet_conv1d.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_einsum.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"
#include "nnet_utils/nnet_sepconv1d_stream.h"

// hls-fpga-machine-learning insert weights
#include "weights/w38.h"
#include "weights/b38.h"
#include "weights/s5.h"
#include "weights/b5.h"
#include "weights/table7.h"
#include "weights/w39.h"
#include "weights/b39.h"
#include "weights/s12.h"
#include "weights/b12.h"
#include "weights/table14.h"
#include "weights/w40.h"
#include "weights/b40.h"
#include "weights/w41.h"
#include "weights/b41.h"
#include "weights/w42.h"
#include "weights/b42.h"
#include "weights/s35.h"
#include "weights/b35.h"
#include "weights/w37.h"
#include "weights/b37.h"


// hls-fpga-machine-learning insert layer-config
// phi_0
struct config43_mult : nnet::dense_config {
    static const unsigned n_in = 14;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 213;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef phi_0_accum_t accum_t;
    typedef phi_0_bias_t bias_t;
    typedef phi_0_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config43 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 400;
    static const unsigned n_chan = 14;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 128;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 400;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 213;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 400;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 400;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_43<data_T, CONFIG_T>;
    typedef phi_0_accum_t accum_t;
    typedef phi_0_bias_t bias_t;
    typedef phi_0_weight_t weight_t;
    typedef config43_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::Conv1DLatency<data_T, res_T, CONFIG_T>;
};
const ap_uint<config43::filt_width> config43::pixels[] = {0};

// nrm0
struct config5 : nnet::batchnorm_config {
    static const unsigned n_in = 400*128;
    static const unsigned n_filt = 128;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef nrm0_bias_t bias_t;
    typedef nrm0_scale_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// phi_act0
struct unary_lut_config7 : nnet::activ_config {
    static const unsigned n_in = 51200;
    static const unsigned table_size = 16384;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef phi_act0_table_t table_t;
};

// phi_1
struct config44_mult : nnet::dense_config {
    static const unsigned n_in = 128;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 5248;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef phi_1_accum_t accum_t;
    typedef phi_1_bias_t bias_t;
    typedef phi_1_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config44 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 400;
    static const unsigned n_chan = 128;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 128;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 400;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 5248;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 400;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 400;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_44<data_T, CONFIG_T>;
    typedef phi_1_accum_t accum_t;
    typedef phi_1_bias_t bias_t;
    typedef phi_1_weight_t weight_t;
    typedef config44_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::Conv1DLatency<data_T, res_T, CONFIG_T>;
};
const ap_uint<config44::filt_width> config44::pixels[] = {0};

// nrm1
struct config12 : nnet::batchnorm_config {
    static const unsigned n_in = 400*128;
    static const unsigned n_filt = 128;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef nrm1_bias_t bias_t;
    typedef nrm1_scale_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// phi_act1
struct unary_lut_config14 : nnet::activ_config {
    static const unsigned n_in = 51200;
    static const unsigned table_size = 8192;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef phi_act1_table_t table_t;
};

// score
struct config45_mult : nnet::dense_config {
    static const unsigned n_in = 128;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 1213;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef score_accum_t accum_t;
    typedef score_bias_t bias_t;
    typedef score_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config45 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 400;
    static const unsigned n_chan = 128;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 32;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 400;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 1213;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 400;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 400;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_45<data_T, CONFIG_T>;
    typedef score_accum_t accum_t;
    typedef score_bias_t bias_t;
    typedef score_weight_t weight_t;
    typedef config45_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::Conv1DLatency<data_T, res_T, CONFIG_T>;
};
const ap_uint<config45::filt_width> config45::pixels[] = {0};

// mask_add_op
struct config21 : nnet::merge_config {
    static const unsigned n_elem = 400*32;
    static const unsigned n_elem1 = 400*32;
    static const unsigned n_elem2 = 400*32;
    static const unsigned reuse_factor = 1;
};

// softmax
struct softmax_config22 : nnet::activ_config {
    static const unsigned n_in = 12800;
    static const unsigned n_slice = 400;
    static const unsigned n_outer = 1;
    static const unsigned n_inner = 32;
    static const unsigned parallelization_factor = 1;
    static const unsigned exp_table_size = 8192;
    static const unsigned inv_table_size = 16384;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned axis = 1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::stable;
    static constexpr float exp_scale = 1.0;
    typedef softmax_exp_table_t exp_table_t;
    typedef softmax_inv_table_t inv_table_t;
    typedef softmax_accum_t accum_t;
    typedef softmax_inv_inp_t inv_inp_t;
    typedef softmax_inp_norm_t inp_norm_t;
};

// v
struct config46_mult : nnet::dense_config {
    static const unsigned n_in = 128;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 5230;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef v_accum_t accum_t;
    typedef v_bias_t bias_t;
    typedef v_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config46 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 400;
    static const unsigned n_chan = 128;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 128;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 400;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 5230;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 400;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 400;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_46<data_T, CONFIG_T>;
    typedef v_accum_t accum_t;
    typedef v_bias_t bias_t;
    typedef v_weight_t weight_t;
    typedef config46_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::Conv1DLatency<data_T, res_T, CONFIG_T>;
};
const ap_uint<config46::filt_width> config46::pixels[] = {0};

// combine
struct config29_tpose_inp0 {
    static const unsigned dims = 3;
    static const unsigned N = 12800;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config29_tpose_inp0_from_shape[3] = {400, 8, 4};
unsigned config29_tpose_inp0_to_shape[3] = {8, 4, 400};
unsigned config29_tpose_inp0_perm[3] = {1, 2, 0};
unsigned config29_tpose_inp0_perm_strides[3] = {4, 1, 32};

const unsigned* const config29_tpose_inp0::from_shape = config29_tpose_inp0_from_shape;
const unsigned* const config29_tpose_inp0::to_shape = config29_tpose_inp0_to_shape;
const unsigned* const config29_tpose_inp0::perm = config29_tpose_inp0_perm;
const unsigned* const config29_tpose_inp0::perm_strides = config29_tpose_inp0_perm_strides;


struct config29_tpose_inp1 {
    static const unsigned dims = 3;
    static const unsigned N = 51200;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config29_tpose_inp1_from_shape[3] = {400, 8, 16};
unsigned config29_tpose_inp1_to_shape[3] = {8, 16, 400};
unsigned config29_tpose_inp1_perm[3] = {1, 2, 0};
unsigned config29_tpose_inp1_perm_strides[3] = {16, 1, 128};

const unsigned* const config29_tpose_inp1::from_shape = config29_tpose_inp1_from_shape;
const unsigned* const config29_tpose_inp1::to_shape = config29_tpose_inp1_to_shape;
const unsigned* const config29_tpose_inp1::perm = config29_tpose_inp1_perm;
const unsigned* const config29_tpose_inp1::perm_strides = config29_tpose_inp1_perm_strides;


struct config29_tpose_out {
    static const unsigned dims = 3;
    static const unsigned N = 512;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config29_tpose_out_from_shape[3] = {8, 4, 16};
unsigned config29_tpose_out_to_shape[3] = {4, 8, 16};
unsigned config29_tpose_out_perm[3] = {1, 0, 2};
unsigned config29_tpose_out_perm_strides[3] = {16, 64, 1};

const unsigned* const config29_tpose_out::from_shape = config29_tpose_out_from_shape;
const unsigned* const config29_tpose_out::to_shape = config29_tpose_out_to_shape;
const unsigned* const config29_tpose_out::perm = config29_tpose_out_perm;
const unsigned* const config29_tpose_out::perm_strides = config29_tpose_out_perm_strides;



struct config29 {
    typedef config29_tpose_inp0 tpose_inp0_config;
    typedef config29_tpose_inp1 tpose_inp1_config;
    typedef config29_tpose_out tpose_out_conf;

    typedef combine_accum_t accum_t;

    // Layer Sizes
    static const unsigned n_free0 = 4;
    static const unsigned n_free1 = 16;
    static const unsigned n_contract = 400;
    static const unsigned n_inplace = 8;

    // Resource reuse info
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 400;
    static const unsigned multiplier_limit = 512;
    static const bool store_weights_in_bram = false; // NOT USED

    template <class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// out_proj
struct config47_mult : nnet::dense_config {
    static const unsigned n_in = 512;
    static const unsigned n_out = 512;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 5589;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef out_proj_accum_t accum_t;
    typedef out_proj_bias_t bias_t;
    typedef out_proj_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config47 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 4;
    static const unsigned n_chan = 128;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 128;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 4;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 5589;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 4;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 1;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_47<data_T, CONFIG_T>;
    typedef out_proj_accum_t accum_t;
    typedef out_proj_bias_t bias_t;
    typedef out_proj_weight_t weight_t;
    typedef config47_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::BatchedDenseForConv1D<data_T, res_T, CONFIG_T>;
};
const ap_uint<config47::filt_width> config47::pixels[] = {0};

// norm_pooled
struct config35 : nnet::batchnorm_config {
    static const unsigned n_in = 512;
    static const unsigned n_filt = 512;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef norm_pooled_bias_t bias_t;
    typedef norm_pooled_scale_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// bottleneck
struct config37 : nnet::dense_config {
    static const unsigned n_in = 512;
    static const unsigned n_out = 6;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 1176;
    static const unsigned n_nonzeros = 1896;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef bottleneck_accum_t accum_t;
    typedef bottleneck_bias_t bias_t;
    typedef bottleneck_weight_t weight_t;
    typedef layer37_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};



#endif
