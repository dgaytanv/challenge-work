#ifndef PMA0_BRIDGE_H_
#define PMA0_BRIDGE_H_

#include "firmware/pma0.h"
#include "firmware/nnet_utils/nnet_helpers.h"
#include <algorithm>
#include <map>

// hls-fpga-machine-learning insert bram

namespace nnet {
bool trace_enabled = false;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

extern "C" {

struct trace_data {
    const char *name;
    void *data;
};

void allocate_trace_storage(size_t element_size) {
    nnet::trace_enabled = true;
    nnet::trace_outputs = new std::map<std::string, void *>;
    nnet::trace_type_size = element_size;
}

void free_trace_storage() {
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        void *ptr = i->second;
        free(ptr);
    }
    nnet::trace_outputs->clear();
    delete nnet::trace_outputs;
    nnet::trace_outputs = NULL;
    nnet::trace_enabled = false;
}

void collect_trace_output(struct trace_data *c_trace_outputs) {
    int ii = 0;
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        c_trace_outputs[ii].name = i->first.c_str();
        c_trace_outputs[ii].data = i->second;
        ii++;
    }
}

// hls-fpga-machine-learning insert tb_input_writer

// Wrapper of top level function for Python bridge
void pma0_float(
    float *x_feat, float *mask_add,
    float *layer37_out
) {

    x_feat_t x_feat_ap[400*14];
    nnet::convert_data<float, x_feat_t, 400*14>(x_feat, x_feat_ap);
    mask_add_t mask_add_ap[400*32];
    nnet::convert_data<float, mask_add_t, 400*32>(mask_add, mask_add_ap);

    result_t layer37_out_ap[6];

    pma0(x_feat_ap,mask_add_ap,layer37_out_ap);

    nnet::convert_data<result_t, float, 6>(layer37_out_ap, layer37_out);
}

void pma0_double(
    double *x_feat, double *mask_add,
    double *layer37_out
) {

    x_feat_t x_feat_ap[400*14];
    nnet::convert_data<double, x_feat_t, 400*14>(x_feat, x_feat_ap);
    mask_add_t mask_add_ap[400*32];
    nnet::convert_data<double, mask_add_t, 400*32>(mask_add, mask_add_ap);

    result_t layer37_out_ap[6];

    pma0(x_feat_ap,mask_add_ap,layer37_out_ap);

    nnet::convert_data<result_t, double, 6>(layer37_out_ap, layer37_out);
}
}

#endif
