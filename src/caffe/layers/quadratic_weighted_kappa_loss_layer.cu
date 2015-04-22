#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void QuadraticWeightedKappaLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Do gpu science
}

template <typename Dtype>
void QuadraticWeightedKappaLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // thousands of cores
}

INSTANTIATE_LAYER_GPU_FUNCS(QuadraticWeightedKappaLossLayer);
  // THOUSANDS.

}  // namespace caffe
