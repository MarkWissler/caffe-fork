#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void QuadraticWeightedKappaLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // do science
}

template <typename Dtype>
void QuadraticWeightedKappaLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // MORE SCIENCE
}

template <typename Dtype>
void QuadraticWeightedKappaLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // SCIENCING INTENSIFIES
}

template <typename Dtype>
void QuadraticWeightedKappaLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // SO MUCH SCIENCE IT HURTS
}

#ifdef CPU_ONLY
STUB_GPU(QuadraticWeightedKappaLossLayer);
#endif

INSTANTIATE_CLASS(QuadraticWeightedKappaLossLayer);
REGISTER_LAYER_CLASS(QuadraticWeightedKappaLoss);

}  // namespace caffe
