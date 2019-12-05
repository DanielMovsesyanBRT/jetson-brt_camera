/*
 * isp.cpp
 *
 *  Created on: Nov 26, 2019
 *      Author: daniel
 */

#include "isp.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>
#include "camera.hpp"

#define NUM_SKIPPS                          (1)
#define NUM_ACCUMULATIONS                   (3)
#define MAX_EXPOSURE                        (20.0)
#define MIN_DOUBLE_COMPARE                  (0.01)
#define EXPOSURE_CORRECTION                 (0.77)
#define HISTERESIS_SIZE                     (0.11)

#define DECAYING_COEFFICIENT                (0.4)

#define SIGN(x)                             ( ((x) < 0) ? -1 : ((x) > 0) ? 1 : 0 )

namespace brt
{
namespace jupiter
{
namespace image
{

/*
 * \\fn ISP::ISP
 *
 * created on: Nov 26, 2019
 * author: daniel
 *
 */
ISP::ISP(bool group /*= false*/)
: _group(group)
{
}

/*
 * \\fn ISP::~ISP
 *
 * created on: Nov 26, 2019
 * author: daniel
 *
 */
ISP::~ISP()
{
}

/*
 * \\fn ISP::consume
 *
 * created on: Nov 26, 2019
 * author: daniel
 *
 */
void ISP::consume(ImageBox box)
{
  int id = box[0]->get("camera_id",-1);
  if (id == -1)
    return;

  // Line fitting function
  auto line_interpolation = [](CameraBlock& block,uint32_t total_pixels)->double
  {
    double mean_x = block._histogram.size() / 2;
    double mean_y = (double)total_pixels / block._histogram.size();
    double numerator = 0.0, denumerator = 0.0;
    for (size_t index = 0; index < block._histogram.size(); index++)
    {
      numerator += (index - mean_x) * (block._histogram[index] - mean_y);
      denumerator += (index - mean_x) * (index - mean_x);
    }

    double coeff = numerator / denumerator;
    coeff = 10.0 * coeff / total_pixels;
    return coeff * coeff * coeff;
  };

  auto expected_value = [](CameraBlock& block,uint32_t total_pixels)->double
  {
    double result = 0.0;
    for (size_t index = 0; index < block._histogram.size(); index++)
      result += (double)index * (double)block._histogram[index] / (double)total_pixels;

    return result;
  };

  // For group exposure we calibrate only first camera
  if (_group && (id != 0))
    return;

  if (!box[0]->get_bits())
    return;

  HistPtr hist = box[0]->get_bits()->get_histogram();
  if (!hist)
    return;

  std::lock_guard<std::mutex> l(_mutex);
  if (id < _cameras.size())
  {
    uint32_t total_pixels = 0;
    CameraBlock  &block = _cameras[id];

    if (++block._num_captured > NUM_SKIPPS)
    {
      for (size_t index = 0; index < hist->_small_hist.size(); index++)
      {
        if (block._histogram.size() <= index)
          block._histogram.push_back(hist->_small_hist[index]);
        else
          block._histogram[index] += hist->_small_hist[index];

        total_pixels += block._histogram[index];
      }

      if (block._num_captured > NUM_ACCUMULATIONS)
      {
        ///
        //
        // Applying formula Xn = Y(n-1) * (Kn*G0 + K(n-1)*G1)
        //
        //  Here:
        ///    Y(n-1) = M - m(n - 1)
        ///         is difference between mean and desired Mean
        ///
        ///
        ///    Kn = Y(n - 1) / (X(n - 1) * K(n-1))
        ///    K(n-1) = Y(n - 2) / (X(n - 2) * * K(n-2))
        ///    G0 + G1 = 1 - Decaying components
        ///

        double current_mean = expected_value(block, total_pixels);

        double real_mean = ((double)(block._histogram.size() - 1) * (2 + block._histogram.size() - 2) / 2.0
                              / block._histogram.size());

        double desired_mean = real_mean * EXPOSURE_CORRECTION;
        double mean_top = desired_mean + real_mean * HISTERESIS_SIZE;
        double mean_bottom = desired_mean - real_mean * HISTERESIS_SIZE;

        double x = desired_mean - current_mean;
        double k1 = DECAYING_COEFFICIENT;

        if ((current_mean < mean_bottom) || (current_mean > mean_top))
        {
          if (block._m0 != -1.0)
          {
            // check for overshoot
            if ((SIGN(x) == SIGN(desired_mean - block._m0)) &&
                (std::fabs(block._m0 - current_mean) > MIN_DOUBLE_COMPARE))
              k1 = block._k0 * x / (block._m0 - current_mean) * DECAYING_COEFFICIENT;
          }
        }
        else
          x = 0.0;

        double exp_value = x * std::fabs(k1);
        double exposure_ms = block._cam->get_exposure();
        double new_exposure = exposure_ms + exp_value;
        if (new_exposure < 0)
        {
          new_exposure = exposure_ms / 2.0;
          exp_value = new_exposure - exposure_ms;
        }
        else if (new_exposure > MAX_EXPOSURE)
        {
          new_exposure = MAX_EXPOSURE;
          exp_value = new_exposure - exposure_ms;
        }

        if (!_group)
          block._cam->set_exposure(new_exposure);
        else
        {
          for (auto& blk : _cameras)
            blk._cam->set_exposure(new_exposure);
        }

//      Enable for debugging only
//        if (block._id != 0)
//        {
//          std::cout << "(" << block._name << ")"
//              << "m=" << current_mean
//              << ", m0=" << block._m0
//              << ", k=" << k1
//              << ", k0=" << block._k0
//              << ", cur_exp = " << exposure_ms << "ms."
//              << ", new_exp = " << new_exposure << "ms." << std::endl;
//        }


        block._histogram.clear();
        block._num_captured = 0;
        block._m0 = current_mean;
        block._k0 = k1;

      }
    }
  }
}

/*
 * \\fn void ISP::add_camera
 *
 * created on: Nov 26, 2019
 * author: daniel
 *
 */
void ISP::add_camera(Camera* camera)
{
  std::lock_guard<std::mutex> l(_mutex);
  std::vector<CameraBlock>::iterator iter =
        std::find_if(_cameras.begin(),_cameras.end(),[camera](CameraBlock blk)
  {
    return (blk._cam == camera);
  });

  if (iter != _cameras.end())
    return;

  CameraBlock block;
  block._cam = camera;
  block._num_captured = 0;

  block._k0 = 1.0;
  block._m0 = -1.0;

  block._name = camera->name();
  block._id = camera->id();

  _cameras.push_back(block);
  camera->register_consumer(this,Metadata().set<int>("camera_id",_cameras.size() - 1));
}

/*
 * \\fn ISP::stop
 *
 * created on: Nov 26, 2019
 * author: daniel
 *
 */
void ISP::stop()
{
  _mutex.lock();
  for (auto block : _cameras)
  {
    block._cam->unregister_consumer(this);
  }
  _cameras.clear();
  _mutex.unlock();
}



} /* namespace image */
} /* namespace jupiter */
} /* namespace brt */
