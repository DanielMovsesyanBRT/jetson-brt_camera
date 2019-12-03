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

#define NUM_SKIPPS                          (2)
#define NUM_ACCUMULATIONS                   (5)
#define MAX_EXPOSURE                        (20.0)

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

      double expected = expected_value(block, total_pixels);
      block._accumulated_mean += expected;
      block._num_accumulations++;
    }

    if (_group)
    {
      double absolute_exposure_ms = 0.1;
      double max_coeff = 0.0;
      bool change_exposure = false;

      for (auto& blk : _cameras)
      {
        if (blk._num_captured < NUM_ACCUMULATIONS)
          // Skip until all cameras are aligned
          return;

        uint32_t total_pixels = 0;
        for (size_t index = 0; index < blk._histogram.size(); index++)
          total_pixels += blk._histogram[index];

        // Fitting points to a line to adjust exposure based on slope
        double coeff = line_interpolation(blk, total_pixels);

        if (std::fabs(max_coeff) < std::fabs(coeff))
        {
          double exposure_ms = blk._cam->get_exposure();
          absolute_exposure_ms = exposure_ms - coeff;
          change_exposure = true;
          max_coeff = coeff;
        }
      }

      if (change_exposure)
      {
        for (auto &blk : _cameras)
        {
          blk._cam->set_exposure(absolute_exposure_ms);
          blk._num_captured = 0;
        }
      }

      for (auto &blk : _cameras)
      {
        blk._histogram.clear();
        blk._num_captured = (blk._num_captured != 0) ? NUM_SKIPPS : 0;
      }
    }
    else
    {
      if (block._num_captured > NUM_ACCUMULATIONS)
      {
        double desired_mean = (double)(block._histogram.size() - 1) * (2 + block._histogram.size() - 2) / 2.0
                                  / block._histogram.size();

        double current_mean = block._accumulated_mean / block._num_accumulations;
        double delta_exposure = desired_mean - current_mean;
        double exp_value = delta_exposure;

        // Recalculate exposure value based on last settings
        if ((block._last_delta_mean != 0.0) && (block._last_exposure_value != 0.0))
        {
          double coeff = std::fabs((block._last_delta_mean - delta_exposure) / (block._last_exposure_value));
          if (coeff != 0.0)
            exp_value = delta_exposure / coeff;
        }

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

//      Enable for debugging only
//        if (block._id != 0)
//        {
//          std::cout << "(" << block._name << ")"
//                    << " mean = " << current_mean
//                    << ", exp = " << exp_value
//                    << ", cur_exp = " << exposure_ms << "ms."
//                    << ", new_exp = " << new_exposure << "ms." << std::endl;
//        }

        block._cam->set_exposure(new_exposure);

        block._histogram.clear();
        block._num_captured = 0;
        block._accumulated_mean = 0.0;
        block._num_accumulations = 0;
        block._last_delta_mean = delta_exposure;
        block._last_exposure_value = exp_value;
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

  block._last_delta_mean = 0.0;
  block._last_exposure_value = 0.0;

  block._accumulated_mean = 0.0;
  block._num_accumulations = 0;

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
