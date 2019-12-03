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
#include "../device/camera.hpp"

#define NUM_SKIPPS                          (4)
#define NUM_ACCUMULATIONS                   (8)

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

  if (!box[0]->get_bits())
    return;

  HistPtr hist = box[0]->get_bits()->get_histogram();
  if (!hist)
    return;

  std::lock_guard<std::mutex> l(_mutex);
  if (id < _cameras.size())
  {
    CameraBlock  &block = _cameras[id];

    if (++block._num_captured > NUM_SKIPPS)
    {
      for (size_t index = 0; index < hist->_small_hist.size(); index++)
      {
        if (block._histogram.size() <= index)
          block._histogram.push_back(hist->_small_hist[index]);
        else
          block._histogram[index] += hist->_small_hist[index];
      }
    }

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

      //coeff = (coeff < 0) ? -(coeff*coeff) : (coeff*coeff);
      return coeff * coeff * coeff;
    };


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
        uint32_t total_pixels = 0;
        for (size_t index = 0; index < block._histogram.size(); index++)
          total_pixels += block._histogram[index];

        // Fitting points to a line to adjust exposure based on slope
        double coeff = line_interpolation(block, total_pixels);
        double exposure_ms = block._cam->get_exposure();

        block._cam->set_exposure(exposure_ms - coeff);

        block._histogram.clear();
        block._num_captured = (block._num_captured != 0) ? NUM_SKIPPS : 0;
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
//  block._max = -1;
//  block._max_val = 0;

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
