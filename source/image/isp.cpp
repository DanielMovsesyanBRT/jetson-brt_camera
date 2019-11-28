/*
 * isp.cpp
 *
 *  Created on: Nov 26, 2019
 *      Author: daniel
 */

#include "isp.hpp"
#include "Camera.hpp"

#include <algorithm>
#include <iostream>
#include <cmath>

#define NUM_SKIPPS                          (4)

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

  // First Check exposure
  int max = -1;
  uint32_t max_val = 0;
  uint32_t total_pixels = 0;

  for (size_t index = 0; index < hist->_small_hist.size(); index++)
  {
    total_pixels += hist->_small_hist[index];
    if (max_val < hist->_small_hist[index])
    {
      max_val = hist->_small_hist[index];
      max = index;
    }
  }

  std::lock_guard<std::mutex> l(_mutex);
  if (id < _cameras.size())
  {
    CameraBlock  &block = _cameras[id];
    block._max = max;
    block._max_val = max_val;

    if (_group)
    {
      ++block._num_captured;

      double absolute_exposure_ms = 0.1;
      double winner_speed = 0.0;
      bool change_exposure = false;

      for (auto blk : _cameras)
      {
        if (blk._num_captured < NUM_SKIPPS)
          // Skip until all cameras are aligned
          return;

        double change_speed = (double)blk._max_val * 10.0 / total_pixels;
        if (change_speed < 10.0)
          change_speed += 0.04 * change_speed * change_speed - 0.4 * change_speed;

        if (winner_speed < change_speed)
        {
          winner_speed = change_speed;
          double exposure_ms = blk._cam->get_exposure();
          if (blk._max == 0)
          {
            // under-expose
            absolute_exposure_ms = exposure_ms + change_speed;
            change_exposure = true;
          }

          else if (blk._max == (hist->_small_hist.size() - 1))
          {
            if (exposure_ms > change_speed)
            {
              absolute_exposure_ms = exposure_ms - change_speed;
              change_exposure = true;
            }
            else if (exposure_ms > 0.1)
            {
              absolute_exposure_ms = exposure_ms / 1.1;
              change_exposure = true;
            }
          }
        }
      }

      if (change_exposure)
      {
        std::cout << "Winner speed: " << winner_speed << std::endl;

        for (auto &blk : _cameras)
        {
          blk._cam->set_exposure(absolute_exposure_ms);
          blk._num_captured = 0;
        }
      }
    }
    else
    {
      if (++block._num_captured > NUM_SKIPPS)
      {
        double change_speed = std::floor((double)max_val * 10.0 / total_pixels);
        double exposure_ms = block._cam->get_exposure();

        if (max == 0)
        {
          // under-expose
          block._cam->set_exposure(exposure_ms + change_speed);
          block._num_captured = 0;
        }

        else if (max == (hist->_small_hist.size() - 1))
        {
          if (exposure_ms > change_speed)
          {
            block._cam->set_exposure(exposure_ms - change_speed);
            block._num_captured = 0;
          }
          else if (exposure_ms > 0.1)
          {
            block._cam->set_exposure(exposure_ms / 1.1);
            block._num_captured = 0;
          }
        }
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
  block._max = -1;
  block._max_val = 0;

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
