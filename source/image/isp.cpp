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


ISP ISP::_object;
/*
 * \\fn ISP::ISP
 *
 * created on: Nov 26, 2019
 * author: daniel
 *
 */
ISP::ISP()
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

  _mutex.lock();
  if (id < _cameras.size())
  {
    CameraBlock  &block = _cameras[id];
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
  _mutex.unlock();


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
