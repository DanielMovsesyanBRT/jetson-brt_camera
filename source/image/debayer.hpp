/*
 * debayer.hpp
 *
 *  Created on: Feb 14, 2020
 *      Author: daniel
 */

#ifndef SOURCE_IMAGE_DEBAYER_HPP_
#define SOURCE_IMAGE_DEBAYER_HPP_

#include <stdint.h>
#include <stddef.h>

#include <mutex>
#include <thread>
#include <deque>
#include <atomic>

#include <condition_variable>

#include "image.hpp"

#define DEFAULT_NUMBER_OF_THREADS           (64)

namespace brt
{
namespace jupiter
{

class Debayer_impl;
/*
 * \\class Debayer1D
 *
 * created on: Feb 14, 2020
 *
 */
class Debayer : public image::ImageConsumer
              , public image::ImageProducer
{
  Debayer();
  virtual ~Debayer();
public:

          bool                    init(size_t width,size_t height,size_t small_hits_size);
          image::RawRGBPtr        ahd(image::RawRGBPtr img);

  virtual void                    consume(image::ImageBox box);
  static Debayer*                 get() { return &_db; }

private:
          void                    loop();
private:
  static Debayer                  _db;
  Debayer_impl*                   _impl;
  size_t                          _width;
  size_t                          _height;

  std::thread                     _th;
  std::deque<image::RawRGBPtr>    _img_stack;
  std::mutex                      _mutex;
  std::condition_variable         _cv;
  std::atomic_bool                _stop_flag;
};

} /* namespace jupiter */
} /* namespace brt */

#endif /* SOURCE_IMAGE_DEBAYER_HPP_ */
