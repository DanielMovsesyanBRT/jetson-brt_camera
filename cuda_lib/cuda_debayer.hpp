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

/*
 * \\enum DebayerAlgorithm
 *
 * created on: Apr 21, 2020
 *
 */
enum DebayerAlgorithm
{
  daBiLinear = 0,

  // Adaptive Homogeneity-directed Demosaicing
  daAHD = 1,

  // Malvar-He-Cutler Linear Image Demosaicing
  daMHC = 2
};

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
public:
  Debayer();
  virtual ~Debayer();

          bool                    init(size_t width,size_t height,size_t small_hits_size);
          image::RawRGBPtr        ahd(image::RawRGBPtr img);
          image::RawRGBPtr        mhc(image::RawRGBPtr img);
          image::RawRGBPtr        bilinear(image::RawRGBPtr img);

  virtual void                    consume(image::ImageBox box);
          DebayerAlgorithm        debayer_algorithm() const { return _daType; }
          void                    set_debayer_algorithm(DebayerAlgorithm da) { _daType = da; }

private:
  static Debayer_impl*            _impl;
  size_t                          _width;
  size_t                          _height;
  DebayerAlgorithm                _daType;
};

} /* namespace jupiter */
} /* namespace brt */

#endif /* SOURCE_IMAGE_DEBAYER_HPP_ */
