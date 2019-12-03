/*
 * image_processor.hpp
 *
 *  Created on: Nov 25, 2019
 *      Author: daniel
 */

#ifndef SOURCE_IMAGE_IMAGE_PROCESSOR_HPP_
#define SOURCE_IMAGE_IMAGE_PROCESSOR_HPP_

#include <stdint.h>
#include <cuda_runtime.h>

#include <vector>
#include <atomic>

#include "../utils.hpp"
#include "image.hpp"

#define DEFAULT_NUMBER_OF_THREADS           (64)

namespace brt
{
namespace jupiter
{
namespace image
{

/*
 * \\class ImageProcessor
 *
 * created on: Nov 25, 2019
 *
 */
class ImageProcessor
{
public:
  ImageProcessor();
  virtual ~ImageProcessor();

          RawRGBPtr               debayer(RawRGBPtr raw, bool outputBGR = false);
          bool                    get_histogram(HistPtr& histogram);

          void                    set_overexp_flag(bool flag) { _overexposure_flag = flag; }

private:
          bool                    runDebayer(bool outputBGR);

private:
  CudaPtr<uint16_t>               _img_buffer;
  CudaPtr<uint16_t>               _img_debayer_buffer;
  CudaPtr<uint32_t>               _histogram;
  CudaPtr<uint32_t>               _histogram_max;
  CudaPtr<uint32_t>               _small_histogram;

  uint16_t                        _width;
  uint16_t                        _height;
  int                             _thx,_thy;
  int                             _blkx,_blky;

  std::atomic_bool                _overexposure_flag;
};


} /* namespace image */
} /* namespace jupiter */
} /* namespace brt */

#endif /* SOURCE_IMAGE_IMAGE_PROCESSOR_HPP_ */
