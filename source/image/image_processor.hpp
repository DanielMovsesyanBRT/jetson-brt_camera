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

#include "Utils.hpp"
#include "Image.hpp"

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

          RawRGBPtr               debayer(RawRGBPtr raw, bool outputBGR);
          bool                    get_histogram(std::vector<uint32_t>&, uint32_t& max);

private:
          bool                    runDebayer(bool outputBGR);

private:
  CudaPtr<uint16_t>               _img_buffer;
  CudaPtr<uint16_t>               _img_debayer_buffer;
  CudaPtr<uint32_t>               _histogram;
  CudaPtr<uint32_t>               _histogram_max;

  uint16_t                        _width;
  uint16_t                        _height;
  int                             _thx,_thy;
  int                             _blkx,_blky;
};


} /* namespace image */
} /* namespace jupiter */
} /* namespace brt */

#endif /* SOURCE_IMAGE_IMAGE_PROCESSOR_HPP_ */
