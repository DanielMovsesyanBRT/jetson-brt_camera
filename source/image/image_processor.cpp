/*
 * image_processor.cpp
 *
 *  Created on: Nov 25, 2019
 *      Author: daniel
 */

#include "image_processor.hpp"

#define BITS_PER_PIXEL                      (1 << 12) // RAW 12

namespace brt
{
namespace jupiter
{
namespace image
{


/*
 * \\fn ImageProcessor::ImageProcessor
 *
 * created on: Nov 25, 2019
 * author: daniel
 *
 */
ImageProcessor::ImageProcessor()
: _img_buffer()
, _img_debayer_buffer()
, _histogram()
, _histogram_max()
, _width(0)
, _height(0)
, _thx(0)
, _thy(0)
, _blkx(0)
, _blky(0)
{
}

/*
 * \\fn ImageProcessor::~ImageProcessor
 *
 * created on: Nov 25, 2019
 * author: daniel
 *
 */
ImageProcessor::~ImageProcessor()
{
}

/*
 * \\fn RawRGBPtr ImageProcessor::debayer
 *
 * created on: Nov 25, 2019
 * author: daniel
 *
 */
RawRGBPtr ImageProcessor::debayer(RawRGBPtr raw,bool outputBGR)
{
  if (!raw)
    return RawRGBPtr();

  size_t img_size = raw->width() * raw->height();
  if (!_img_buffer.put((uint16_t*)raw->bytes(), raw->width() * raw->height()))
  {
    // assert
    return RawRGBPtr();
  }

  size_t debayer_img_size = img_size * 3; /* RGB*/

  if (!_img_debayer_buffer || (_img_debayer_buffer.size() != debayer_img_size))
    _img_debayer_buffer = CudaPtr<uint16_t>(debayer_img_size);

  if (!_img_debayer_buffer)
  {
    // assert
    return RawRGBPtr();
  }

  if (!_histogram || (_histogram.size() != BITS_PER_PIXEL))
    _histogram = CudaPtr<uint32_t>(BITS_PER_PIXEL);

  if (!_histogram_max || (_histogram_max.size() != BITS_PER_PIXEL))
    _histogram_max = CudaPtr<uint32_t>(BITS_PER_PIXEL);

  // Check, whether image dimensions have changed
  if ((_width != raw->width()) || (_height != raw->height()))
  {
    _width = raw->width();
    _height = raw->height();

    _thx = std::min(DEFAULT_NUMBER_OF_THREADS, (1 << __builtin_ctz(_width >> 1)));
    if (_thx == 0)
      _thx = 1;

    _thy = std::min(DEFAULT_NUMBER_OF_THREADS, (1 << __builtin_ctz(_height >> 1)));
    if (_thy == 0)
      _thy = 1;

    _blkx = (_width >> 1) / _thx;
    if (((_width >> 1) % _thx) != 0)
      _blkx++;

    _blky = (_height >> 1) / _thy;
    if (((_height >> 1) % _thy) != 0)
      _blky++;
  }

  runDebayer(outputBGR);

  RawRGBPtr result(new RawRGB(raw->width(), raw->height(), 3 * sizeof(uint16_t)));
  _img_debayer_buffer.get((uint16_t*)result->bytes(), debayer_img_size);

  return result;
}



} /* namespace image */
} /* namespace jupiter */
} /* namespace brt */
