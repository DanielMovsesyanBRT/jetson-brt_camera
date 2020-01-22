/*
 * image_processor.cpp
 *
 *  Created on: Nov 25, 2019
 *      Author: daniel
 */

#include "image_processor.hpp"

#define BITS_PER_PIXEL                      (1 << 16)
#define SMALL_HIST_SIZE                     (9)

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
, _overexposure_flag(false)
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
  if (!_img_buffer.put((uint16_t*)raw->bytes(), img_size))
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

  if (!_small_histogram || (_small_histogram.size() != SMALL_HIST_SIZE))
    _small_histogram = CudaPtr<uint32_t>(SMALL_HIST_SIZE);

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

  _histogram.fill(0);
  _histogram_max.fill(0);
  _small_histogram.fill(0);

  runDebayer(outputBGR);

  RawRGBPtr result(new RawRGB(raw->width(), raw->height(), 3 * sizeof(uint16_t)));
  _img_debayer_buffer.get((uint16_t*)result->bytes(), debayer_img_size);

  image::HistPtr  full_hist;
  get_histogram(full_hist);
  result->set_histogram(full_hist);

  return result;
}

/*
 * \\fn bool ImageProcessor::get_histogram
 *
 * created on: Nov 25, 2019
 * author: daniel
 *
 */
bool ImageProcessor::get_histogram(HistPtr& histogram)
{
  if (!_histogram || !_histogram_max)
    return false;

  if (!histogram)
    histogram.reset(new Histogram);

  if (histogram->_histogram.size() != _histogram.size())
    histogram->_histogram.resize(_histogram.size());

  if (histogram->_small_hist.size() != _small_histogram.size())
    histogram->_small_hist.resize(_small_histogram.size());


  _histogram.get(histogram->_histogram.data(), histogram->_histogram.size());
  _small_histogram.get(histogram->_small_hist.data(), histogram->_small_hist.size());
  _histogram_max.get(&histogram->_max_value, 1);

  return true;
}


/*
 * \\fn void ImageProcessor::consume
 *
 * created on: Jan 21, 2020
 * author: daniel
 *
 */
void ImageProcessor::consume(ImageBox box)
{
  for (ImagePtr img : box)
  {
    image::RawRGBPtr result = debayer(img->get_bits());
    if (result)
      ImageProducer::consume(image::ImageBox(result));
  }
}


} /* namespace image */
} /* namespace jupiter */
} /* namespace brt */
