/*
 * Image.cpp
 *
 *  Created on: Jul 2, 2019
 *      Author: daniel
 */

#include <string.h>
#include <iostream>
#include <fstream>
#include "image.hpp"

namespace brt
{
namespace jupiter
{
namespace image
{

/*
 * \\fn Constructor RawRGB::RawRGB
 *
 * created on: Nov 25, 2019
 * author: daniel
 *
 */
RawRGB::RawRGB(size_t w, size_t h, int bytes_per_pixel /*= 2*/)
{
  _width = w;
  _height = h;

  uint32_t srcPitch = _width * bytes_per_pixel;  // 2 bytes for RAW 12 format
  uint32_t srcImageSize = srcPitch * _height;

  if (srcImageSize > 0)
    _buffer = (uint8_t*)::malloc(srcImageSize);
  else
    _buffer = nullptr;
}

/*
 * \\fn Constructor RawRGB::RawRGB
 *
 * created on: Aug 8, 2019
 * author: daniel
 *
 */
RawRGB::RawRGB(const uint8_t* buffer, size_t w, size_t h, int bytes_per_pixel/* = 2*/)
: _buffer(nullptr)
{
  _width = w;
  _height = h;

  uint32_t srcPitch = _width * bytes_per_pixel;  // 2 bytes for RAW 12 format
  uint32_t srcImageSize = srcPitch * _height;


  if (srcImageSize > 0)
  {
    _buffer = (uint8_t*)::malloc(srcImageSize);
    memcpy(_buffer, buffer, srcImageSize);
//
//    uint16_t max = 0;
//    uint16_t* buff = reinterpret_cast<uint16_t*>(_buffer);
//    size_t ss = w * h;
//    while (ss-- != 0)
//    {
//      //*buff++ <<= 4;
//      max = (max < *buff) ? *buff : max;
//      buff++;
//    }
//    buff = reinterpret_cast<uint16_t*>(_buffer);
  }
}

/*
 * \\fn Constructor RawRGB::RawRGB
 *
 * created on: Aug 9, 2019
 * author: daniel
 *
 */
RawRGB::RawRGB(const char *raw_image_file)
: _width(0)
, _height(0)
, _buffer(nullptr)
{
  std::ifstream image_file(raw_image_file, std::ios::in | std::ios::binary);
  if (image_file.is_open())
  {
    try
    {
      uint32_t w,h, bytes;
      if (image_file.read(reinterpret_cast<char*>(&w), sizeof(w)).rdstate() != std::ios_base::goodbit)
        throw;

      if (image_file.read(reinterpret_cast<char*>(&h), sizeof(h)).rdstate() != std::ios_base::goodbit)
        throw;

      if (image_file.read(reinterpret_cast<char*>(&bytes), sizeof(bytes)).rdstate() != std::ios_base::goodbit)
        throw;


      _buffer = new uint8_t[w * h * bytes];

      if (image_file.read(reinterpret_cast<char*>(_buffer), w * h * bytes).rdstate() &
          (std::ios_base::badbit | std::ios_base::failbit) != 0)
        throw;

      _width = w;
      _height = h;
    }
    catch(...)
    {
      if (_buffer != nullptr)
      {
        delete _buffer;
        _buffer = nullptr;
      }
    }
  }
}

/*
 * \\fn Destructor RawRGB::~RawRGB
 *
 * created on: Jul 2, 2019
 * author: daniel
 *
 */
RawRGB::~RawRGB()
{
  if (_buffer != nullptr)
    free(_buffer);
}

/*
 * \\fn Constructor Image::Image
 *
 * created on: Jul 2, 2019
 * author: daniel
 *
 */
Image::Image(RawRGBPtr other_source)
: _other_source(other_source)
{

}

/*
 * \\fn Destructor Image::~Image
 *
 * created on: Jul 2, 2019
 * author: daniel
 *
 */
Image::~Image()
{
}

/*
 * \\fn RawRGBPtr Image::get_bits
 *
 * created on: Jul 2, 2019
 * author: daniel
 *
 */
RawRGBPtr Image::get_bits()
{
  return _other_source;
}

/*
 * \\fn Constructor ImageProducer::ImageProducer
 *
 * created on: Jul 2, 2019
 * author: daniel
 *
 */
ImageProducer::ImageProducer()
{

}

/*
 * \\fn Destructor ImageProducer::~ImageProducer
 *
 * created on: Jul 2, 2019
 * author: daniel
 *
 */
ImageProducer::~ImageProducer()
{

}

/*
 * \\fn void ImageProducer::register_consumer
 *
 * created on: Jul 2, 2019
 * author: daniel
 *
 */
void ImageProducer::register_consumer(ImageConsumer* consumer,const Metadata& meta /*= Metadata()*/)
{
  std::unique_lock<std::mutex> l(_mutex);
  _consumers.insert(consumer);
  _meta += meta;
}

/*
 * \\fn void ImageProducer::unregister_consumer
 *
 * created on: Jul 2, 2019
 * author: daniel
 *
 */
void ImageProducer::unregister_consumer(ImageConsumer* consumer)
{
  std::unique_lock<std::mutex> l(_mutex);
  _consumers.erase(consumer);
}

/*
 * \\fn void ImageProducer::consume
 *
 * created on: Jul 2, 2019
 * author: daniel
 *
 */
void ImageProducer::consume(ImageBox box)
{
  std::unique_lock<std::mutex> l(_mutex);

  for (ImagePtr img : box)
  {
    if (img)
      *(img.get()) += _meta;
  }

  for (ImageConsumer* consumer : _consumers)
  {
    consumer->consume(box);
  }
}


} /* namespace image */
} /* namespace jupiter */
} /* namespace brt */
