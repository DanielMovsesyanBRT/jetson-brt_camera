/**
 *
 * Author : Author
 * Created On : Wed May 20 2020 - ${TIME}
 * File : encoder.cpp
 *
 */


#include "encoder.hpp"
#include <linux/videodev2.h>


#define V4L2_PIX_FMT_H265     v4l2_fourcc('H', '2', '6', '5')


int convert(uint32_t width, uint32_t height, uint32_t input_format, const uint8_t* input_buffer, uint32_t output_format);

namespace brt
{
namespace jupiter
{

/**
 * \fn  Encoder::consume
 *
 * @param   box : image::ImageBox
 * \brief <description goes here>
 */
void Encoder::consume(image::ImageBox box)
{
  for (auto img : box)
  {
    image::RawRGBPtr yuv;
    image::RawRGBPtr bt =img->get_bits();
    if (bt)
      yuv = _converter.rgb_2_yuv(bt);

    if (yuv)
      convert(yuv->width(), yuv->height(), V4L2_PIX_FMT_YUV444M, yuv->bytes(), V4L2_PIX_FMT_H265);
    
  }
}


} // namespace jupiter
} // namespace brt


