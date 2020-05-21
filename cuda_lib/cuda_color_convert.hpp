/**
 *
 * Author : Author
 * Created On : Wed May 20 2020 - ${TIME}
 * File : cuda_color_convert.hpp
 *
 */

#include "image.hpp"
#include <mutex>

namespace brt
{
namespace jupiter
{

class Convert_impl;
/**
 * \class Convert
 *
 * \brief <description goes here>
 */
class Convert
{
public:
  Convert();
  virtual ~Convert();

          image::RawRGBPtr        rgb_2_yuv(image::RawRGBPtr img);

private:
  std::mutex                      _mutex;
  Convert_impl*                   _impl;
};

} // namespace jupiter
} // namespace brt