/**
 *
 * Author : Author
 * Created On : Wed May 20 2020 - ${TIME}
 * File : cuda_color_convert.cpp
 *
 */


#include "cuda_color_convert.hpp"
#include <cuda_mem.hpp>
#include <algorithm>

#define DEFAULT_NUMBER_OF_THREADS           (64)

namespace brt
{
namespace jupiter
{

/**
 * \struct RGB
 *
 * \brief <description goes here>
 */
struct RGB
{
  uint16_t                        _r;
  uint16_t                        _g;
  uint16_t                        _b;
};


/**
 * \struct YUV
 *
 * \brief <description goes here>
 */
struct YUV
{
  __device__ YUV() : _y(0), _u(0), _v(0) {}
  uint8_t                         _y;
  int8_t                          _u;
  int8_t                          _v;

  /**
   * \fn  from
   *
   * @param  rgb : RGB& 
   * @return  __device__ void
   * \brief 
   *    Y =  0.299R + 0.587G + 0.114B
   *    U = -0.147R - 0.289G + 0.436B
   *    V =  0.615R - 0.515G - 0.100B
   */
  __device__ void                 from(const RGB& rgb)
  {
    int32_t value;
    /* Y */ 
    value = (int32_t)(0.299 * rgb._r + 0.587 * rgb._g + 0.114 * rgb._b);
    _y = (uint8_t)(value >> 8);

    /* U */
    value = (int32_t)(-0.147 * rgb._r - 0.289 * rgb._g + 0.436 * rgb._b);
    _u = (int8_t)(value >> 8);

    /* V */
    value = (int32_t)(0.615 * rgb._r - 0.515 * rgb._g - 0.100 * rgb._b);
    _v = (int8_t)(value >> 8);
  }
};


/**
 * \fn  cuda_rgb_2_yuv
 *
 * @param  width : size_t 
 * @param  height :  size_t 
 * @param  raw :  const RGB* 
 * @param  out :  uint8_t* 
 * @return  __global__ void
 * \brief <description goes here>
 */
__global__ void cuda_rgb_2_yuv(size_t width, size_t height, const RGB* raw, uint8_t* out)
{
  uint8_t*  y_plane = out;
  int8_t*   u_plane = (int8_t*)(y_plane + (width * height));
  int8_t*   v_plane = (int8_t*)(u_plane + (width * height));

  int x0 = ((blockIdx.x * blockDim.x) + threadIdx.x);
  int y0 = ((blockIdx.y * blockDim.y) + threadIdx.y);
  int g_loc = x0 + y0 * width; // input offset

  YUV yuv;
  yuv.from(raw[g_loc]);

  y_plane[g_loc] = yuv._y;
  u_plane[g_loc] = yuv._u;
  v_plane[g_loc] = yuv._v;
}


/**
 * \class Convert_impl
 *
 * \brief <description goes here>
 */
class Convert_impl
{
public:
  Convert_impl() {}
  virtual ~Convert_impl() {}

          image::RawRGBPtr        rgb_2_yuv(image::RawRGBPtr img);

private:
  CudaPtr<uint8_t>                _buffer;
  CudaPtr<RGB>                    _img;
};


/**
 * \fn  Convert_impl::rgb_2_yuv
 *
 * @param   img : image::RawRGBPtr
 * @return  image::RawRGBPtr
 * \brief <description goes here>
 */
image::RawRGBPtr Convert_impl::rgb_2_yuv(image::RawRGBPtr img)
{
  size_t size = 3 * img->width() * img->height();

  if (_buffer.size() != size)
    _buffer = CudaPtr<uint8_t>(size);

  if (!_buffer)
    return image::RawRGBPtr();

  int thx = std::min(DEFAULT_NUMBER_OF_THREADS, (1 << __builtin_ctz( img->width() )) );
  if (thx == 0)
    thx = 1;

  int thy = std::min(DEFAULT_NUMBER_OF_THREADS, (1 << __builtin_ctz( img->height() )) );
  if (thy == 0)
    thy = 1;

  int blkx = img->width() / thx;
  if ((img->width() % thx) != 0)
    blkx++;

  int blky = img->height() / thy;
  if ((img->height() % thy) != 0)
    blky++;

  _img.put((RGB*)img->bytes(),img->width() * img->height());

  dim3 threads(thx, thy);
  dim3 blocks(blkx, blky);

  cuda_rgb_2_yuv<<<blocks,threads>>>(img->width(), img->height(), _img.ptr(), _buffer.ptr());

  image::RawRGBPtr result(new image::RawRGB(img->width(), img->height(), 8, image::eRGB));
  _buffer.get((uint8_t*)result->bytes(), result->width() * result->height());

  return result;
}


/**
 * \fn  constructor Convert::Convert
 *
 * \brief <description goes here>
 */
Convert::Convert()
{
  _impl = new Convert_impl();
}

/**
 * \fn  destructor Convert::~Convert
 *
 * \brief <description goes here>
 */
Convert::~Convert()
{
  delete _impl;
}

/**
 * \fn  Convert::rgb_2_yuv
 *
 * @param   img : image::RawRGBPtr
 * @return  image::RawRGBPtr
 * \brief <description goes here>
 */
image::RawRGBPtr Convert::rgb_2_yuv(image::RawRGBPtr img)
{
  std::lock_guard<std::mutex> l(_mutex);
  return _impl->rgb_2_yuv(img);
}

} // namespace jupiter
} // namespace brt