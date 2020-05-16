/**
 * 
 */

#include <image.hpp>

#include "cuda_lib.hpp"
#include "cuda_debayer.hpp"


namespace brt 
{
namespace jupiter
{

CudaLib CudaLib::_object;

/**
 * \fn  CudaLib::init
 *
 * @return  bool
 * \brief <description goes here>
 */
bool CudaLib::init()
{
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if ((err != cudaSuccess) || (count == 0))
    return false;

  if (count > 1)
  {
    int device = -1, major = -1, minor = -1;
    for (int index = 0; index < count; index++)
    {
      int value_mj = 0;
      err = cudaDeviceGetAttribute(&value_mj, cudaDevAttrComputeCapabilityMajor, index);
      if (err != cudaSuccess)
        continue;

      int value_mn = 0;
      err = cudaDeviceGetAttribute(&value_mn, cudaDevAttrComputeCapabilityMinor, index);
      if (err != cudaSuccess)
        continue;

      if ((value_mj > major) || ((value_mj == major) && (value_mn > minor)))
      {
        major = value_mj;
        minor = value_mn;
        device = index;
      }
    }

    if (device == -1)
      return false;

    err = cudaSetDevice(device);
    if (err != cudaSuccess)
      return false;
  }

  return true;
}

} // jupiter
} // brt

/**
 * \fn  initialize_cuda_lib
 *
 * @return  extern "C" int
 * \brief <description goes here>
 */
extern "C" int initialize_cuda_lib()
{
  return brt::jupiter::CudaLib::get()->init() ? 1 : 0;
}

/**
 * \fn  debayer
 *
 * @param   image : image::RawRGBPtr
 * @param  algorithm :  int 
 * @return  extern "C" image::RawRGBPtr
 * \brief <description goes here>
 */
extern "C" brt::jupiter::image::RawRGBPtr debayer(brt::jupiter::image::RawRGBPtr image, 
                                                  brt::jupiter::DebayerAlgorithm algorithm)
{
  if (!image)
    return brt::jupiter::image::RawRGBPtr();
  brt::jupiter::Debayer db;
  db.init(image->width(), image->height(), 9);
  
  switch (algorithm)
  {
  case brt::jupiter::daBiLinear:
    return db.bilinear(image);
  
  case brt::jupiter::daAHD:
    return db.ahd(image);
  
  case brt::jupiter::daMHC:
    return db.mhc(image);
  }
  
  return brt::jupiter::image::RawRGBPtr();
}