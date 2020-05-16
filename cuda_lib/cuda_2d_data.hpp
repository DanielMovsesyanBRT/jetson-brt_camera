/*
 * cuda_3d_data.hpp
 *
 *  Created on: Feb 14, 2020
 *      Author: daniel
 */

#ifndef BRT_COMMON_PRIVATE_CUDA_CUDA_2D_DATA_HPP_
#define BRT_COMMON_PRIVATE_CUDA_CUDA_2D_DATA_HPP_

#include <atomic>
#include <stddef.h>


#ifndef __device__
#define __device__
#endif

namespace brt {
namespace jupiter {


/*
 * \\class Cuda2DData
 *
 * created on: Feb 14, 2020
 *
 */
class Cuda2DData
{
public:
  Cuda2DData(size_t xsize,size_t ysize);
  virtual ~Cuda2DData();

          void addref()  { _ref_cnt++; }

          void release()
          {
            if (--_ref_cnt == 0)
              delete this;
          }

          void                    memset(int value);

          void                    from_host(void* data,int width,int height,int xoffset, int yoffset);
          void                    to_host(void* data,int width,int height,int xoffset, int yoffset);

          size_t                  width() const;
          size_t                  height() const;

          cudaPitchedPtr          mem() { return _pitchedDevPtr; }

          bool                    valid() const { return _valid; }
private:
  cudaPitchedPtr                  _pitchedDevPtr;
  bool                            _valid;
  std::atomic_int_fast32_t        _ref_cnt;
};

}
}


#endif /* BRT_COMMON_PRIVATE_CUDA_CUDA_2D_DATA_HPP_ */

