/*
 * isp.hpp
 *
 *  Created on: Nov 26, 2019
 *      Author: daniel
 */

#ifndef SOURCE_IMAGE_ISP_HPP_
#define SOURCE_IMAGE_ISP_HPP_

#include "image.hpp"
#include <vector>
#include <mutex>

namespace brt
{
namespace jupiter
{
class Camera;

namespace image
{

/*
 * \\class ISP
 *
 * created on: Nov 26, 2019
 *
 */
class ISP : public ImageConsumer
{
public:
  ISP(bool group = false);
  virtual ~ISP();

  virtual void                    consume(ImageBox);

          void                    add_camera(Camera*);
          void                    stop();

private:
  bool                            _group;
  struct CameraBlock
  {
    Camera*                         _cam;
    uint32_t                        _num_captured;
    int                             _max;
    uint32_t                        _max_val;
  };

  std::vector<CameraBlock>        _cameras;
  std::mutex                      _mutex;

};

} /* namespace image */
} /* namespace jupiter */
} /* namespace brt */

using isp = brt::jupiter::image::ISP;

#endif /* SOURCE_IMAGE_ISP_HPP_ */
