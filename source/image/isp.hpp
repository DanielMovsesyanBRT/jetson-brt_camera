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

          void                    add_camera(Camera*,ImageProducer*);
          void                    stop();

private:
  bool                            _group;
  struct CameraBlock
  {
    Camera*                         _cam;
    ImageProducer*                  _ip;
    uint32_t                        _num_captured;
    std::vector<uint32_t>           _histogram;

    double                          _k0; // previous coefficient
    double                          _m0; // previous mean value

    std::string                     _name;
    int                             _id;
  };

  std::vector<CameraBlock>        _cameras;
  std::mutex                      _mutex;

};

} /* namespace image */
} /* namespace jupiter */
} /* namespace brt */

using isp = brt::jupiter::image::ISP;

#endif /* SOURCE_IMAGE_ISP_HPP_ */
