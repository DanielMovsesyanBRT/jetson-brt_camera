/*
 * Camera.hpp
 *
 *  Created on: Nov 8, 2019
 *      Author: daniel
 */

#ifndef SOURCE_DEVICE_CAMERA_HPP_
#define SOURCE_DEVICE_CAMERA_HPP_

#include <atomic>
#include <thread>
#include <string>

#include <sys/ioctl.h>
#include <linux/videodev2.h>

#include <image.hpp>
//#include "image_processor.hpp"

#define EVENT_STOP                          (1)

namespace brt
{
namespace jupiter
{


enum eCameraGain
{
//###       0 = 1/8x    5 = 4/6x        10 = 6/3x
//###       1 = 2/8x    6 = 4/5x        11 = 7/3x
//###       2 = 2/7x    7 = 5/5x (1)    12 = 7/2x
//###       3 = 3/7x    8 = 5/4x        13 = 8/2x
//###       4 = 3/6x    9 = 6/4x        14 = 8/1x

  eCG_1_DIV_8_X = 0,
  eCG_2_DIV_8_X = 1,
  eCG_2_DIV_7_X = 2,
  eCG_3_DIV_7_X = 3,
  eCG_3_DIV_6_X = 4,
  eCG_4_DIV_6_X = 5,
  eCG_4_DIV_5_X = 6,
  eCG_5_DIV_5_X = 7, // 1x
  eCG_5_DIV_4_X = 8,
  eCG_6_DIV_4_X = 9,
  eCG_6_DIV_3_X = 10,
  eCG_7_DIV_3_X = 11,
  eCG_7_DIV_2_X = 12,
  eCG_8_DIV_2_X = 13,
  eCG_8_DIV_1_X = 14
};

class Deserializer;

/*
 * \\class Camera
 *
 * created on: Nov 8, 2019
 *
 */
class Camera : public image::ImageProducer
{
public:
  Camera(Deserializer* owner,int id);
  virtual ~Camera();

          int                     id() const { return _id; }

          bool                    start_streaming();
          bool                    stop_streaming();

          const v4l2_format*      format() const { return &_fmt; }

          void                    set_exposure(double ms);
          double                  get_exposure();
          double                  get_temperature(int temp_sensor_id);

          void                    read_exposure();
          void                    set_gain(eCameraGain);

          std::string             name() const { return _device_name; }
private:
          bool                    open_device();

          bool                    init_device();
          bool                    uninit_device();

          bool                    init_read();
          bool                    init_mmap();
          bool                    start_capturing();
          bool                    stop_capturing();


          void                    main_loop();
          bool                    read_frame();


          inline int xioctl(int fh, int request, void *arg)
          {
            int r;

            do
            {
              r = ioctl(fh, request, arg);
            }
            while (-1 == r && EINTR == errno);

            return r;
          }
private:
  Deserializer*                   _owner;
  int                             _id;
  bool                            _active;
  std::string                     _device_name;
  int                             _handle;

  std::atomic_bool                _terminate;
  std::atomic_int                 _skip_frames;
  std::thread                     _thread;
  int                             _pipe[2];

  enum io_method
  {
    IO_METHOD_READ, IO_METHOD_MMAP
  }                               _io_method;
  v4l2_format                     _fmt;

  struct buffer
  {
    void *start;
    size_t length;
  }                               *_buffers;
  unsigned int                    _n_buffers;
};

} /* namespace jupiter */
} /* namespace brt */

#endif /* SOURCE_DEVICE_CAMERA_HPP_ */
