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

#include "Image.hpp"

#define EVENT_STOP                          (1)

namespace brt
{
namespace jupiter
{

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

          bool                    activate(bool = true);
          bool                    is_active() const { return _active; }
          int                     id() const { return _id; }

          bool                    start_streaming();
          bool                    stop_streaming();

          const v4l2_format*      format() const { return &_fmt; }

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
  int                             _video_id;
  std::string                     _device_name;
  int                             _handle;

  std::atomic_bool                _terminate;
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
