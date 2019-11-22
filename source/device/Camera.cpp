/*
 * Camera.cpp
 *
 *  Created on: Nov 8, 2019
 *      Author: daniel
 */

#include "Camera.hpp"
#include "Deserializer.hpp"
#include "CameraManager.hpp"
#include "Utils.hpp"

#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <dirent.h>
#include <string.h>

namespace brt
{
namespace jupiter
{

/*
 * \\fn Constructor Camera::Camera
 *
 * created on: Nov 8, 2019
 * author: daniel
 *
 */
Camera::Camera(Deserializer* owner,int id)
: _owner (owner)
, _id(id)
, _active(false)
, _device_name()
, _handle(-1)
, _terminate(false)
, _thread()
, _io_method(IO_METHOD_MMAP)
, _fmt()
, _buffers(nullptr)
, _n_buffers(0)
{
  brt_camera_name video_name;
  video_name._deser_id = owner->id();
  video_name._camera_id = id;

  if (ioctl(DeviceManager::get()->handle(),BRT_CAMERA_GET_NAME,(unsigned long)&video_name) < 0)
    std::cerr << "Name extraction error " << errno << std::endl;

  else
  {
    if (strlen(video_name._name) > 0)
      _device_name = Utils::string_format("/dev/%s", video_name._name);
  }
}

/*
 * \\fn Camera::~Camera
 *
 * created on: Nov 8, 2019
 * author: daniel
 *
 */
Camera::~Camera()
{
  stop_streaming();
}

/*
 * \\fn bool Camera::start_streaming
 *
 * created on: Nov 8, 2019
 * author: daniel
 *
 */
bool Camera::start_streaming()
{
  std::cout << "Start streaming " << _device_name << std::endl;

  if (_thread.joinable())
    return false;

  if (!open_device())
    return false;

  if (!init_device())
    return false;

  if (!start_capturing())
    return false;

  _terminate.store(false);
  if (pipe(_pipe) == -1)
  {
    std::cerr << "pipe error:" << errno << ", "
        << strerror(errno) << std::endl;
    return false;
  }

  _thread = std::thread([](Camera* camera)
  {
    camera->main_loop();

  },this);

  return true;
}

/*
 * \\fn bool Camera::stop_streaming
 *
 * created on: Nov 8, 2019
 * author: daniel
 *
 */
bool Camera::stop_streaming()
{
  if (!_thread.joinable())
    return false;

  _terminate.store(true);
  uint32_t value = EVENT_STOP;
  ::write(_pipe[1], &value, sizeof(uint32_t));

  _thread.join();
  _thread = std::thread();

  stop_capturing();
  uninit_device();

  if (-1 == ::close(_handle))
  {
    std::cerr << "close identify " << _device_name << ":"
          << errno << ", " << strerror(errno) << std::endl;
    return false;
  }

  _handle = -1;

  return true;
}



/*
 * \\fn bool Camera::open_device
 *
 * created on: Nov 8, 2019
 * author: daniel
 *
 */
bool Camera::open_device()
{
  struct stat st;

  if (-1 == ::stat(_device_name.c_str(), &st))
  {
    std::cerr << "Cannot identify " << _device_name << ":"
          << errno << ", " << strerror(errno) << std::endl;
    return false;
  }

  if (!S_ISCHR(st.st_mode))
  {
    std::cerr << _device_name << "is not a device" << std::endl;
    return false;
  }

  _handle = ::open(_device_name.c_str(), O_RDWR /* required */| O_NONBLOCK, 0);

  if (-1 == _handle)
  {
    std::cerr << "Cannot open " << _device_name << ":"
          << errno << ", " << strerror(errno) << std::endl;
    return false;
  }

  return true;
}

/*
 * \\fn bool Camera::init_device
 *
 * created on: Nov 8, 2019
 * author: daniel
 *
 */
bool Camera::init_device()
{
  struct v4l2_capability cap;
  struct v4l2_cropcap cropcap;
  struct v4l2_crop crop;
  unsigned int min;

  if (-1 == xioctl(_handle, VIDIOC_QUERYCAP, &cap))
  {
    if (EINVAL == errno)
    {
      std::cerr << _device_name << "is not a device" << std::endl;
    }
    else
    {
      std::cerr << "VIDIOC_QUERYCAP error:" << errno << ", "
          << strerror(errno) << std::endl;
    }
    return false;
  }

  if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE))
  {
    std::cerr << _device_name << "is not a video capture device"
          << std::endl;
    return false;
  }

  // Try MMAP first
  if ((cap.capabilities & V4L2_CAP_STREAMING) != 0)
    _io_method = IO_METHOD_MMAP;
  else if ((cap.capabilities & V4L2_CAP_READWRITE) != 0)
    _io_method = IO_METHOD_READ;
  else
  {
    std::cerr << _device_name << "does not support streaming" << std::endl;
    return false;
  }

  /* Select video input, video standard and tune here. */
  memset(&cropcap, 0, sizeof(cropcap));

  cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

  if (0 == xioctl(_handle, VIDIOC_CROPCAP, &cropcap))
  {
    crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    crop.c = cropcap.defrect; /* reset to default */

    if (-1 == xioctl(_handle, VIDIOC_S_CROP, &crop))
    {
      switch (errno)
      {
      case EINVAL:
        /* Cropping not supported. */
        break;
      default:
        /* Errors ignored. */
        break;
      }
    }
  }
  else
  {
    /* Errors ignored. */
  }

  memset(&_fmt, 0, sizeof(_fmt));

  _fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (-1 == xioctl(_handle, VIDIOC_G_FMT, &_fmt))
  {
    std::cerr << "VIDIOC_G_FMT error:" << errno << ", "
        << strerror(errno) << std::endl;

    return false;
  }

  /* Buggy driver paranoia. */
  min = _fmt.fmt.pix.width * 2;
  if (_fmt.fmt.pix.bytesperline < min)
    _fmt.fmt.pix.bytesperline = min;
  min = _fmt.fmt.pix.bytesperline * _fmt.fmt.pix.height;
  if (_fmt.fmt.pix.sizeimage < min)
    _fmt.fmt.pix.sizeimage = min;

//  if (fmt_cback != nullptr)
//    fmt_cback(fmt);

  switch (_io_method)
  {
  case IO_METHOD_READ:
    return init_read();
    break;

  case IO_METHOD_MMAP:
    return init_mmap();
    break;

  default:
    break;
  }

  return false;
}

/*
 * \\fn bool Camera::uninit_device
 *
 * created on: Nov 8, 2019
 * author: daniel
 *
 */
bool Camera::uninit_device()
{
  switch (_io_method)
  {
  case IO_METHOD_READ:
    free(_buffers[0].start);
    break;

  case IO_METHOD_MMAP:
    for (int i = 0; i < _n_buffers; ++i)
      if (-1 == munmap(_buffers[i].start, _buffers[i].length))
      {
        std::cerr << "munmap error:" << errno << ", "
            << strerror(errno) << std::endl;

        return false;
      }
    break;

  default:
    break;
  }

  ::free(_buffers);
  return true;
}

/*
 * \\fn bool Camera::init_read
 *
 * created on: Nov 8, 2019
 * author: daniel
 *
 */
bool Camera::init_read()
{
  _buffers = (buffer*)calloc(1, sizeof(*_buffers));

  if (_buffers == nullptr)
  {
    std::cerr << "Out of memory" << std::endl;
    return false;
  }

  _buffers[0].length = _fmt.fmt.pix.sizeimage;
  _buffers[0].start = malloc(_fmt.fmt.pix.sizeimage);

  if (_buffers[0].start == nullptr)
  {
    std::cerr << "Out of memory" << std::endl;
    return false;
  }
  return true;
}

/*
 * \\fn bool Camera::init_mmap
 *
 * created on: Nov 8, 2019
 * author: daniel
 *
 */
bool Camera::init_mmap()
{
  v4l2_requestbuffers req;

  memset(&req, 0, sizeof(req));

  req.count = 4;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;

  if (-1 == xioctl(_handle, VIDIOC_REQBUFS, &req))
  {
    if (EINVAL == errno)
    {
      std::cerr << _device_name << " does not support "
          "memory mappingn" << std::endl;
    }
    else
    {
      std::cerr << "VIDIOC_REQBUFS error:" << errno << ", "
          << strerror(errno) << std::endl;
    }
    return false;
  }

  if (req.count < 2)
  {
    std::cerr << "Insufficient buffer memory on " <<  _device_name << std::endl;
    return false;
  }

  _buffers = (buffer*)calloc(req.count, sizeof(*_buffers));
  if (!_buffers)
  {
    std::cerr << "Out of memory" << std::endl;
    return false;
  }

  for (_n_buffers = 0; _n_buffers < req.count; ++_n_buffers)
  {
    v4l2_buffer buf;

    memset(&buf, 0, sizeof(buf));

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = _n_buffers;

    if (-1 == xioctl(_handle, VIDIOC_QUERYBUF, &buf))
    {
      std::cerr << "VIDIOC_QUERYBUF error:" << errno << ", "
          << strerror(errno) << std::endl;

      ::free(_buffers);
      _buffers = nullptr;
      return false;
    }

    _buffers[_n_buffers].length = buf.length;
    _buffers[_n_buffers].start = mmap(NULL /* start anywhere */, buf.length,
                              PROT_READ | PROT_WRITE /* required */,
                              MAP_SHARED /* recommended */, _handle, buf.m.offset);

    if (MAP_FAILED == _buffers[_n_buffers].start)
    {
      std::cerr << "mmap error:" << errno << ", "
          << strerror(errno) << std::endl;

      ::free(_buffers);
      _buffers = nullptr;
      return false;
    }
  }

  return true;
}

/*
 * \\fn bool Camera::start_capturing
 *
 * created on: Nov 8, 2019
 * author: daniel
 *
 */
bool Camera::start_capturing()
{
  v4l2_buf_type type;

  switch (_io_method)
  {
  case IO_METHOD_READ:
    /* Nothing to do. */
    break;

  case IO_METHOD_MMAP:
    for (int i = 0; i < _n_buffers; ++i)
    {
      v4l2_buffer buf;
      memset(&buf, 0, sizeof(buf));

      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;

      if (-1 == xioctl(_handle, VIDIOC_QBUF, &buf))
      {
        std::cerr << "VIDIOC_QBUF error:" << errno << ", "
            << strerror(errno) << std::endl;

        ::free(_buffers);
        _buffers = nullptr;
        return false;
      }
    }

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(_handle, VIDIOC_STREAMON, &type))
    {
      std::cerr << "VIDIOC_STREAMON error:" << errno << ", "
          << strerror(errno) << std::endl;

      ::free(_buffers);
      _buffers = nullptr;
      return false;
    }
    break;

  default:
    break;
  }
  return true;
}

/*
 * \\fn bool Camera::stop_capturing
 *
 * created on: Nov 8, 2019
 * author: daniel
 *
 */
bool Camera::stop_capturing()
{
  v4l2_buf_type type;

  switch (_io_method)
  {
  case IO_METHOD_READ:
    /* Nothing to do. */
    break;

  case IO_METHOD_MMAP:
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(_handle, VIDIOC_STREAMOFF, &type))
    {
      std::cerr << "VIDIOC_STREAMOFF error:" << errno << ", "
          << strerror(errno) << std::endl;

      return false;
    }
    break;
  }

  return true;
}

/*
 * \\fn void void Camera::main_loop
 *
 * created on: Nov 8, 2019
 * author: daniel
 *
 */
void Camera::main_loop()
{
  while (!_terminate.load())
  {
    for (;;)
    {
      fd_set fds;
      //struct timeval tv;
      int r;

      FD_ZERO(&fds);
      FD_SET(_handle, &fds);
      FD_SET(_pipe[0], &fds);

      /* Timeout. */
//      tv.tv_sec = 2;
//      tv.tv_usec = 0;

      //r = select(_handle + 1, &fds, NULL, NULL, &tv);
      r = select(std::max(_handle,_pipe[0]) + 1, &fds, nullptr, nullptr, nullptr);

      if (-1 == r)
      {
        if (EINTR != errno)
        {
          std::cerr << "select error:" << errno << ", "
              << strerror(errno) << std::endl;
        }
        continue;
      }

      if (0 == r)
        continue;

      if (FD_ISSET(_pipe[0],&fds))
      {
        uint32_t value;
        if (::read(_pipe[0], &value, sizeof(value)) != sizeof(value))
        {
          std::cerr << "select error:" << errno << ", "
              << strerror(errno) << std::endl;
        }

        else if (value == EVENT_STOP)
          break;
      }

      if (read_frame())
        break;

      /* EAGAIN - continue select loop. */
    }
  }
}

/*
 * \\fn bool Camera::read_frame
 *
 * created on: Nov 8, 2019
 * author: daniel
 *
 */
bool Camera::read_frame()
{
  v4l2_buffer buf;
  unsigned int i;

  switch (_io_method)
  {
  case IO_METHOD_READ:
    if (-1 == read(_handle, _buffers[0].start, _buffers[0].length))
    {
      switch (errno)
      {
      case EAGAIN:
        return 0;

      case EIO:
        /* Could ignore EIO, see spec. */

        /* fall through */

      default:
        std::cerr << "read error:" << errno << ", "
            << strerror(errno) << std::endl;
        return false;
      }
    }

    if (buf.length >= (_fmt.fmt.pix.width * _fmt.fmt.pix.height * 2))
    {
      image::ImageBox box(image::RawRGBPtr(new image::RawRGB((uint8_t*)_buffers[0].start,_fmt.fmt.pix.width,_fmt.fmt.pix.height)));
      consume(box);
    }
    //process_image(buffers[0].start, buffers[0].length);
    break;

  case IO_METHOD_MMAP:
    memset(&buf, 0, sizeof(buf));

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(_handle, VIDIOC_DQBUF, &buf))
    {
      switch (errno)
      {
      case EAGAIN:
        return 0;

      case EIO:
        /* Could ignore EIO, see spec. */

        /* fall through */

      default:
        std::cerr << "VIDIOC_DQBUF error:" << errno << ", "
            << strerror(errno) << std::endl;
        return false;
      }
    }

    assert(buf.index < _n_buffers);

    if (buf.length >= (_fmt.fmt.pix.width * _fmt.fmt.pix.height * 2))
    {
      image::ImageBox box(image::RawRGBPtr(new image::RawRGB((uint8_t*)_buffers[buf.index].start,_fmt.fmt.pix.width,_fmt.fmt.pix.height)));
      consume(box);
    }
    //process_image(_buffers[buf.index].start, buf.bytesused?buf.bytesused:buf.length);

    if (-1 == xioctl(_handle, VIDIOC_QBUF, &buf))
    {
      std::cerr << "VIDIOC_QBUF error:" << errno << ", "
          << strerror(errno) << std::endl;
      return false;
    }
    break;

  default:
    break;
  }

  return true;
}




} /* namespace jupiter */
} /* namespace brt */
