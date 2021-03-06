/*
 * Camera.cpp
 *
 *  Created on: Nov 8, 2019
 *      Author: daniel
 */

#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <dirent.h>
#include <string.h>
#include <sstream>

#include "camera.hpp"

#include <utils.hpp>
#include <metadata.hpp>

#include "deserializer.hpp"
#include "device_manager.hpp"

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
Camera::Camera(Deserializer* owner,int id,const Value::byte_buffer& bb /*= Value::byte_buffer()*/)
: _owner (owner)
, _id(id)
, _active(false)
, _device_name()
, _handle(-1)
, _terminate(false)
, _skip_frames(0)
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

  if (!bb.empty())
    memcpy(&_camera_params, &bb.front(), std::min(bb.size(), sizeof(_camera_params)));

  
  std::string db_alg = gm::get()["args"].get<std::string>("debayering","mhc");
  if (db_alg.compare("mhc") == 0)
    _ip.set_debayer_algorithm(daMHC);

  else if (db_alg.compare("ahd") == 0)
    _ip.set_debayer_algorithm(daAHD);
  
  else if (db_alg.compare("bilinear") == 0)
    _ip.set_debayer_algorithm(daBiLinear);

  register_consumer(&_ip);
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
  unregister_consumer(&_ip);
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

  std::cout << "Stopping device " << _device_name << std::endl;

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

  std::cout << "Stopped device " << _device_name << std::endl;

  return true;
}

/*
 * \\fn void Camera::set_exposure
 *
 * created on: Nov 24, 2019
 * author: daniel
 *
 */
void Camera::set_exposure(double ms)
{
  std::vector<Value> args;
  args.push_back(Value().set<int>(_id));
  args.push_back(Value().set(ms));

  _skip_frames = 4;
  _owner->run_macro("set_exposure", args);
}

/*
 * \\fn Camera::get_exposure
 *
 * created on: Nov 26, 2019
 * author: daniel
 *
 */
double Camera::get_exposure()
{
  std::vector<Value> args;
  args.push_back(Value().set<int>(_id));

  return _owner->run_macro("get_exposure", args);
}

/*
 * \\fn double Camera::get_temperature
 *
 * created on: Dec 3, 2019
 * author: daniel
 *
 */
double Camera::get_temperature(int temp_sensor_id)
{
  std::vector<Value> args;
  args.push_back(Value().set<int>(_id));
  args.push_back(Value().set<int>(temp_sensor_id));

  return _owner->run_macro("get_temperature", args);
}


/*
 * \\fn void Camera::read_exposure
 *
 * created on: Nov 24, 2019
 * author: daniel
 *
 */
void Camera::read_exposure()
{
  std::vector<Value> args;
  args.push_back(Value().set<int>(_id));

  _owner->run_macro("read_exposure", args);
}


/*
 * \\fn void Camera::set_gain
 *
 * created on: Nov 24, 2019
 * author: daniel
 *
 */
void Camera::set_gain(eCameraGain gain)
{
  std::vector<Value> args;
  args.push_back(Value().set<int>(_id));
  args.push_back(Value().set<int>(gain, 2));

  _skip_frames = 4;
  _owner->run_macro("set_gain", args);
}

/*
 * \\fn void Camera::get_camera_parameters_json
 *
 * created on: Mar 19, 2020
 * author: daniel
 *
 */
void Camera::get_camera_parameters_json(std::string& jstring)
{
  const CameraParameters* cam_params = get_camera_parameters_bin();
  const Intrinsics primary_intrinsics = cam_params->_lense_parameters;
  const Intrinsics secondary_intrinsics = cam_params->_companion_lense_params;
  const Extrinsics extrinsics = cam_params->_stereo_params;

  std::stringstream json_stream;
  json_stream << "{ \"camera_name\": \"" << _device_name << "\", " << std::endl << "    ";
  json_stream << "\"K1\": [" << primary_intrinsics._fx << ", 0, " << primary_intrinsics._cx << ", ";
  json_stream << "0, " << primary_intrinsics._fy << ", " << primary_intrinsics._cy << ", ";
  json_stream << "0, 0, 1], " << std::endl << "    ";
  json_stream << "\"K2\": [" << secondary_intrinsics._fx << ", 0, " << secondary_intrinsics._cx << ", ";
  json_stream << "0, " << secondary_intrinsics._fy << ", " << secondary_intrinsics._cy << ", ";
  json_stream << "0, 0, 1], " << std::endl << "    ";
  json_stream << "\"D1\": [" << primary_intrinsics._k1 << ", " << primary_intrinsics._k2 << ", ";
  json_stream << primary_intrinsics._p1 << ", " << primary_intrinsics._p2 << ", ";
  json_stream << primary_intrinsics._k3 << ", " << primary_intrinsics._k4 << ", ";
  json_stream << primary_intrinsics._k5 << ", " << primary_intrinsics._k6 << "], " << std::endl << "    ";
  json_stream << "\"D2\": [" << secondary_intrinsics._k1 << ", " << secondary_intrinsics._k2 << ", ";
  json_stream << secondary_intrinsics._p1 << ", " << secondary_intrinsics._p2 << ", ";
  json_stream << secondary_intrinsics._k3 << ", " << secondary_intrinsics._k4 << ", ";
  json_stream << secondary_intrinsics._k5 << ", " << secondary_intrinsics._k6 << "]," << std::endl << "    ";
  json_stream << "\"R\": [";
  json_stream << extrinsics._rot[0][0] << ", " << extrinsics._rot[0][1] << ", " << extrinsics._rot[0][2] << ", ";
  json_stream << extrinsics._rot[1][0] << ", " << extrinsics._rot[1][1] << ", " << extrinsics._rot[1][2] << ", ";
  json_stream << extrinsics._rot[2][0] << ", " << extrinsics._rot[2][1] << ", " << extrinsics._rot[2][2] << "]," << std::endl << "    ";
  json_stream << "\"T\": [" << extrinsics._tx << ", " << extrinsics._ty << ", " << extrinsics._tz << "]," << std::endl << "    ";
  json_stream << "\"stereoRotRodriguesVec\": [" << extrinsics._rx << ", " << extrinsics._ry << ", " << extrinsics._rz << "]" << std::endl << "}";
  jstring = json_stream.str();
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

  if (-1 == xioctl(_handle, static_cast<int>(VIDIOC_QUERYCAP), &cap))
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

  if (0 == xioctl(_handle, static_cast<int>(VIDIOC_CROPCAP), &cropcap))
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
  if (-1 == xioctl(_handle, static_cast<int>(VIDIOC_G_FMT), &_fmt))
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


  _ip.init(_fmt.fmt.pix.width, _fmt.fmt.pix.height, 9);

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
    for (unsigned int i = 0; i < _n_buffers; ++i)
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

  if (-1 == xioctl(_handle, static_cast<int>(VIDIOC_REQBUFS), &req))
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

    if (-1 == xioctl(_handle, static_cast<int>(VIDIOC_QUERYBUF), &buf))
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
    for (unsigned int i = 0; i < _n_buffers; ++i)
    {
      v4l2_buffer buf;
      memset(&buf, 0, sizeof(buf));

      buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buf.memory = V4L2_MEMORY_MMAP;
      buf.index = i;

      if (-1 == xioctl(_handle, static_cast<int>(VIDIOC_QBUF), &buf))
      {
        std::cerr << "VIDIOC_QBUF error:" << errno << ", "
            << strerror(errno) << std::endl;

        ::free(_buffers);
        _buffers = nullptr;
        return false;
      }
    }

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(_handle, static_cast<int>(VIDIOC_STREAMON), &type))
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
    if (-1 == xioctl(_handle, static_cast<int>(VIDIOC_STREAMOFF), &type))
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
  pthread_setname_np(pthread_self(), _device_name.c_str());


  while (!_terminate.load())
  {
    for (;;)
    {
      fd_set fds;
      struct timeval tv;
      int r;

      FD_ZERO(&fds);
      FD_SET(_handle, &fds);
      FD_SET(_pipe[0], &fds);

      /* Timeout. */
      tv.tv_sec = 2;
      tv.tv_usec = 0;

      //r = select(_handle + 1, &fds, NULL, NULL, &tv);
      r = select(std::max(_handle,_pipe[0]) + 1, &fds, nullptr, nullptr, &tv);

      if (-1 == r)
      {
        if (EINTR != errno)
        {
          std::cerr << "select error:" << errno << ", "
              << strerror(errno) << std::endl;
        }
        break;
      }

      if (0 == r)
      {
        std::cerr << "timeout" << std::endl;
        break;
      }

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

      if (FD_ISSET(_handle,&fds))
        read_frame();
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
      image::RawRGBPtr raw12(new image::RawRGB((uint8_t*)_buffers[0].start,_fmt.fmt.pix.width,_fmt.fmt.pix.height, 16));
      if (_skip_frames-- == 0)
      {
        consume(image::ImageBox(raw12).set_meta(Metadata().set<unsigned long>("time_tag",time(nullptr))));
        _skip_frames = 0;
      }
    }
    break;

  case IO_METHOD_MMAP:
    memset(&buf, 0, sizeof(buf));

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(_handle, static_cast<int>(VIDIOC_DQBUF), &buf))
    {
      switch (errno)
      {
      case EAGAIN:
        return false;

      case EIO:
        /* Could ignore EIO, see spec. */

        /* fall through */

      default:
        std::cerr << "VIDIOC_DQBUF error:" << errno << ", "
            << strerror(errno) << std::endl;
        return false;
      }
    }

    if ((buf.flags & V4L2_BUF_FLAG_ERROR) != 0)
    {
      std::cerr << "Data of " << _device_name << " is corrupted" << std::endl;
      return false;
    }

    assert(buf.index < _n_buffers);

    if (buf.length >= (_fmt.fmt.pix.width * _fmt.fmt.pix.height * 2))
    {
      image::RawRGBPtr raw12(new image::RawRGB((uint8_t*)_buffers[buf.index].start,_fmt.fmt.pix.width,_fmt.fmt.pix.height, 16));
      if (_skip_frames-- == 0)
      {
        consume(image::ImageBox(raw12).set_meta(Metadata().set<unsigned long>("time_tag",time(nullptr))));
        _skip_frames = 0;
      }
    }

    if (-1 == xioctl(_handle, static_cast<int>(VIDIOC_QBUF), &buf))
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
