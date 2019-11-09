/*
 * WinStructures.hpp
 *
 *  Created on: Jun 28, 2019
 *      Author: daniel
 */

#ifndef WINDOW_WINSTRUCTURES_HPP_
#define WINDOW_WINSTRUCTURES_HPP_

#include <stdint.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#include <climits>
#include <vector>


namespace brt
{
namespace jupiter
{
namespace window
{

class bytestream : public std::vector<uint8_t>
{
public:
  const uint8_t*                  buf() const
  {
    return data();
  }

  template <typename T>
  bytestream& operator<<(const T& data)
  {
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&data);
    size_t cur_size = size();
    resize(cur_size + sizeof(T));
    memcpy(&at(cur_size),&data,sizeof(T));
    return *this;
  }
};

/*
 * \\enum name
 *
 * created on: Jul 1, 2019
 *
 */
enum X_Events
{
  eEventStop = 0,
  eCreateWindowEvent = 1,
  eShowImageStructure = 2
};

/*
 * \\struct X_Event
 *
 * created on: Jul 1, 2019
 *
 */
struct X_Event
{
  uint32_t  _type;

  virtual ~X_Event() {}
  virtual bytestream              serialize()
  {
    bytestream result;
    result << _type;
    return result;
  }
};

/*
 * \\struct X_StopEvent
 *
 * created on: Jul 1, 2019
 *
 */
struct X_StopEvent : public X_Event
{
  X_StopEvent() {_type = eEventStop; }
};


class Window;
/*
 * \\struct X_CreateWindowEvent
 *
 * created on: Jul 1, 2019
 *
 */
struct X_CreateWindowEvent : public X_Event
{
  X_CreateWindowEvent(int pipe_fd)
  : _wnd(nullptr)
  {
    _type = eCreateWindowEvent;
    ::read(pipe_fd, &_wnd,sizeof(_wnd));
  }

  X_CreateWindowEvent(Window *wnd)
  : _wnd(wnd)
  {
    _type = eCreateWindowEvent;
  }

  virtual bytestream              serialize()
  {
    bytestream result = X_Event::serialize();
    result << _wnd;
    return result;
  }

  Window*                         _wnd;
};


/*
 * \\struct X_ShowImageEvent
 *
 * created on: Jul 1, 2019
 *
 */
struct X_ShowImageEvent : public X_Event
{
  X_ShowImageEvent(size_t id, Window* wnd)
  : _id(id), _wnd(wnd)
  {
    _type = eShowImageStructure;
  }

  X_ShowImageEvent(int pipe_fd)
  : _wnd(nullptr)
  {
    _type = eShowImageStructure;
    ::read(pipe_fd, &_id,sizeof(_id));
    ::read(pipe_fd, &_wnd,sizeof(_wnd));
  }

  virtual bytestream              serialize()
  {
    bytestream result = X_Event::serialize();
    result << _id << _wnd;
    return result;
  }

  size_t                          _id;
  Window*                         _wnd;
};
/*
 * \\struct WinSize
 *
 * created on: Jun 28, 2019
 *
 */
struct WinSize
{
  WinSize() : _width(0), _height(0) {}
  WinSize(int width, int height) : _width(width), _height(height) {}
  WinSize(const WinSize& size) : _width(size._width), _height(size._height) {}

  int                             _width;
  int                             _height;
};

} /* namespace window */
} /* namespace jupiter */
} /* namespace brt */


#endif /* WINDOW_WINSTRUCTURES_HPP_ */
