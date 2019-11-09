/*
 * WindowManager.cpp
 *
 *  Created on: Jun 28, 2019
 *      Author: daniel
 */

#include "WindowManager.hpp"
#include "Window.hpp"

#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>

using namespace std::chrono_literals;

namespace brt
{
namespace jupiter
{
namespace window
{

WindowManager WindowManager::_object;

/*
 * \\fn Constructor WindowManager::WindowManager
 *
 * created on: Jun 28, 2019
 * author: daniel
 *
 */
WindowManager::WindowManager()
: _display(nullptr)
, _blackColor(0)
, _whiteColor(0)
, _thread()
, _cv()
, _mutex()
, _terminate(false)
{
  srand(time(nullptr));
}

/*
 * \\fn Destructor WindowManager::~WindowManager
 *
 * created on: Jun 28, 2019
 * author: daniel
 *
 */
WindowManager::~WindowManager()
{
}


/*
 * \\fn void WindowManager::init
 *
 * created on: Jun 28, 2019
 * author: daniel
 *
 */
void WindowManager::init()
{
  if (_display != nullptr)
    return;

  // Messaging Pipe
  pipe(_pipe);

  _thread = std::thread([&](WindowManager *winm)
  {
    winm->x_loop();
  },this);


//  XInitThreads();
}

/*
 * \\fn void WindowManager::release
 *
 * created on: Jun 28, 2019
 * author: daniel
 *
 */
void WindowManager::release()
{
  if (_display == nullptr)
    return;

  X_StopEvent se;
  bytestream buff = se.serialize();
  ::write(_pipe[1],buff.buf(), buff.size());
  _thread.join();

  XCloseDisplay(_display);
  _display = nullptr;
}

/*
 * \\fn WinSize WindowManager::resolution
 *
 * created on: Jun 28, 2019
 * author: daniel
 *
 */
WinSize WindowManager::resolution(size_t screen_id) const
{
  if (screen_id >= _screens.size())
    return WinSize(-1,-1);

  return WinSize(_screens[screen_id]->width, _screens[screen_id]->height);
}

/*
 * \\fn Screen WindowManager::default_screen
 *
 * created on: Jun 28, 2019
 * author: daniel
 *
 */
int WindowManager::default_screen() const
{
  return DefaultScreen(_display);
}

/*
 * \\fn WindowManager::create_window
 *
 * created on: Jun 28, 2019
 * author: daniel
 *
 */
Window* WindowManager::create_window(const char* title, size_t num_views, size_t width, size_t height,
              int x /*= RANDOM_POS*/, int y /*= RANDOM_POS*/)
{
  int rows, cols;
  double min_square = std::sqrt((double)num_views);
  cols = (int)min_square;
  rows = (int)min_square;

  if (min_square > std::floor(min_square))
    cols++;

  Window* result = new Window(title, x, y, cols, rows, width, height, DefaultRootWindow(_display));
  {
    std::unique_lock<std::mutex> l(_mutex);
    _wind_set.insert(result);
  }

  X_CreateWindowEvent ce(result);
  post_message(ce.serialize());
  return result;
}

/*
 * \\fn void WindowManager::post_message
 *
 * created on: Jul 1, 2019
 * author: daniel
 *
 */
void WindowManager::post_message(const bytestream& msg)
{
  ::write(_pipe[1],msg.buf(),msg.size());
}

/*
 * \\fn void WindowManager::x_loop
 *
 * created on: Jun 28, 2019
 * author: daniel
 *
 */
void WindowManager::x_loop()
{
  XSetErrorHandler(x_error_handler);
  _display = XOpenDisplay(nullptr);
  if (_display == nullptr)
  {
    std::cerr << "Error: cannot open display " << std::endl;
    return;
  }

  _blackColor = BlackPixel(_display, DefaultScreen(_display));
  _whiteColor = WhitePixel(_display, DefaultScreen(_display));

  int number_of_screens = ScreenCount(_display);
  for (int index = 0; index < number_of_screens; index++)
  {
    Screen* sc = ScreenOfDisplay(_display, index);
    if (sc != nullptr)
      _screens.push_back(sc);
  }
  int     x11_fd = ConnectionNumber(_display);
  fd_set  in_fds;
  struct  timeval tv;

  _terminate = false;

  while(!_terminate)
  {
    FD_ZERO(&in_fds);
    FD_SET(_pipe[0],&in_fds);
    FD_SET(x11_fd,&in_fds);

    tv.tv_usec = 0;
    tv.tv_sec = 1;

    if (::select(std::max(_pipe[0],x11_fd) + 1, &in_fds, nullptr, nullptr, &tv) > 0)
    {
      if (FD_ISSET(_pipe[0], &in_fds))
        process_local_event();

      if (FD_ISSET(x11_fd, &in_fds))
      {
        XEvent e;
        while(!_terminate && XPending(_display))
        {
          XNextEvent(_display, &e);
          if (e.type == DestroyNotify) // MapNotify)
            _terminate = true;

          _mutex.lock();
          window_set::iterator iter = std::find_if(_wind_set.begin(), _wind_set.end(), [e](Window* wnd)->bool
          {
            return (wnd->handle() ==  e.xany.window);
          });

          if (iter != _wind_set.end())
            (*iter)->processEvent(e);

          _mutex.unlock();

        }
      }
    }
    else //timeout
      continue;
  }
}

/*
 * \\fn void WindowManager::process_local_event
 *
 * created on: Jul 1, 2019
 * author: daniel
 *
 */
void WindowManager::process_local_event()
{
  uint32_t  type;
  if (::read(_pipe[0], &type, sizeof(type)) != sizeof(type))
    return;

  switch (type)
  {
  case eEventStop:
    _terminate = true;
    break;

  case eCreateWindowEvent:
    {
      X_CreateWindowEvent ce(_pipe[0]);
      if (ce._wnd != nullptr)
        ce._wnd->processEvent(&ce);
    }
    break;

  case eShowImageStructure:
    {
      X_ShowImageEvent se(_pipe[0]);
      if (se._wnd != nullptr)
        se._wnd->processEvent(&se);
    }
    break;

  default:
    break;
  }
}

/*
 * \\fn int WindowManager::x_error_handler
 *
 * created on: Jun 29, 2019
 * author: daniel
 *
 */
int WindowManager::x_error_handler(Display* display, XErrorEvent* err)
{
  char text[1024];
  XGetErrorText(display,err->error_code,text,sizeof(text));
  std::cerr << "Error: " << text << std::endl;
  std::cerr << "Request code: " << err->request_code << std::endl;
  std::cerr << "Minor code: " << err->minor_code << std::endl;
  return 0;
}

} /* namespace window */
} /* namespace jupiter */
} /* namespace brt */
