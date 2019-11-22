/*
 * WindowManager.cpp
 *
 *  Created on: Jun 28, 2019
 *      Author: daniel
 */

#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>

#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include "window.hpp"
#include "window_manager.hpp"
#include "Utils.hpp"

#define _PATH_PROCNET_X11                   "/tmp/.X11-unix"
#define _PATH_PROCNET_TCP                   "/proc/net/tcp"
#define _PATH_PROCNET_TCP6                  "/proc/net/tcp6"
#define X11_PORT_MIN                        (6000)
#define X11_PORT_MAX                        (6100)

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
: _wind_set()
    // _display(nullptr)
// , _blackColor(0)
// , _whiteColor(0)
// , _thread()
, _cv()
, _mutex()
// , _terminate(false)
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
  XInitThreads();

  // Check for default display
  char* display_name = getenv("DISPLAY");
  //if ((display_name == nullptr) || (strlen(display_name) == 0))
  {
    std::vector<std::string> displays = Utils::enumerate_displays();
    if (displays.size() == 0)
    {
      if ((display_name == nullptr) || (strlen(display_name) == 0))
        _default_display = display_name;
    }

    else if (displays.size() == 1)
    {
      _default_display = displays[0];
    }
    else
    {
      char buffer[1024];
      std::string line;
      int id;

      do
      {
        std::cout << "Please select default display " << std::endl;

        for (size_t index = 0; index < displays.size(); index++)
        {
          std::cout << (index + 1) << ") " << displays[index];
          if ((display_name != nullptr) && (displays[index].compare(display_name) == 0))
            std::cout << "[default]";

          std::cout << std::endl;
        }

        std::cin >> id;
        if ((id < 1) || (id > displays.size()))
        {
          if ((display_name == nullptr) || (strlen(display_name) == 0))
          {
            _default_display = display_name;
            break;
          }

          std::cout << "Invalid entry" << std::endl;
        }
      }
      while((id < 1) || (id > displays.size()));

      if ((id >= 1) && (id <= displays.size()))
        _default_display = displays[id - 1];
    }
  }
}

/*
 * \\fn Context WindowManager::get_context
 *
 * created on: Nov 20, 2019
 * author: daniel
 *
 */
Context WindowManager::get_context(const char* display_name,bool create /*= false*/)
{
  if (display_name == nullptr)
    display_name = _default_display.c_str();

  std::unordered_set<_Context*>::iterator iter =
      std::find_if(_display_db.begin(), _display_db.end(),[display_name](_Context* db)
  {
    return db->_display_name == display_name;
  });

  if (iter == _display_db.end())
  {
    if (!create)
      return nullptr;

    _Context* db = new _Context;
    db->_display_name = (display_name != nullptr) ? display_name : "";
    pipe(db->_pipe);
    db->_terminate.store(false);

    db->_thread  = std::thread([](WindowManager* m, _Context* d)
    {
      m->x_loop(d);
    },this, db);

    _mutex.lock();
    iter  = _display_db.insert(db).first;
    _mutex.unlock();
  }

  return (*iter);
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
  while (_wind_set.size())
  {
    (*_wind_set.begin())->close();
    _wind_set.erase(_wind_set.begin());
  }

  while (_display_db.size())
  {
    LStopEvent se;

    post_message(*_display_db.begin(), se.serialize());
    (*_display_db.begin())->_thread.join();
    delete (*_display_db.begin());
    _display_db.erase(_display_db.begin());
  }
}

/*
 * \\fn Display* WindowManager::display
 *
 * created on: Nov 19, 2019
 * author: daniel
 *
 */
Display* WindowManager::display(Context ctx) const
{
  if (ctx == nullptr)
    return nullptr;

  return reinterpret_cast<_Context*>(ctx)->_display;
}

/*
 * \\fn Screen WindowManager::default_screen
 *
 * created on: Jun 28, 2019
 * author: daniel
 *
 */
int WindowManager::default_screen(Context ctx) const
{
  if (ctx == nullptr)
    return -1;

  return DefaultScreen(display(ctx));
}

/*
 * \\fn int WindowManager::black_color
 *
 * created on: Nov 19, 2019
 * author: daniel
 *
 */
int WindowManager::black_color(Context ctx) const
{
  if (ctx == nullptr)
    return -1;

  return reinterpret_cast<_Context*>(ctx)->_blackColor;
}

/*
 * \\fn WindowManager::white_color
 *
 * created on: Nov 19, 2019
 * author: daniel
 *
 */
int WindowManager::white_color(Context ctx) const
{
  if (ctx == nullptr)
    return -1;

  return reinterpret_cast<_Context*>(ctx)->_whiteColor;
}

/*
 * \\fn WinSize WindowManager::resolution
 *
 * created on: Jun 28, 2019
 * author: daniel
 *
 */
WinSize WindowManager::resolution(Context context,size_t screen_id) const
{
  _Context *ctx = reinterpret_cast<_Context*>(context);
  if (ctx == nullptr)
    return WinSize(-1,-1);

  if (screen_id >= ctx->_screens.size())
    return WinSize(-1,-1);

  return WinSize(ctx->_screens[screen_id]->width, ctx->_screens[screen_id]->height);
}

///*
// * \\fn Window* WindowManager::create_window
// *
// * created on: Nov 19, 2019
// * author: daniel
// *
// */
//Window* WindowManager::create_window(const char* display_name, const char* title,
//                                        Window* parent, size_t width, size_t height,
//                                        int x /*= RANDOM_POS*/, int y /*= RANDOM_POS*/)
//{
//  if (display_name == nullptr)
//    display_name = _default_display.c_str();
//
//  std::unordered_set<_Context*>::iterator iter =
//      std::find_if(_display_db.begin(), _display_db.end(),[display_name](_Context* db)
//  {
//    return db->_display_name == display_name;
//  });
//
//  if (iter == _display_db.end())
//  {
//    _Context* db = new _Context;
//    db->_display_name = (display_name != nullptr) ? display_name : "";
//    pipe(db->_pipe);
//    db->_terminate.store(false);
//
//    db->_thread  = std::thread([](WindowManager* m, _Context* d)
//    {
//      m->x_loop(d);
//    },this, db);
//
//    _mutex.lock();
//    iter  = _display_db.insert(db).first;
//    _mutex.unlock();
//  }
//
//  Window* new_wnd = new Window(title, x, y,width, height, parent);
//
//  _mutex.lock();
//  _wind_set.insert(new_wnd);
//  _mutex.unlock();
//
//  LCreateWindowEvent ce(new_wnd);
//  post_message((*iter), ce.serialize());
//
//  return new_wnd;
//}

/*
 * \\fn void WindowManager::post_message
 *
 * created on: Jul 1, 2019
 * author: daniel
 *
 */
void WindowManager::post_message(Context ctx, const bytestream& msg)
{
  if (ctx != nullptr)
  {
    std::lock_guard<std::mutex> l(_mutex);
    ::write(reinterpret_cast<_Context*>(ctx)->_pipe[1],msg.buf(),msg.size());
  }
}

/*
 * \\fn void WindowManager::x_loop
 *
 * created on: Jun 28, 2019
 * author: daniel
 *
 */
void WindowManager::x_loop(_Context* ctx)
{
  XSetErrorHandler(x_error_handler);
  Display* dsp = XOpenDisplay((ctx->_display_name.size() > 0) ? ctx->_display_name.c_str() : nullptr);

  if (dsp == nullptr)
  {
    std::cerr << "Error: cannot open display " << std::endl;
    return;
  }

  _mutex.lock();
  ctx->_display = dsp;
  ctx->_blackColor = BlackPixel(dsp, DefaultScreen(dsp));
  ctx->_whiteColor = WhitePixel(dsp, DefaultScreen(dsp));

  int number_of_screens = ScreenCount(dsp);
  for (int index = 0; index < number_of_screens; index++)
  {
    Screen* sc = ScreenOfDisplay(dsp, index);
    if (sc != nullptr)
      ctx->_screens.push_back(sc);
  }
  _mutex.unlock();

  int     x11_fd = ConnectionNumber(dsp);
  fd_set  in_fds;
  while(!ctx->_terminate)
  {
    FD_ZERO(&in_fds);
    FD_SET(ctx->_pipe[0],&in_fds);
    FD_SET(x11_fd,&in_fds);

    if (::select(std::max(ctx->_pipe[0],x11_fd) + 1, &in_fds, nullptr, nullptr, nullptr) > 0)
    {
      if (FD_ISSET(ctx->_pipe[0], &in_fds))
        l_event(ctx);

      if (FD_ISSET(x11_fd, &in_fds))
        x_event(ctx);
    }
  }

  XCloseDisplay(dsp);
}


/*
 * \\fn void WindowManager::x_event
 *
 * created on: Nov 19, 2019
 * author: daniel
 *
 */
void WindowManager::x_event(_Context* ctx)
{
  XEvent e;
  while(!ctx->_terminate && XPending(ctx->_display))
  {
    XNextEvent(ctx->_display, &e);
    if (e.type == DestroyNotify) // MapNotify)
      ctx->_terminate = true;

    _mutex.lock();
    window_set::iterator iter = std::find_if(_wind_set.begin(), _wind_set.end(), [e](Window* wnd)->bool
    {
      return (wnd->handle() ==  e.xany.window);
    });

    if (iter != _wind_set.end())
      (*iter)->x_event(ctx, e);

    _mutex.unlock();

  }
}

/*
 * \\fn void WindowManager::l_event
 *
 * created on: Jul 1, 2019
 * author: daniel
 *
 */
void WindowManager::l_event(_Context* ctx)
{
  uint32_t  type;
  if (::read(ctx->_pipe[0], &type, sizeof(type)) != sizeof(type))
    return;

  switch (type)
  {
  case eEventStop:
    ctx->_terminate.store(true);
    break;

  case eCreateWindowEvent:
    {
      LCreateWindowEvent ce(ctx->_pipe[0]);
      if (ce._wnd != nullptr)
        ce._wnd->l_event(ctx, &ce);
    }
    break;

  case eShowImageStructure:
    {
      LShowImageEvent se(ctx->_pipe[0]);
      if (se._wnd != nullptr)
        se._wnd->l_event(ctx, &se);
    }
    break;

  case eCloseWindowEvent:
    {
      LCloseWindowEvent se(ctx->_pipe[0]);
      if (se._wnd != nullptr)
        se._wnd->l_event(ctx, &se);
    }
    break;

  case eUpdateWindowEvent:
    {
      LUpdateWindowEvent ue(ctx->_pipe[0]);
      if (ue._wnd != nullptr)
        ue._wnd->l_event(ctx, &ue);
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