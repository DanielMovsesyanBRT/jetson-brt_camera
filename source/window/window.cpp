/*
 * window.cpp
 *
 *  Created on: Nov 19, 2019
 *      Author: daniel
 */

#include "window.hpp"
#include "window_manager.hpp"

namespace brt
{
namespace jupiter
{
namespace window
{

/*
 * \\fn Constructor Window::Window
 *
 * created on: Nov 19, 2019
 * author: daniel
 *
 */
Window::Window(const char* title, int x, int y,
    size_t width, size_t height, Window* parent)
: _handle(0)
, _parent(parent)
, _context(nullptr)
, _title(title)
, _x(x)
, _y(y)
, _width(width)
, _height(height)
, _visual(nullptr)
, _depth(-1)
, _class(InputOutput)
, _value_mask(CWBackPixel | CWColormap | CWBorderPixel | CWEventMask)
, _swa()
, _border_width(2)
{
  wm::get()->_mutex.lock();
  wm::get()->_wind_set.insert(this);
  wm::get()->_mutex.unlock();
}

/*
 * \\fn Destructor Window::~Window
 *
 * created on: Nov 19, 2019
 * author: daniel
 *
 */
Window::~Window()
{
  // TODO Auto-generated destructor stub
}

/*
 * \\fn Window* Window::create
 *
 * created on: Nov 20, 2019
 * author: daniel
 *
 */
Window* Window::create(const char* title, Window* parent, size_t width, size_t height,
                                        int x /* = RANDOM_POS*/, int y /* = RANDOM_POS*/)
{
  Window* wnd = new Window(title, x, y, width, height, parent);
  return wnd;
}

/*
 * \\fn void Window::show
 *
 * created on: Nov 21, 2019
 * author: daniel
 *
 */
void Window::show(const char* display_name)
{
  Context ctx = wm::get()->get_context(display_name, true);
  if (ctx == nullptr)
    return;

  LCreateWindowEvent ce(this);
  wm::get()->post_message(ctx, ce.serialize());
}

/*
 * \\fn bool Window::x_event
 *
 * created on: Nov 19, 2019
 * author: daniel
 *
 */
bool Window::x_event(Context context,const XEvent &event)
{
  switch (event.type)
  {
  case Expose:
    on_draw(context,DefaultGC(display(), DefaultScreen(display())));
    return true;

  default:
    break;
  }
  return false;
}

/*
 * \\fn bool Window::l_event
 *
 * created on: Nov 19, 2019
 * author: daniel
 *
 */
bool Window::l_event(Context context,const LEvent *event)
{
  switch (event->_type)
  {
  case eCreateWindowEvent:
    x_create(context);
    return true;

  case eCloseWindowEvent:
    X11Library::get().call<int,::Display*,::Window>("XDestroyWindow",display(), _handle);
    // XDestroyWindow(display(), _handle);
    return true;

  case eUpdateWindowEvent:
    on_draw(context,DefaultGC(display(), DefaultScreen(display())));
    return true;

  default:
    break;
  }
  return false;
}

/*
 * \\fn Display* Window::display
 *
 * created on: Nov 20, 2019
 * author: daniel
 *
 */
Display* Window::display(Context ctx /*= nullptr*/) const
{
  if (ctx == nullptr)
    ctx = _context;
  return wm::get()->display(ctx);
}

/*
 * \\fn int Window::screen
 *
 * created on: Nov 20, 2019
 * author: daniel
 *
 */
int Window::screen(Context ctx /*= nullptr*/) const
{
  if (ctx == nullptr)
    ctx = _context;
  return wm::get()->default_screen(ctx);
}

/*
 * \\fn void Window::close
 *
 * created on: Nov 19, 2019
 * author: daniel
 *
 */
void Window::close()
{
  LCloseWindowEvent cw(this);
  wm::get()->post_message(_context,cw.serialize());
}

/*
 * \\fn void Window::update
 *
 * created on: Nov 20, 2019
 * author: daniel
 *
 */
void Window::update()
{
  LUpdateWindowEvent ue(this);
  wm::get()->post_message(_context,ue.serialize());
}

/*
 * \\fn WinRect Window::get_window_rect
 *
 * created on: Nov 20, 2019
 * author: daniel
 *
 */
WinRect Window::get_window_rect() const
{
  XWindowAttributes attr;
  X11Library::get().call<Status,::Display*,::Window,XWindowAttributes*>
      ("XGetWindowAttributes", display(), _handle, &attr);
  // XGetWindowAttributes(display(), _handle, &attr);

  return WinRect(attr.x,attr.y,attr.width,attr.height);
}

/*
 * \\fn void Window::x_create
 *
 * created on: Nov 19, 2019
 * author: daniel
 *
 */
void Window::x_create(Context context)
{
  if (_handle != 0)
    return;

  _context = context;
  Display* dspl = display();
  int screenId = DefaultScreen(dspl);

  WinSize screen_size = wm::get()->resolution(context, screenId);
  if ((_x == RANDOM_POS) || (_y == RANDOM_POS))
  {
    WinSize randomWindow;
    randomWindow._width = screen_size._width - _width;
    randomWindow._height = screen_size._height - _height;

    _x = (_x == RANDOM_POS) ? (rand() % randomWindow._width) : _x;
    _y = (_y == RANDOM_POS) ? (rand() % randomWindow._height) : _y;
  }
  else
  {
    _x = (_x == RANDOM_POS)?0:_x;
    _y = (_y == RANDOM_POS)?0:_y;
  }

  _visual = DefaultVisual(dspl, screenId);
  _depth  = DefaultDepth(dspl, screenId);

  _swa.colormap = X11Library::get().call<Colormap,::Display*,::Window,Visual*,int>
            ("XCreateColormap", dspl, RootWindow(dspl, screenId), _visual, AllocNone);
  // _swa.colormap = XCreateColormap(dspl, RootWindow(dspl, screenId), _visual, AllocNone);

  _swa.override_redirect = True;
  _swa.background_pixel = wm::get()->white_color(context);
  _swa.border_pixel = wm::get()->black_color(context);
  _swa.event_mask = ExposureMask | KeyPressMask;

  //::Window  parent = (_parent != nullptr) ? _parent->handle() : XRootWindow(dspl, screenId);
  ::Window  parent = (_parent != nullptr) ? _parent->handle() :
          X11Library::get().call<::Window,::Display*,int>("XRootWindow", dspl, screenId);

  pre_create_window(context);

  _handle = X11Library::get().call<::Window,::Display*,::Window,int,
                  int,unsigned int,unsigned int,unsigned int,int,
                  unsigned int,Visual*,unsigned long,XSetWindowAttributes*>
            ("XCreateWindow", dspl, parent,
                _x, _y, _width, _height, _border_width, _depth,
                InputOutput, _visual, _value_mask, &_swa);

//  _handle = XCreateWindow(dspl, parent,
//      _x, _y, _width, _height, _border_width, _depth,
//      InputOutput, _visual, _value_mask, &_swa);

  if (_handle != 0)
  {
    on_create_window(context);

    X11Library::get().call<int,::Display*,::Window>("XClearWindow", dspl, _handle);
    // XClearWindow(dspl, _handle);
    X11Library::get().call<int,::Display*,::Window,_Xconst char*>("XStoreName",dspl, _handle, _title.c_str());
    // XStoreName(dspl, _handle, _title.c_str());

    X11Library::get().call<int,::Display*,::Window>("XMapWindow", dspl, _handle);
    //XMapWindow(dspl, _handle);
    X11Library::get().call<int,::Display*,::Window,int,int>("XMoveWindow", dspl, _handle, _x, _y);
    //XMoveWindow(dspl, _handle, _x, _y);

    X11Library::get().call<int,Display*>("XFlush",dspl);
    //XFlush(dspl);
  }
}

/*
 * \\fn void Window::x_close
 *
 * created on: Nov 19, 2019
 * author: daniel
 *
 */
void Window::x_close(Context context)
{
  if (_handle == 0)
    return;

  on_close(context);
  X11Library::get().call<int,::Display*,::Window>("XDestroyWindow",display(), _handle);
  // XDestroyWindow(display(), _handle);

  delete this;
}

} /* namespace window */
} /* namespace jupiter */
} /* namespace brt */
