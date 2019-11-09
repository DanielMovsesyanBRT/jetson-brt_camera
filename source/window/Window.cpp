/*
 * Window.cpp
 *
 *  Created on: Jun 28, 2019
 *      Author: daniel
 */

#include "Window.hpp"
#include "WindowManager.hpp"
#include "Utils.hpp"
#include "Image.hpp"

//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc.hpp>

#include "cuda_debayering.h"

#include <iostream>
#include <fstream>


namespace brt
{
namespace jupiter
{
namespace window
{

/*
 * \\fn Constructor Window::Window
 *
 * created on: Jun 28, 2019
 * author: daniel
 *
 */
Window::Window(const char* title, int x, int y, size_t cols, size_t rows, size_t width, size_t height,::Window root)
: _hndl(0)
, _root(root)
, _title(title)
, _x(x)
, _y(y)
, _video_width(width)
, _video_height(height)
, _actual_width(width)
, _actual_height(height)
, _cols(cols)
, _rows(rows)
, _glc(0)
, _texture(0)
, _click(-1)
, _global_number(0)
{
}

/*
 * \\fn Destructor Window::~Window
 *
 * created on: Jun 28, 2019
 * author: daniel
 *
 */
Window::~Window()
{
}

/*
 * \\fn void Window::processEvent
 *
 * created on: Jul 1, 2019
 * author: daniel
 *
 */
void Window::processEvent(const X_Event* evt)
{
  switch(evt->_type)
  {
  case eCreateWindowEvent:
    x_create();
    break;

  case eShowImageStructure:
    x_show((X_ShowImageEvent*)evt);
    break;

  default:
    break;
  }
}


/*
 * \\fn void Window::processEvent
 *
 * created on: Jun 28, 2019
 * author: daniel
 *
 */
void Window::processEvent(const XEvent& evt)
{
  switch (evt.type)
  {
  case Expose:
    {
      XWindowAttributes attribs;
      XGetWindowAttributes(wm::get()->display(), _hndl, &attribs);
      glViewport(0, 0, attribs.width, attribs.height);

      glXSwapBuffers(wm::get()->display(), _hndl);
    }
    break;

  case ButtonPress:
    {
      int col = evt.xbutton.x / _actual_width;
      int row = evt.xbutton.y / _actual_height;

      size_t id = (size_t)-1;

      _mutex.lock();
      for (size_t index = 0; index < _gl_map.size(); index++)
      {
        if ((_gl_map[index]._col == col) && (_gl_map[index]._row == row))
        {
          id = index;
          break;
        }
      }
      _mutex.unlock();

      if (id < 4)
      {
        int_fast32_t expected = -1;
        _click.compare_exchange_strong(expected, id);
      }
      std::cout << "Button Press x=" << col << "; y=" << row << "; id=" << id << std::endl;
    }
    break;

  default:break;
  }
}

/*
 * \\fn unsigned int Window::create_subwnd
 *
 * created on: Jul 1, 2019
 * author: daniel
 *
 */
size_t Window::create_subwnd(int col,int row, image::ImageProducer* ip)
{
  if (col >= _cols)
    return (unsigned int)-1;

  if (row >= _rows)
    return (unsigned int)-1;

  float fcol_size = 2.0 / _cols;
  float frow_size = 2.0 / _rows;

  GLWindow  wnd;
  wnd._left = -1.0 + col * fcol_size;
  wnd._top  = -1.0 + row * frow_size;
  wnd._right = -1.0 + (col + 1) * fcol_size;
  wnd._bottom = -1.0 + (row + 1) * frow_size;
  wnd._col = col;
  wnd._row = row;

  _mutex.lock();
  _gl_map.push_back(wnd);
  size_t result = _gl_map.size() - 1;
  _mutex.unlock();

  if (ip != nullptr)
    ip->register_consumer(this,Metadata().set("col",col).set("row",row).set("id",result));

  return result;
}

/*
 * \\fn void Window::consume
 *
 * created on: Jul 2, 2019
 * author: daniel
 *
 */
void Window::consume(image::ImageBox box)
{
  if (box.empty())
    return;

  int id = box[0]->get("id",-1);
  if ((id == -1) || box.empty())
    return;

  std::unique_lock<std::mutex> l(_mutex);
  if (id >= _gl_map.size())
    return;

  if (_gl_map[id]._image)
    return;

  _gl_map[id]._image = box[0]->get_bits();
  wm::get()->post_message(X_ShowImageEvent(id, this).serialize());
}

/*
 * \\fn void Window::x_create
 *
 * created on: Jul 1, 2019
 * author: daniel
 *
 */
void Window::x_create()
{
  if (_hndl != 0)
    return;

  Display* display = wm::get()->display();
  int screenId = wm::get()->default_screen();

  WinSize screen_size = wm::get()->resolution(screenId);
  int full_width = _actual_width * _cols;
  int full_height = _actual_height * _rows;

#define WIDTH_GAP             (150)
#define HEIGHT_GAP            (50)

  if (full_width > (screen_size._width - WIDTH_GAP))
  {
    float ratio = (float)full_width / (screen_size._width - WIDTH_GAP);
    _actual_width = (size_t)((float)_actual_width / ratio);
    _actual_height = (size_t)((float)_actual_height / ratio);

    full_width = _actual_width * _cols;
    full_height = _actual_height * _rows;
  }

  if (full_height > (screen_size._height - HEIGHT_GAP))
  {
    float ratio = (float)full_height / (screen_size._height - HEIGHT_GAP);
    _actual_width = (size_t)((float)_actual_width / ratio);
    _actual_height = (size_t)((float)_actual_height / ratio);

    full_width = _actual_width * _cols;
    full_height = _actual_height * _rows;
  }

  if ((_x == RANDOM_POS) || (_y == RANDOM_POS))
  {
    WinSize randomWindow;
    randomWindow._width = screen_size._width - full_width;
    randomWindow._height = screen_size._height - full_height;

    _x = (_x == RANDOM_POS) ? (rand() % randomWindow._width) : _x;
    _y = (_y == RANDOM_POS) ? (rand() % randomWindow._height) : _y;

  }
  else
  {
    _x = (_x == RANDOM_POS)?0:_x;
    _y = (_y == RANDOM_POS)?0:_y;
  }

  GLint att[] =
  {
    GLX_RGBA,
    GLX_DOUBLEBUFFER,
    GLX_DEPTH_SIZE, 24,
    GLX_STENCIL_SIZE, 8,
    GLX_RED_SIZE, 8,
    GLX_GREEN_SIZE, 8,
    GLX_BLUE_SIZE, 8,
    GLX_SAMPLE_BUFFERS, 0,
    GLX_SAMPLES, 0,
    None
  };

  XVisualInfo* vi = glXChooseVisual(display,screenId, att);

  if (vi == nullptr)
    return;
  XSetWindowAttributes  swa;
  swa.colormap = XCreateColormap(display, RootWindow(display, screenId), vi->visual, AllocNone);
  swa.override_redirect = True;
  swa.background_pixel = wm::get()->white_color();
  swa.border_pixel = wm::get()->black_color();
  swa.event_mask = ExposureMask | KeyPressMask;

  _hndl = XCreateWindow(wm::get()->display(), _root,
      _x, _y, full_width, full_height, 0, vi->depth, InputOutput, vi->visual,
      CWBackPixel | CWColormap | CWBorderPixel | CWEventMask, &swa);


  _glc = glXCreateContext(wm::get()->display(), vi, NULL, GL_TRUE);
  glXMakeCurrent(wm::get()->display(), _hndl, _glc);


  XClearWindow(wm::get()->display(), _hndl);
  XMapRaised(wm::get()->display(), _hndl);
  XStoreName(wm::get()->display(), _hndl, _title.c_str());

  XMoveWindow(display, _hndl, _x, _y);

  // Set GL Sample stuff
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

  glGenTextures(1, &_texture);

  XFlush(display);
  XSelectInput(display, _hndl, ButtonPressMask);
  //glEnable(GL_DEPTH_TEST);
}

/*
 * \\fn void Window::x_show
 *
 * created on: Jul 1, 2019
 * author: daniel
 *
 */
void Window::x_show(X_ShowImageEvent* evt)
{
  _mutex.lock();
  GLWindow  wnd = _gl_map[evt->_id];
  _mutex.unlock();

  if (!wnd._image)
    return;

  if (_click.load() == evt->_id)
  {
    std::string file_name = Utils::string_format("image_%04d_%04d.raw", evt->_id, _global_number++);
    uint32_t w = wnd._image->width(), h = wnd._image->height(), bytes = 2 /* RAW12*/;
    std::ofstream raw_file (file_name, std::ios::out | std::ios::binary);
    raw_file.write(reinterpret_cast<const char*>(&w), sizeof(w));
    raw_file.write(reinterpret_cast<const char*>(&h), sizeof(h));
    raw_file.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));

    raw_file.write(reinterpret_cast<const char*>(wnd._image->bytes()),w * h * bytes);

    _click.store(-1);
  }

//////#if (defined ARCH) && (defined ARM) && (ARCH==ARM)
//  cv::Mat mat16uc1_bayer(wnd._image->height(), wnd._image->width(), CV_16UC1, wnd._image->bytes());
//
//  // Decode the Bayer data to RGB but keep using 16 bits per channel
//  cv::Mat mat16uc4_rgb(wnd._image->height(), wnd._image->width(), CV_16UC3);
//  cv::cvtColor(mat16uc1_bayer, mat16uc4_rgb,cv::COLOR_BayerGR2BGR);
//
//  // Convert the 16-bit per channel RGB image to 8-bit per channel
//  cv::Mat mat8uc4_rgb(wnd._image->height(), wnd._image->width(), CV_8UC3);
//  mat16uc4_rgb.convertTo(mat8uc4_rgb, CV_8UC3, 1.0/256);


  const size_t outputImgWidth = wnd._image->width();
  const size_t outputImgHeight = wnd._image->height();
  uint8_t *outputImgBuffer = (uint8_t*)malloc(outputImgWidth * outputImgHeight * 3 * sizeof(uint8_t));
  memset(outputImgBuffer, 128, outputImgWidth * outputImgHeight * 3 * sizeof(uint8_t));

  cuda::debayerUsingBilinearInterpolation((uint16_t*)wnd._image->bytes(), outputImgBuffer, false, false);

  glBindTexture(GL_TEXTURE_2D, _texture);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,  wnd._image->width(), wnd._image->height(), 0, GL_RGB, GL_UNSIGNED_BYTE, mat8uc4_rgb.data);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,  wnd._image->width(), wnd._image->height(), 0, GL_RGB, GL_UNSIGNED_BYTE, outputImgBuffer);

  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, _texture);
  glEnable(GL_TEXTURE_2D);
  glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex3f(wnd._left, wnd._top, 0.0);
    glTexCoord2f(0.0, 1.0); glVertex3f(wnd._left, wnd._bottom, 0.0);
    glTexCoord2f(1.0, 1.0); glVertex3f(wnd._right, wnd._bottom, 0.0);
    glTexCoord2f(1.0, 0.0); glVertex3f(wnd._right, wnd._top, 0.0);
  glEnd();

  glXSwapBuffers(wm::get()->display(), _hndl);
  glDisable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, 0);

  ::free(outputImgBuffer);

//#else
//  cv::Mat mat16uc1_bayer(wnd._image->height(), wnd._image->width(), CV_16UC1, wnd._image->bytes());
//
//  // Decode the Bayer data to RGB but keep using 16 bits per channel
//  cv::Mat mat16uc4_rgb(wnd._image->height(), wnd._image->width(), CV_16UC4);
//  cv::cvtColor(mat16uc1_bayer, mat16uc4_rgb,cv::COLOR_BayerGR2BGRA);
//
//  // Convert the 16-bit per channel RGB image to 8-bit per channel
//  cv::Mat mat8uc4_rgb(wnd._image->height(), wnd._image->width(), CV_8UC4);
//  mat16uc4_rgb.convertTo(mat8uc4_rgb, CV_8UC4, 1.0/256);
//  //cv::flip(mat8uc4_rgb,mat8uc4_rgb,-1);
//
//  Display* display = wm::get()->display();
//  int screenId = wm::get()->default_screen();
//  XVisualInfo visual_template;
//  int nxvisuals = 0;
//  visual_template.screen = wm::get()->default_screen();
//  XVisualInfo* vi = XGetVisualInfo (display, VisualScreenMask, &visual_template, &nxvisuals);
//  if (vi != nullptr)
//  {
//    XImage* ximage = XCreateImage(display, vi->visual, 24, ZPixmap, 0,reinterpret_cast<char*>(mat8uc4_rgb.data),
//                wnd._image->width(), wnd._image->height(), 32, 0);
//
//
//
//    XPutImage(display, _hndl, DefaultGC(display, screenId), ximage, 0, 0, wnd._col * _actual_width, wnd._row * _actual_height,
//        _actual_width, _actual_height);
//
//    //XDestroyImage(ximage);
//  }
//#endif

  _mutex.lock();
  _gl_map[evt->_id]._image.reset();
  _mutex.unlock();
}


} /* namespace window */
} /* namespace jupiter */
} /* namespace brt */
