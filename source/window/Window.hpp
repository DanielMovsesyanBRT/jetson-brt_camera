/*
 * Window.hpp
 *
 *  Created on: Jun 28, 2019
 *      Author: daniel
 */

#ifndef WINDOW_WINDOW_HPP_
#define WINDOW_WINDOW_HPP_

#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glx.h>

#include <string>
#include <mutex>
#include <atomic>

#include "WinStructures.hpp"
#include "Image.hpp"

namespace brt
{
namespace jupiter
{
namespace window
{

/*
 * \\class Window
 *
 * created on: Jun 28, 2019
 *
 */
class Window : public image::ImageConsumer
{
public:
  Window(const char* title, int x, int y, size_t cols, size_t rows, size_t width, size_t height, ::Window root);
  virtual ~Window();

          void                    processEvent(const XEvent&);
          void                    processEvent(const X_Event*);

          ::Window                handle() const { return _hndl; }

          size_t                  create_subwnd(int col,int row, image::ImageProducer* ip);
  virtual void                    consume(image::ImageBox);
          size_t                  rows() const { return _rows; }
          size_t                  cols() const { return _cols; }

private:
          void                    create_colormap();
          void                    x_create();
          void                    x_show(X_ShowImageEvent*);

private:
  ::Window                        _hndl;
  ::Window                        _root;
  std::string                     _title;
  int                             _x, _y;
  size_t                          _video_width, _video_height;
  size_t                          _actual_width, _actual_height;
  size_t                          _rows, _cols;
  GLXContext                      _glc;
  GLuint                          _texture;

  struct GLWindow
  {
    GLfloat                       _left;
    GLfloat                       _top;
    GLfloat                       _right;
    GLfloat                       _bottom;
    GLuint                        _tex;
    size_t                        _col;
    size_t                        _row;
    image::RawRGBPtr              _image;
  };

  std::vector<GLWindow>           _gl_map;
  std::mutex                      _mutex;

  std::atomic_int_fast32_t        _click;
  uint32_t                        _global_number;
};

} /* namespace window */
} /* namespace jupiter */
} /* namespace brt */

#endif /* WINDOW_WINDOW_HPP_ */
