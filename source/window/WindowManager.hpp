/*
 * WindowManager.hpp
 *
 *  Created on: Jun 28, 2019
 *      Author: daniel
 */

#ifndef WINDOW_WINDOWMANAGER_HPP_
#define WINDOW_WINDOWMANAGER_HPP_

#include <X11/X.h>
#include <X11/Xlib.h>
#include <climits>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>

#include <vector>
#include <unordered_set>
#include <thread>
#include <condition_variable>
#include <mutex>

#include "WinStructures.hpp"

#define RANDOM_POS                          (INT_MAX)


namespace brt
{
namespace jupiter
{
namespace window
{

class Window;

/*
 * \\class WindowManager
 *
 * created on: Jun 28, 2019
 *
 */
class WindowManager
{
  WindowManager();
  virtual ~WindowManager();
public:
  static  WindowManager*          get() { return &_object; }
          void                    init();
          void                    release();

          Display*                display() const { return _display; }
          int                     default_screen() const;

          size_t                  num_screens() const { return _screens.size(); }
          WinSize                 resolution(size_t screen_id) const;

          Window*                 create_window(const char* title, size_t num_views, size_t width, size_t height,int x = RANDOM_POS, int y = RANDOM_POS);
          void                    post_message(const bytestream& msg);

          int                     black_color() const { return  _blackColor; }
          int                     white_color() const { return _whiteColor; }

private:
          void                    x_loop();
   static int                     x_error_handler(Display*, XErrorEvent*);

          void                    process_local_event();

private:
  static WindowManager            _object;
  std::string                     _default_display;

  Display*                        _display;

  int                             _blackColor;
  int                             _whiteColor;

  std::vector<Screen*>            _screens;
  typedef std::unordered_set<Window*> window_set;
  window_set                      _wind_set;

  std::thread                     _thread;
  std::condition_variable         _cv;
  std::mutex                      _mutex;
  int                             _pipe[2];
  bool                            _terminate;
};

} /* namespace window */
} /* namespace jupiter */
} /* namespace brt */

using wm = brt::jupiter::window::WindowManager;

#endif /* WINDOW_WINDOWMANAGER_HPP_ */
