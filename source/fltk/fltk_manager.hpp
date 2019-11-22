/*
 * fltk_manager.hpp
 *
 *  Created on: Nov 21, 2019
 *      Author: daniel
 */

#ifndef SOURCE_FLTK_FLTK_MANAGER_HPP_
#define SOURCE_FLTK_FLTK_MANAGER_HPP_

#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Box.H>

#include <GL/gl.h>
#include <GL/glx.h>


#include <vector>
#include <string>

namespace brt
{
namespace jupiter
{
namespace fltk
{

/*
 * \\class FLTKManager
 *
 * created on: Nov 21, 2019
 *
 */
class FLTKManager
{
  FLTKManager();
  virtual ~FLTKManager();

public:
  static  FLTKManager*            get() { return &_object; }
          void                    init();


private:
  static FLTKManager              _object;
  std::string                     _default_display;
};

} /* namespace fltk */
} /* namespace jupiter */
} /* namespace brt */


using fm = brt::jupiter::fltk::FLTKManager;
#endif /* SOURCE_FLTK_FLTK_MANAGER_HPP_ */
