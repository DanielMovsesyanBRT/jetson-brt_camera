/*
 * fltk_inteface_exports.hpp
 *
 *  Created on: Mar 16, 2020
 *      Author: daniel
 */

#ifndef FLTK_INTERFACE_FLTK_INTERFACE_EXPORTS_HPP_
#define FLTK_INTERFACE_FLTK_INTERFACE_EXPORTS_HPP_

namespace brt
{
namespace jupiter
{
namespace fltk
{

/*
 * \\struct CallbackInterface
 *
 * created on: Mar 16, 2020
 *
 */
struct CallbackInterface
{
  virtual ~CallbackInterface() {}

  virtual void                    run(int) = 0;
  virtual void                    dir(int,const char*) = 0;
  virtual const char*             destination(int) = 0;
};

} /* namespace fltk */
} /* namespace jupiter */
} /* namespace brt */

extern "C" void fltk_initialize();
extern "C" void fltk_interface_run(brt::jupiter::fltk::CallbackInterface* ci);

#endif /* FLTK_INTERFACE_FLTK_INTERFACE_EXPORTS_HPP_ */
