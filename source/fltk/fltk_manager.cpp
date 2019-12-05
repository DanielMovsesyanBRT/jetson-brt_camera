/*
 * fltk_manager.cpp
 *
 *  Created on: Nov 21, 2019
 *      Author: daniel
 */

#include "fltk_manager.hpp"
#include <dirent.h>
#include <unistd.h>
#include <iostream>
#include "../utils.hpp"

#define _PATH_PROCNET_X11                   "/tmp/.X11-unix"
#define _PATH_PROCNET_TCP                   "/proc/net/tcp"
#define _PATH_PROCNET_TCP6                  "/proc/net/tcp6"
#define X11_PORT_MIN                        (6000)
#define X11_PORT_MAX                        (6100)

extern const char *fl_display_name;

namespace brt
{
namespace jupiter
{
namespace fltk
{

FLTKManager FLTKManager::_object;

/*
 * \\fn Constructor FLTKManager::FLTKManager
 *
 * created on: Nov 21, 2019
 * author: daniel
 *
 */
FLTKManager::FLTKManager()
{

}

/*
 * \\fn Destructor FLTKManager::~FLTKManager
 *
 * created on: Nov 21, 2019
 * author: daniel
 *
 */
FLTKManager::~FLTKManager()
{

}

/*
 * \\fn FLTKManager::init
 *
 * created on: Nov 21, 2019
 * author: daniel
 *
 */
void FLTKManager::init()
{
  // Check for default display
  _default_display = Utils::aquire_display();
  fl_display_name = _default_display.c_str();

  Fl_Window *window = new Fl_Window(340,180);
  Fl_Box *box = new Fl_Box(20,40,300,100,"Hello, World!");
  box->box(FL_UP_BOX);
  box->labelfont(FL_BOLD+FL_ITALIC);
  box->labelsize(36);
  box->labeltype(FL_SHADOW_LABEL);
  window->end();
  window->show();
  Fl::run();


}



} /* namespace fltk */
} /* namespace jupiter */
} /* namespace brt */
