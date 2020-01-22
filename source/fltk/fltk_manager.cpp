/*
 * fltk_manager.cpp
 *
 *  Created on: Nov 21, 2019
 *      Author: daniel
 */

#include "fltk_manager.hpp"
#include "camera_menu.hpp"

#include <dirent.h>
#include <unistd.h>
#include <iostream>
#include <utils.hpp>
#include <stdlib.h>
#include <thread>

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
  _default_display = Utils::aquire_display("controls");
  fl_display_name = _default_display._name.c_str();
  setenv("DISPLAY",fl_display_name,1);
}

/*
 * \\fn FLTKManager::run
 *
 * created on: Jan 16, 2020
 * author: daniel
 *
 */
void FLTKManager::run()
{
  CameraWindow* cm = new CameraWindow;
  cm->make_window()->show();
  Fl::run();
  delete cm;
}

} /* namespace fltk */
} /* namespace jupiter */
} /* namespace brt */
