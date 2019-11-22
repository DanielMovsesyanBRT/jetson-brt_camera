/*
 * fltk_manager.cpp
 *
 *  Created on: Nov 21, 2019
 *      Author: daniel
 */

#include "fltk_manager.hpp"
#include "Utils.hpp"

#include <dirent.h>
#include <unistd.h>
#include <iostream>

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
  char* display_name = getenv("DISPLAY");
  if ((display_name == nullptr) || (strlen(display_name) == 0))
  {
    std::vector<std::string> displays = Utils::enumerate_displays();
    if (displays.size() == 0)
      return;

    if (displays.size() == 1)
      _default_display = displays[0];
    else
    {
      char buffer[1024];
      std::string line;
      int id;

      do
      {
        std::cout << "Please select default display " << std::endl;

        for (size_t index = 0; index < displays.size(); index++)
          std::cout << (index + 1) << ") " << displays[index] <<std::endl;

        std::cin >> id;
        if ((id < 1) || (id > displays.size()))
          std::cout << "Invalid entry" << std::endl;
      }
      while((id < 1) || (id > displays.size()));

      _default_display = displays[id - 1];
    }
  }

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
