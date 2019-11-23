/**
 *
 *
 *
 *
 */
//#include "Args.hpp"
#include "Utils.hpp"
#include "DeviceManager.hpp"
#include "Deserializer.hpp"
#include "Camera.hpp"

//#include "WindowManager.hpp"
//#include "Window.hpp"

#include "window_manager.hpp"
#include "camera_window.hpp"

// #include "fltk_manager.hpp"


#include <map>
#include <iostream>
#include <string>

#include <stdint.h>


using namespace brt::jupiter;


/*
 * \\fn int main
 *
 * created on: Nov 5, 2019
 * author: daniel
 *
 */
int main(int argc, char **argv)
{
  wm::get()->init();

  brt::jupiter::Metadata meta_args;
  meta_args.parse(argc,argv);

  double frame_rate = brt::jupiter::Utils::frame_rate(meta_args.get<std::string>("frame_rate","10fps").c_str());
  std::cout << "Setting trigger to " << frame_rate << "fps" << std::endl;
  brt_camera_trigger trg;

  trg._pwm_period = (int)(frame_rate * 1000);
  trg._duty_period = 5; //ms

  if (::ioctl(brt::jupiter::DeviceManager::get()->handle(),BRT_CAMERA_TRIGGER_SET_PWM,(unsigned long)&trg))
    std::cerr << "Unable to set trigger error:" << errno << std::endl;


  std::vector<int> ids;
  std::vector<std::string> devices = meta_args.matching_keys("device\\d");
  for (auto device : devices)
  {
    std::cout << "Loading device " << device << std::endl;
    int id = strtol(device.substr(6).c_str(),nullptr,0);

    std::string script_file = meta_args.get<std::string>(device.c_str(),"");

    auto deserializer = brt::jupiter::DeviceManager::get()->get_device(id);
    if (deserializer != nullptr)
    {
      deserializer->load_script(script_file.c_str());
      ids.push_back(id);
    }
  }

  // Check Displays

  window::CameraWindow *wnd = nullptr;

  for (auto device : devices)
  {
    std::cout << "Activating cameras " << device << std::endl;
    int id = strtol(device.substr(6).c_str(),nullptr,0);

    Deserializer* des = DeviceManager::get()->get_device(id);
    if (des != nullptr)
    {
      for (size_t index = 0; index < 2; index++)
      {
        Camera* cam = des->get_camera(index);
        if (cam != nullptr)
        {
          if (cam->start_streaming())
          {
            if (wnd == nullptr)
            {
              wnd = window::CameraWindow::create("Video Streaming", nullptr,
                  cam->format()->fmt.pix.width,
                  cam->format()->fmt.pix.height);
            }
            wnd->add_subwnd(cam);
          }
        }
      }
    }
  }

  if (wnd != nullptr)
    wnd->show(nullptr);
    
    
  char buffer[1024];
  std::string line;
  do
  {
    std::cout << "T.... :" << line << ":" << std::endl;
    std::cin.getline(buffer, sizeof(buffer));
    line = buffer;

  } while (line != "q");

  wm::get()->release();

//  fm::get()->init();
  return 0;
}
