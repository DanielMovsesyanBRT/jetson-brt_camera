/**
 *
 *
 *
 *
 */
#include "Args.hpp"
#include "Utils.hpp"
#include "CameraManager.hpp"
#include "Deserializer.hpp"
#include "Camera.hpp"
#include "WindowManager.hpp"
#include "Window.hpp"



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
  if (!args.parse(argc,(const char**)argv))
  {
    std::cout << "Not enough arguments" << std::endl << args.help();
    return 1;
  }

  wm::get()->init();

  brt::jupiter::Metadata meta_args = args.get_as_metadata();

  std::vector<int> ids;

  std::vector<std::string> devices = meta_args.matching_keys("device\\d");
  for (auto device : devices)
  {
    std::cout << "Loading device " << device << std::endl;
    int id = strtol(device.substr(6).c_str(),nullptr,0);

    std::string script_file = meta_args.get<std::string>(device.c_str(),"");

    auto deserializer = brt::jupiter::CameraManager::get()->get_device(id);
    if (deserializer != nullptr)
    {
      deserializer->load_script(script_file.c_str());
      ids.push_back(id);
    }
  }

  char buffer[1024];
  std::string line;

//  double frame_rate = brt::jupiter::Utils::frame_rate(meta_args.get<std::string>("frame_rate","10fps").c_str());
//  std::cout << "Setting trigger to " << frame_rate << "fps" << std::endl;

//  brt::jupiter::Trigger* tr = new brt::jupiter::Trigger(frame_rate);

  window::Window *wnd = nullptr;


  do
  {
    std::cout << "T.... :" << line << ":" << std::endl;
    std::cin.getline(buffer, sizeof(buffer));
    line = buffer;

    if (Utils::stristr(line, "run") == 0)
    {
      line = line.substr(3);
      int cam_id = strtoul(line.c_str(),nullptr,0);

      Deserializer* des = CameraManager::get()->get_device(cam_id >> 1);
      if (des != nullptr)
      {
        Camera* cam = des->get_camera(cam_id & 1);
        if (cam != nullptr)
        {
          if (cam->start_streaming())
          {
            if (wnd == nullptr)
            {
              wnd = wm::get()->create_window("Video Streaming", devices.size() * 2,
                    cam->format()->fmt.pix.width,
                    cam->format()->fmt.pix.height);
            }

            wnd->create_subwnd(cam_id & 1, cam_id >> 1, cam);
          }
        }
      }
    }

    else if (Utils::stristr(line, "stop") == 0)
    {
      line = line.substr(4);
      int cam_id = strtoul(line.c_str(),nullptr,0);

      Deserializer* des = CameraManager::get()->get_device(cam_id >> 1);
      if (des != nullptr)
      {
        Camera* cam = des->get_camera(cam_id & 1);
        if (cam != nullptr)
        {
          cam->stop_streaming();
        }
      }
    }

  } while (line != "q");

  wm::get()->release();

  return 0;
}
