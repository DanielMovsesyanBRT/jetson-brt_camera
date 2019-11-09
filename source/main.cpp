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

  std::vector<std::string> cameras = meta_args.matching_keys("camera\\d");
  for (auto camera : cameras)
  {
    std::cout << "Loading device " << camera << std::endl;
    int id = strtol(camera.substr(6).c_str(),nullptr,0);
    uint8_t deserializer_id = static_cast<uint8_t>(id >> 1);
    uint8_t camera_id = static_cast<uint8_t>(id & 1);

    std::string script_file = meta_args.get<std::string>(camera.c_str(),"");

    auto device = brt::jupiter::CameraManager::get()->get_device(MAKE_CAM_ID(deserializer_id, camera_id));
    if (device != nullptr)
    {
      device->load_script(script_file.c_str());
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

    if (Utils::stristr(line, "activate") == 0)
    {
      line = line.substr(8);
      int cam_id = strtoul(line.c_str(),nullptr,0);

      Deserializer* des = CameraManager::get()->get_device(cam_id >> 1);
      if (des != nullptr)
      {
        Camera* cam = des->get_camera(cam_id & 1);
        if (cam != nullptr)
          cam->activate();
      }
    }

    else if (Utils::stristr(line, "release") == 0)
    {
      line = line.substr(7);
      int cam_id = strtoul(line.c_str(),nullptr,0);

      Deserializer* des = CameraManager::get()->get_device(cam_id >> 1);
      if (des != nullptr)
      {
        Camera* cam = des->get_camera(cam_id & 1);
        if (cam != nullptr)
          cam->activate(false);
      }
    }

    else if (Utils::stristr(line, "run") == 0)
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
              wnd = wm::get()->create_window("Video Streaming", 2,
                    cam->format()->fmt.pix.width,
                    cam->format()->fmt.pix.height);
            }

            wnd->create_subwnd(cam_id & 1, 0, cam);
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
