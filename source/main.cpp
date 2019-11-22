/**
 *
 *
 *
 *
 */
#include "Args.hpp"
#include "Utils.hpp"
#include "DeviceManager.hpp"
#include "Deserializer.hpp"
#include "Camera.hpp"
#include "WindowManager.hpp"
#include "Window.hpp"



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
  if (!args.parse(argc,(const char**)argv))
  {
    std::cout << "Not enough arguments" << std::endl << args.help();
    return 1;
  }

  wm::get()->init();

  brt::jupiter::Metadata meta_args = args.get_as_metadata();

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

  window::Window *wnd = nullptr;
  std::map<Camera*,uint16_t>  camera_pos_map;
  uint8_t row = 0, col = 0;

  for (auto device : devices)
  {
    std::cout << "Activating cameras" << device << std::endl;
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
//              wnd = window::CameraWindow::create("Video Streaming", nullptr,
//                  cam->format()->fmt.pix.width,
//                  cam->format()->fmt.pix.height);

              wnd = wm::get()->create_window("Video Streaming", devices.size() * 2,
                    cam->format()->fmt.pix.width,
                    cam->format()->fmt.pix.height);
            }
//            wnd->add_subwnd(cam);

            if (camera_pos_map.find(cam) == camera_pos_map.end())
            {
              wnd->create_subwnd(col, row, cam);
              camera_pos_map[cam] = (col << 8) | row;

              if (++col >= wnd->cols())
              {
                col = 0;
                row++;
              }
            }
          }
        }
      }
    }
  }

  char buffer[1024];
  std::string line;
  //window::Window *wnd = nullptr;

  //std::map<Camera*,uint16_t>  camera_pos_map;
  //uint8_t row = 0, col = 0;

  do
  {
    std::cout << "T.... :" << line << ":" << std::endl;
    std::cin.getline(buffer, sizeof(buffer));
    line = buffer;

//    if (Utils::stristr(line, "run") == 0)
//    {
//      line = line.substr(3);
//      int cam_id = strtoul(line.c_str(),nullptr,0);
//
//      Deserializer* des = DeviceManager::get()->get_device(cam_id >> 1);
//      if (des != nullptr)
//      {
//        Camera* cam = des->get_camera(cam_id & 1);
//        if (cam != nullptr)
//        {
//          if (cam->start_streaming())
//          {
//            if (wnd == nullptr)
//            {
//              wnd = wm::get()->create_window("Video Streaming", devices.size() * 2,
//                    cam->format()->fmt.pix.width,
//                    cam->format()->fmt.pix.height);
//            }
//
//            if (camera_pos_map.find(cam) == camera_pos_map.end())
//            {
//              wnd->create_subwnd(col, row, cam);
//              camera_pos_map[cam] = (col << 8) | row;
//
//              if (++col >= wnd->cols())
//              {
//                col = 0;
//                row++;
//              }
//            }
//          }
//        }
//      }
//    }
//
//    else if (Utils::stristr(line, "stop") == 0)
//    {
//      line = line.substr(4);
//      int cam_id = strtoul(line.c_str(),nullptr,0);
//
//      Deserializer* des = DeviceManager::get()->get_device(cam_id >> 1);
//      if (des != nullptr)
//      {
//        Camera* cam = des->get_camera(cam_id & 1);
//        if (cam != nullptr)
//        {
//          cam->stop_streaming();
//        }
//      }
//    }

  } while (line != "q");

  wm::get()->release();

  return 0;
}
