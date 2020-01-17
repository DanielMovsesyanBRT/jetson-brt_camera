/**
 *
 *
 *
 *
 */
#include "isp.hpp"
#include "isp_manager.hpp"

#include "window_manager.hpp"
#include "camera_window.hpp"

#include "fltk_manager.hpp"
#include "camera_menu.hpp"


#include <map>
#include <iostream>
#include <fstream>
#include <string>

#include <stdint.h>
#include "device/camera.hpp"
#include "device/deserializer.hpp"
#include "device/device_manager.hpp"
#include "utils.hpp"


using namespace brt::jupiter;


/*
 * \\class Consumer
 *
 * created on: Jan 16, 2020
 *
 */
class Consumer : public image::ImageConsumer
{
public:
  Consumer(const char* prefix) : _prefix(prefix), _directory(), _flag(false), _unique(0) {}

  Consumer(const Consumer& cons)
  : _prefix(cons._prefix)
  , _directory(cons._directory)
  , _flag(cons._flag.load())
  , _unique(cons._unique)
  {}

  virtual void                    consume(image::ImageBox box)
  {
    if (box.empty())
      return;

    int id = box[0]->get("<id>", -1);
    if ((id == -1) || (id > 1))
      return;

    _images[id] = box[0];
    if (_images[0] && _images[1])
    {
      bool expected = true;
      if (_flag.compare_exchange_strong(expected, false))
      {
        std::string file_name = Utils::string_format("%s/%s_%04d_left.raw", _directory.c_str(),_prefix.c_str(),_unique);
        image::RawRGBPtr bits = _images[0]->get_bits();
        uint32_t w = bits->width(), h = bits->height(), bytes = 2 /* RAW12*/;

        std::ofstream raw_file (file_name, std::ios::out | std::ios::binary);
        raw_file.write(reinterpret_cast<const char*>(&w), sizeof(w));
        raw_file.write(reinterpret_cast<const char*>(&h), sizeof(h));
        raw_file.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
        raw_file.write(reinterpret_cast<const char*>(bits->bytes()),w * h * bytes);

        file_name = Utils::string_format("%s/%s_%04d_right.raw", _directory.c_str(),_prefix.c_str(),_unique);

        bits = _images[1]->get_bits();
        w = bits->width(), h = bits->height(), bytes = 2 /* RAW12*/;

        raw_file = std::ofstream(file_name, std::ios::out | std::ios::binary);
        raw_file.write(reinterpret_cast<const char*>(&w), sizeof(w));
        raw_file.write(reinterpret_cast<const char*>(&h), sizeof(h));
        raw_file.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));

        raw_file.write(reinterpret_cast<const char*>(bits->bytes()),w * h * bytes);

        _unique++;
      }

      _images[0].reset();
      _images[1].reset();
    }
  }

  void                            set_destination(const char* dir)
  {
    _directory = dir;
  }

  const char*                     get_destination() const
  {
    return _directory.c_str();
  }

  void                            trigger()
  {
    bool expected = false;
    _flag.compare_exchange_strong(expected, true);
  }

private:
  std::string                     _prefix;
  std::string                     _directory;

  std::atomic_bool                _flag;
  uint32_t                        _unique;

  image::ImagePtr                 _images[2];
};

Consumer camera[3] = { "camera1", "camera2", "camera3" };

/*
 * \\class MenuCallback
 *
 * created on: Jan 17, 2020
 *
 */
class MenuCallback : public CallbackInterface
{
public:
  MenuCallback(window::Window* window) : _window(window) {}

  virtual void                    run(int camera)
  {
    ::camera[camera].trigger();
  }

  virtual void                    dir(int camera,const char* directory)
  {
    ::camera[camera].set_destination(directory);
  }

  virtual const char*             destination(int camera)
  {
    return ::camera[camera].get_destination();
  }

private:
  window::Window*                 _window;
};

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
  fm::get()->init();

  image::ISPManager isp_manager;

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
      deserializer->load_script(script_file.c_str(), meta_args);
      ids.push_back(id);
    }
  }

  char cwd[1024];
  getcwd(cwd, sizeof(cwd));

  // Check Displays

  window::CameraWindow *wnd = nullptr;
  std::vector<uint16_t> cam_des;

  image::ISP* current_isp = nullptr;
  if (devices.size() != 0 && !meta_args.get<bool>("group_isp") && !meta_args.get<bool>("no_isp"))
    current_isp = isp_manager.new_isp();

  for (auto device : devices)
  {
    std::cout << "Activating cameras " << device << std::endl;
    int id = strtol(device.substr(6).c_str(),nullptr,0);

    Deserializer* des = DeviceManager::get()->get_device(id);
    if (des != nullptr)
    {
      if (meta_args.get<bool>("group_isp") && !meta_args.get<bool>("no_isp"))
        current_isp = isp_manager.new_isp(true);

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

          cam_des.push_back(id << 8 | index);
          if (current_isp != nullptr)
            current_isp->add_camera(cam);

          cam->register_consumer(&camera[id],Metadata().set("<id>", index));
          camera[id].set_destination(cwd);
        }
      }
    }
  }

  if (wnd != nullptr)
    wnd->show(nullptr);

  MenuCallback aa(wnd);
  CameraWindow* cm = new CameraWindow;
  cm->make_window(&aa)->show();
  Fl::run();
  delete cm;

    
//
//  char buffer[1024];
//  std::string line;
//  do
//  {
//    std::cout << "T.... :" << line << ":" << std::endl;
//    std::cin.getline(buffer, sizeof(buffer));
//    line = buffer;
//
//    if (Utils::stristr(line, "exp") == 0)
//    {
//      line = line.substr(3);
//      double exposure = strtod(line.c_str(), nullptr);
//
//      for (auto id : cam_des)
//      {
//        Deserializer *des = DeviceManager::get()->get_device(id >> 8);
//        if (des != nullptr)
//        {
//          Camera *cam = des->get_camera(id & 0xff);
//          if (cam != nullptr)
//          {
//            if (exposure == 0.0)
//            {
//              double read = cam->get_exposure();
//              std::cout << "Exposure = " << read << std::endl;
//            }
//            else
//              cam->set_exposure(exposure);
//          }
//        }
//      }
//    }
//    else if (Utils::stristr(line, "gain") == 0)
//    {
//      line = line.substr(4);
//      long gain = strtol(line.c_str(), nullptr, 0);
//
//      for (auto id : cam_des)
//      {
//        Deserializer *des = DeviceManager::get()->get_device(id >> 8);
//        if (des != nullptr)
//        {
//          Camera *cam = des->get_camera(id & 0xff);
//          if (cam != nullptr)
//            cam->set_gain((eCameraGain)gain);
//        }
//      }
//    }
//    else if (Utils::stristr(line, "temp") == 0)
//    {
//      line = line.substr(4);
//      long gain = strtol(line.c_str(), nullptr, 0);
//
//      for (auto id : cam_des)
//      {
//        Deserializer *des = DeviceManager::get()->get_device(id >> 8);
//        if (des != nullptr)
//        {
//          Camera *cam = des->get_camera(id & 0xff);
//          if (cam != nullptr)
//          {
//            double t0 = cam->get_temperature(0);
//            double t1 = cam->get_temperature(1);
//            std::cout << "Camera (" << cam->name() << ") T0 = " << t0 << ", T1 = " << t1 << std::endl;
//          }
//        }
//      }
//    }
//
//  } while (line != "q");

  isp_manager.release();
  wm::get()->release();

  DeviceManager::get()->stop_all();

  return 0;
}
