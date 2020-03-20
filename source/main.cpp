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

#include <map>
#include <iostream>
#include <fstream>
#include <string>

#include <stdint.h>

#include "fltk_interface_exports.hpp"
#include "device/camera.hpp"
#include "device/deserializer.hpp"
#include "device/device_manager.hpp"
#include "utils.hpp"
#include "console_cli.hpp"


using namespace brt::jupiter;

#define FLTK_LIB_NAME                       "libfltk_interfaces.so"
/*
 * \\class FLTKLibrary
 *
 * created on: Mar 16, 2020
 *
 */
class FLTKLibrary : public DynamicLibrary<FLTKLibrary>
{
friend DynamicLibrary<FLTKLibrary>;
  FLTKLibrary() : DynamicLibrary<FLTKLibrary>(_path.c_str()) {}

public:
  static  void                    set_fltk_path(const char *path) { _path = path; }
private:
  static  std::string             _path;
};

std::string FLTKLibrary::_path = "";

/*
 * \\class Consumer
 *
 * created on: Jan 16, 2020
 *
 */
class Consumer : public image::ImageConsumer
{
public:
  Consumer(const char* prefix) : _prefix(prefix), _directory(), _flag(false), _unique(0), _terminate(false) {}

  Consumer(const Consumer& cons)
  : _prefix(cons._prefix)
  , _directory(cons._directory)
  , _flag(cons._flag.load())
  , _unique(cons._unique)
  , _terminate(cons._terminate.load())
  {}

  virtual ~Consumer()
  {
    stop();
  }

  virtual void                    consume(image::ImageBox box)
  {
    if (box.empty())
      return;

    int id = box[0]->get("<id>", -1);
    if ((id == -1) || (id > 1))
      return;

    if (!_images[id])
      _images[id] = box[0]->get_bits();

    _event.notify_all();
  }

  void                            start()
  {
    if (!_thread.joinable())
    {
      _thread = std::thread([](Consumer *cons)
      {
        cons->thread_loop();
      },this);
    }
  }

  void                            stop()
  {
    if (_thread.joinable())
    {
      _terminate.store(true);
      _event.notify_all();
      _thread.join();
    }
  }

  bool                            is_started() const
  {
    return _thread.joinable();
  }

  void                            set_destination(const char* dir)
  {
    _directory = dir;
  }

  const char*                     get_destination() const
  {
    return _directory.c_str();
  }

  void                            trigger(bool block = false)
  {
    bool expected = false;
    _flag.compare_exchange_strong(expected, true);

    if (block)
    {
      std::unique_lock<std::mutex> l(_mutex);
      _event_back.wait(l);
    }
  }

  void                            get_last_file_names(std::string name[2])
  {
    std::lock_guard<std::mutex> l(_mutex);
    name[0] = _file_name[0];
    name[1] = _file_name[1];
  }

private:
  void                            thread_loop()
  {
    while(!_terminate)
    {
      {
        std::unique_lock<std::mutex> l(_mutex);
        _event.wait(l);
      }

      if (_images[0] && _images[1])
      {
        bool expected = true;
        if (_flag.compare_exchange_strong(expected, false))
        {
          std::unique_lock<std::mutex> l(_mutex);
          _file_name[0] = Utils::string_format("%s/%s_%04d_left.raw", _directory.c_str(),_prefix.c_str(),_unique);
          uint32_t w = _images[0]->width(), h = _images[0]->height(), bytes = 2 /* RAW12*/;

          std::ofstream raw_file (_file_name[0], std::ios::out | std::ios::binary);
          raw_file.write(reinterpret_cast<const char*>(&w), sizeof(w));
          raw_file.write(reinterpret_cast<const char*>(&h), sizeof(h));
          raw_file.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
          raw_file.write(reinterpret_cast<const char*>(_images[0]->bytes()),w * h * bytes);

          _file_name[1] = Utils::string_format("%s/%s_%04d_right.raw", _directory.c_str(),_prefix.c_str(),_unique);

          w = _images[1]->width(), h = _images[1]->height(), bytes = 2 /* RAW12*/;

          raw_file = std::ofstream(_file_name[1], std::ios::out | std::ios::binary);
          raw_file.write(reinterpret_cast<const char*>(&w), sizeof(w));
          raw_file.write(reinterpret_cast<const char*>(&h), sizeof(h));
          raw_file.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));

          raw_file.write(reinterpret_cast<const char*>(_images[1]->bytes()),w * h * bytes);

          _unique++;
          _event_back.notify_all();
        }
        _images[0].reset();
        _images[1].reset();
      }
    }
  }

private:
  std::string                     _prefix;
  std::string                     _directory;

  std::atomic_bool                _flag;
  uint32_t                        _unique;

  image::RawRGBPtr                _images[2];

  std::mutex                      _mutex;
  std::thread                     _thread;
  std::atomic_bool                _terminate;
  std::condition_variable         _event, _event_back;

  std::string                     _file_name[2];
};

Consumer camera[3] = { "camera1", "camera2", "camera3" };


/*
 * \\class MenuCallback
 *
 * created on: Jan 17, 2020
 *
 */
class MenuCallback : public fltk::CallbackInterface
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
  std::string _cur_file = argv[0];
  size_t idx = _cur_file.rfind('/');
  if (idx != std::string::npos)
  {
    _cur_file = _cur_file.substr(0, idx + 1) + FLTK_LIB_NAME;
    FLTKLibrary::set_fltk_path(_cur_file.c_str());
  }
  else
    FLTKLibrary::set_fltk_path(FLTK_LIB_NAME);

  image::ISPManager isp_manager;

  brt::jupiter::Metadata meta_args;
  meta_args.parse(argc,argv);

  bool cli_only = meta_args.get<bool>("cli_only",false);
  bool print_eeprom = meta_args.get<bool>("print_eeprom",false);

  if (!cli_only)
  {
    cli_only = cli_only || (!FLTKLibrary::get() || !window::GLLibrary::get() || !window::X11Library::get());
    if (cli_only)
    {
      std::cout << "Cannot properly initialize one of the graphic libraries." << std::endl;
      std::cout << "  Turning into CLI only mode" << std::endl;
    }
  }

  if (!cli_only)
  {
    wm::get()->init();
    FLTKLibrary::get().call<void>("fltk_initialize");
  }


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
          if (cam->start_streaming() && !cli_only)
          {
            if (wnd == nullptr)
            {
              wnd = window::CameraWindow::create("Video Streaming", nullptr,
                  cam->format()->fmt.pix.width,
                  cam->format()->fmt.pix.height);
            }
            wnd->add_subwnd(cam->debayer_producer());
          }

          cam_des.push_back(id << 8 | index);
          if (current_isp != nullptr)
            current_isp->add_camera(cam);

          cam->register_consumer(&camera[id],Metadata().set("<id>", index));
          camera[id].set_destination(cwd);

          if (print_eeprom)
          {
            std::string json;
            cam->get_camera_parameters_json(json);
            std::cout << json << std::endl << std::endl;
          }
        }
      }
      camera[id].start();
    }
  }

  if (!cli_only)
  {
    if (wnd != nullptr)
      wnd->show(nullptr);

    MenuCallback aa(wnd);
    FLTKLibrary::get().call<void,brt::jupiter::fltk::CallbackInterface*>
              ("fltk_interface_run", &aa);
    wm::get()->release();
  }
  else
  {
    // CLI only mode
    ConsoleCLI cli;
    std::cout << std::endl
              << std::endl
              << std::endl
              << std::endl
              << std::endl;

    cli.move_to(0, 5);
    std::cout << "Press camera index [0, 1, 2] to capture camera image. q or Q to quit" << std::endl;
    std::cout << "0:" << std::endl << "1:" << std::endl << "2:";
    cli.move_to(-2,-1);

    int character;
    while ( ((character = getchar()) != 'Q') && (character != 'q') )
    {
      if (std::isdigit(character))
      {
        int index = character - '0';
        if ((index < 0) || (index > 2))
        {
          cli.move_to(-100,0);
          std::cout << "Invalid index " << character;
        }
        else
        {
          cli.move_to(-100,0);
          cli.move_to(2,3 - index);

          if (::camera[index].is_started())
          {
            ::camera[index].trigger(true);
            std::string file_name[2];
            ::camera[index].get_last_file_names(file_name);

            std::cout << "  " << file_name[0] << ", " << file_name[1];
          }
          else
            std::cout << " NOT INITIALIZED!!!";

          cli.move_to(-100,0);
          cli.move_to(0,index - 3);
        }
      }
      else
      {
        std::cout << "Invalid key " << (char)character;
        cli.move_to(-100,0);
      }
    }
  }
  isp_manager.release();

  DeviceManager::get()->stop_all();
  return 0;
}
