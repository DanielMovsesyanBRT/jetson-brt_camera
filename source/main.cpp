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
#include "png.h"
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
      _images[id] = box[0];//->get_bits();

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

  /**
   * \fn  writeImage
   *
   * @param  filename : const char* 
   * @param  width :  int 
   * @param  height :  int 
   * @param  *buffer :  uint8_t 
   * @param  title : const char* 
   * \brief <description goes here>
   */
  void writeImage(const char* filename, int width, int height, uint8_t *buffer,const char* title)
  {
    FILE *fp = NULL;
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    png_bytep row = NULL;

    try
    {
      // Open file for writing (binary mode)
      fp = fopen(filename, "wb");
      if (fp == NULL)
      {
        fprintf(stderr, "Could not open file %s for writing\n", filename);
        throw 1;
      }

      // Initialize write structure
      png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
      if (png_ptr == NULL)
      {
        fprintf(stderr, "Could not allocate write struct\n");
        throw 1;
      }

      // Initialize info structure
      info_ptr = png_create_info_struct(png_ptr);
      if (info_ptr == NULL)
      {
        fprintf(stderr, "Could not allocate info struct\n");
        throw 1;
      }

      // Setup Exception handling
      if (setjmp(png_jmpbuf(png_ptr)))
      {
        fprintf(stderr, "Error during png creation\n");
        throw 1;
      }

      png_init_io(png_ptr, fp);

      // Write header (8 bit colour depth)
      png_set_IHDR(png_ptr, info_ptr, width, height,
          8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
          PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

      // Set title
      if (title != NULL)
      {
        png_text title_text;
        title_text.compression = PNG_TEXT_COMPRESSION_NONE;
        title_text.key = (png_charp)"Title";
        title_text.text = (png_charp)title;
        png_set_text(png_ptr, info_ptr, &title_text, 1);
      }

      png_write_info(png_ptr, info_ptr);

      // Allocate memory for one row (3 bytes per pixel - RGB)
      row = (png_bytep) malloc(4 * width * sizeof(png_byte));
  //
      // Write image data
      int x, y;
      for (y=0 ; y<height ; y++)
      {
        for (x=0 ; x<width ; x++)
        {
          *((uint16_t*)&row[x * 4]) = *((uint16_t*)&buffer[(y*width +x) * 8]) >> 8;
          *((uint16_t*)&row[x * 4 + 1]) = *((uint16_t*)&buffer[(y*width +x) * 8 + 2]) >> 8;
          *((uint16_t*)&row[x * 4 + 2]) = *((uint16_t*)&buffer[(y*width +x) * 8 + 4]) >> 8;
          *((uint16_t*)&row[x * 4 + 3]) = *((uint16_t*)&buffer[(y*width +x) * 8  + 6]) >> 8;
        }
        png_write_row(png_ptr, row);
      }

      // End write
      png_write_end(png_ptr, NULL);
    }
    catch(...)
    {

    }

    if (fp != NULL) fclose(fp);
    if (info_ptr != NULL) png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
    if (png_ptr != NULL) png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    if (row != NULL) free(row);
  }


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
          image::RawRGBPtr img = _images[0]->get_bits();
          uint32_t timetag = _images[0]->get<unsigned long>("time_tag", 0);

          std::unique_lock<std::mutex> l(_mutex);
          uint32_t w = img->width(), h = img->height(), bytes = img->depth(); /* RAW12*/;

          if (img->type() == image::eBayer)
          {
            _file_name[0] = Utils::string_format("%s/%s_%08X_left.raw", _directory.c_str(),_prefix.c_str(), timetag);
            std::ofstream raw_file (_file_name[0], std::ios::out | std::ios::binary);
            raw_file.write(reinterpret_cast<const char*>(&w), sizeof(w));
            raw_file.write(reinterpret_cast<const char*>(&h), sizeof(h));
            raw_file.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
            raw_file.write(reinterpret_cast<const char*>(img->bytes()),w * h * bytes);
          }
          else
          {
            _file_name[0] = Utils::string_format("%s/%s_%08X_left.png", _directory.c_str(),_prefix.c_str(),timetag);
            writeImage(_file_name[0].c_str(), (int)w, (int)h, img->bytes(), "left");
          }

          img = _images[1]->get_bits();
          timetag = _images[1]->get<unsigned long>("time_tag", 0);
          w = img->width(); h = img->height(); bytes = img->depth(); /* RAW12*/;

          if (img->type() == image::eBayer)
          {
            _file_name[1] = Utils::string_format("%s/%s_%08X_right.raw", _directory.c_str(),_prefix.c_str(),timetag);
            std::ofstream raw_file(_file_name[1], std::ios::out | std::ios::binary);
            raw_file.write(reinterpret_cast<const char*>(&w), sizeof(w));
            raw_file.write(reinterpret_cast<const char*>(&h), sizeof(h));
            raw_file.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));

            raw_file.write(reinterpret_cast<const char*>(img->bytes()),w * h * bytes);
          }
          else
          {
            _file_name[1] = Utils::string_format("%s/%s_%08X_right.png", _directory.c_str(),_prefix.c_str(),timetag);
            writeImage(_file_name[1].c_str(), (int)w, (int)h, img->bytes(), "right");
          }

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

  //image::RawRGBPtr                _images[2];
  image::ImagePtr                 _images[2];

  std::mutex                      _mutex;
  std::thread                     _thread;
  std::atomic_bool                _terminate;
  std::condition_variable         _event, _event_back;

  std::string                     _file_name[2];
};

Consumer camera_raw[3] = { "camera1", "camera2", "camera3" };
Consumer camera_png[3] = { "camera1", "camera2", "camera3" };


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
    ::camera_raw[camera].trigger();
    ::camera_png[camera].trigger();
  }

  virtual void                    dir(int camera,const char* directory)
  {
    ::camera_raw[camera].set_destination(directory);
    ::camera_png[camera].set_destination(directory);
  }

  virtual const char*             destination(int camera)
  {
    return ::camera_raw[camera].get_destination();
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

          //cam->register_consumer(&camera[id],Metadata().set("<id>", index));
          if (cam->debayer_producer() != nullptr)
            cam->debayer_producer()->register_consumer(&camera_png[id],Metadata().set("<id>", index));

          cam->register_consumer(&camera_raw[id],Metadata().set("<id>", index));

          camera_raw[id].set_destination(cwd);
          camera_png[id].set_destination(cwd);

          if (print_eeprom)
          {
            std::string json;
            cam->get_camera_parameters_json(json);
            std::cout << json << std::endl << std::endl;
          }
        }
      }
      camera_raw[id].start();
      camera_png[id].start();
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

          if (::camera_raw[index].is_started())
          {
            ::camera_raw[index].trigger(true);
            std::string file_name[2];
            ::camera_raw[index].get_last_file_names(file_name);

            std::cout << "  " << file_name[0] << ", " << file_name[1];
          }
          else
            std::cout << " NOT INITIALIZED!!!";

          if (::camera_png[index].is_started())
          {
            ::camera_png[index].trigger(true);
            std::string file_name[2];
            ::camera_png[index].get_last_file_names(file_name);

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
