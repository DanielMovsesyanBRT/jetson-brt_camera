/*
 * I2CDevice.cpp
 *
 *  Created on: Aug 19, 2019
 *      Author: daniel
 */

#include "deserializer.hpp"
#include "camera.hpp"
#include "device_manager.hpp"
#include "device_action.hpp"

#include <iostream>
#include <algorithm>

#include <utils.hpp>


namespace brt
{
namespace jupiter
{



/*
 * \\fn script::ScriptAction* Deserializer::ActionCreator::get_action
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
script::ScriptAction* Deserializer::ActionCreator::create_action(const char* action)
{
  if (action == nullptr)
    return nullptr;

  switch (action[0])
  {
  case 'r':
  case 'R':
    return new ActionRead();

  case 'w':
  case 'W':
    return new ActionWrite();

  default:
    break;
  }

  return nullptr;
}


/*
 * \\fn Constructor I2CDevice::I2CDevice
 *
 * created on: Aug 19, 2019
 * author: daniel
 *
 */
Deserializer::Deserializer(uint16_t id)
: _handle(INVALID_DEVICE_HANDLE)
, _id(id)
{
//  _i2cname = Utils::string_format("/dev/i2c-%d", i2c_id);
//  _handle = open(_i2cname.c_str(), O_RDWR);
}

/*
 * \\fn Destructor I2CDevice::~I2CDevice
 *
 * created on: Aug 19, 2019
 * author: daniel
 *
 */
Deserializer::~Deserializer()
{
  if (_handle != INVALID_DEVICE_HANDLE)
    close(_handle);
}

/*
 * \\fn bool I2CDevice::read
 *
 * created on: Aug 19, 2019
 * author: daniel
 *
 */
bool Deserializer::read(uint8_t address, uint16_t offset, size_t offset_size, uint8_t* buffer, size_t size)
{
  int ret = 0;
  if (DeviceManager::get()->handle() == INVALID_DEVICE_HANDLE)
    return false;

  while (size > 0)
  {
    size_t transfer_size = std::min(size, (size_t)MAX_DATA_SIZE);

    brt_camera_xfer xfer;
    xfer._deser_id = _id;
    xfer._device_addr = address;
    xfer._register_address_size = static_cast<uint8_t>(offset_size);
    xfer._register_address = offset;
    xfer._data_size = static_cast<uint16_t>(transfer_size); //size);

    if ((ret = ioctl(DeviceManager::get()->handle(),BRT_CAMERA_READ,(unsigned long)&xfer)) < 0)
    {
      std::cerr << "Read error " << errno << std::endl;
      return false;
    }

    memcpy(buffer, xfer._data,transfer_size); //size);
    buffer += transfer_size;
    size -= transfer_size;
    offset += transfer_size;
  }
  return true;
}

/*
 * \\fn bool I2CDevice::write
 *
 * created on: Aug 19, 2019
 * author: daniel
 *
 */
bool Deserializer::write(uint8_t address, uint16_t offset, size_t offset_size, const uint8_t* buffer, size_t size)
{
  int ret = 0;
  if (DeviceManager::get()->handle() == INVALID_DEVICE_HANDLE)
    return false;

  while (size > 0)
  {
    size_t transfer_size = std::min(size, (size_t)MAX_DATA_SIZE);

    brt_camera_xfer xfer;
    xfer._deser_id = _id;
    xfer._device_addr = address;
    xfer._register_address_size = static_cast<uint8_t>(offset_size);
    xfer._register_address = offset;
    xfer._data_size = static_cast<uint16_t>(transfer_size);
    memcpy(xfer._data,buffer,transfer_size);

    if ((ret = ioctl(DeviceManager::get()->handle(),BRT_CAMERA_WRITE,(unsigned long)&xfer)) < 0)
    {
      std::cerr << "Read error " << errno << std::endl;
      return false;
    }
    buffer += transfer_size;
    size -= transfer_size;
    offset += transfer_size;
  }

  return true;
}

/*
 * \\fn bool I2CDevice::load_script
 *
 * created on: Aug 20, 2019
 * author: daniel
 *
 */
bool Deserializer::load_script(const char *file_path,const Metadata& extra_args/* = Metadata()*/)
{
  ActionCreator ac;
  if (!_script.load_from_file(file_path,script::CreatorContainer(&ac)))
    return false;

  _script.set(Metadata(extra_args).set<void*>("device", this));
  _script.run();

  // Activate Cameras
  size_t number_of_cameras = static_cast<int>(_script.get("num_cameras"));
  for (size_t index = 0; index < number_of_cameras; index++)
  {
    Camera *cam = nullptr;
    if (_script.exist("camera_eeprom"))
    {
      Value::byte_buffer bb = _script.get("camera_eeprom").at(index);
      if (!bb.empty())
        cam = new Camera(this, index,bb);
    }

    if (cam == nullptr)
      cam = new Camera(this, index);
    _cameras.push_back(cam);
  }

  return true;
}

/*
 * \\fn bool I2CDevice::run_script
 *
 * created on: Oct 30, 2019
 * author: daniel
 *
 */
bool Deserializer::run_script(const char *text)
{
  if (_script.empty())
    return false;

  ActionCreator ac;
  script::Script sc;
  if (!sc.load(text,script::CreatorContainer(&ac)))
    return false;

  _script.run(sc);
  return true;
}


/*
 * \\fn bool bool I2CDevice::run_macro
 *
 * created on: Nov 1, 2019
 * author: daniel
 *
 */
Value Deserializer::run_macro(const char *macro_name,std::vector<Value> arguments /*= std::vector<Value>()*/)
{
  if (_script.empty())
    return Value();

  return _script.run_macro(macro_name,arguments);
}


} /* namespace jupiter */
} /* namespace brt */
