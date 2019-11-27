/*
 * I2CDevice.cpp
 *
 *  Created on: Aug 19, 2019
 *      Author: daniel
 */

#include "Deserializer.hpp"
#include "Utils.hpp"
#include "Camera.hpp"

#include <iostream>
#include "DeviceManager.hpp"

namespace brt
{
namespace jupiter
{

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

  brt_camera_xfer xfer;
  xfer._deser_id = _id;
  xfer._device_addr = address;
  xfer._register_address_size = static_cast<uint8_t>(offset_size);
  xfer._register_address = offset;
  xfer._data_size = static_cast<uint16_t>(size);

  if ((ret = ioctl(DeviceManager::get()->handle(),BRT_CAMERA_READ,(unsigned long)&xfer)) < 0)
  {
    std::cerr << "Read error " << errno << std::endl;
    return false;
  }

  memcpy(buffer, xfer._data,size);
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

  brt_camera_xfer xfer;
  xfer._deser_id = _id;
  xfer._device_addr = address;
  xfer._register_address_size = static_cast<uint8_t>(offset_size);
  xfer._register_address = offset;
  xfer._data_size = static_cast<uint16_t>(size);
  memcpy(xfer._data,buffer,size);

  if ((ret = ioctl(DeviceManager::get()->handle(),BRT_CAMERA_WRITE,(unsigned long)&xfer)) < 0)
  {
    std::cerr << "Read error " << errno << std::endl;
    return false;
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
bool Deserializer::load_script(const char *file_path)
{
  _script = ScriptPtr(file_path);
  if (!_script)
    return false;

  if (!_script->load())
    return false;

  _script->set<void*>("device", this);

  if (!_script->run())
    return false;

  // Activate Cameras
  size_t number_of_cameras = _script->get<int>("num_cameras", 0);
  for (size_t index = 0; index < number_of_cameras; index++)
    _cameras.push_back(new Camera(this, index));

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
  if (!_script)
    return false;

  return _script->run(text);
}


/*
 * \\fn bool bool I2CDevice::run_macro
 *
 * created on: Nov 1, 2019
 * author: daniel
 *
 */
bool Deserializer::run_macro(const char *macro_name,std::vector<script::Value> arguments /*= std::vector<script::Value>()*/)
{
  if (!_script)
    return false;

  script::Value dummy_result;
  return _script->run_macro(macro_name,dummy_result,arguments);
}

/*
 * \\fn bool Deserializer::run_macro
 *
 * created on: Nov 26, 2019
 * author: daniel
 *
 */
bool Deserializer::run_macro(const char *macro_name,script::Value& result,
                          std::vector<script::Value> arguments /*= std::vector<script::Value>()*/)
{
  if (!_script)
    return false;

  return _script->run_macro(macro_name,result, arguments);
}


} /* namespace jupiter */
} /* namespace brt */
