/*
 * CameraManager.cpp
 *
 *  Created on: Aug 19, 2019
 *      Author: daniel
 */

#include "CameraManager.hpp"

#include "Deserializer.hpp"

namespace brt
{
namespace jupiter
{

CameraManager CameraManager::_object;

/*
 * \\class Constructor CameraManager::CameraManager
 *
 * created on: Aug 19, 2019
 *
 */
CameraManager::CameraManager()
{
  _brt_handle = ::open(DRIVER_NAME, O_RDWR);
}

/*
 * \\fn Destructor CameraManager::~CameraManager
 *
 * created on: Aug 19, 2019
 * author: daniel
 *
 */
CameraManager::~CameraManager()
{
  while (_device_map.size())
  {
    delete (_device_map.begin())->second;
    _device_map.erase(_device_map.begin());
  }

  if (_brt_handle != INVALID_DEVICE_HANDLE)
    ::close(_brt_handle);
}

/*
 * \\fn I2CDevice* CameraManager::get_device
 *
 * created on: Aug 19, 2019
 * author: daniel
 *
 */
Deserializer* CameraManager::get_device(uint16_t id)
{
  device_map_type::iterator iter = _device_map.find(id);
  if (iter != _device_map.end())
    return iter->second;

  Deserializer* device = new Deserializer(id);
  _device_map[id] = device;

  return device;
}


} /* namespace jupiter */
} /* namespace brt */
