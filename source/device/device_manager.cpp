/*
 * CameraManager.cpp
 *
 *  Created on: Aug 19, 2019
 *      Author: daniel
 */

#include "device_manager.hpp"

#include "camera.hpp"
#include "deserializer.hpp"

namespace brt
{
namespace jupiter
{

DeviceManager DeviceManager::_object;

/*
 * \\class Constructor CameraManager::CameraManager
 *
 * created on: Aug 19, 2019
 *
 */
DeviceManager::DeviceManager()
{
  _brt_handle = ::open(DRIVER_NAME, O_RDWR);
  if (_brt_handle >= 0)
  {
    char  board_name[20];
    brt_board_name bname;
    bname._max_len = sizeof(board_name);
    bname._buf     = board_name;

    if (ioctl(_brt_handle, BRT_CAMERA_BOARD_NAME, (unsigned long)&bname) < 0)
      std::cerr << "Cannot access board name: " << errno << std::endl;
    else
      std::cout << "Board name is " << bname._buf << std::endl;
  }
}

/*
 * \\fn Destructor CameraManager::~CameraManager
 *
 * created on: Aug 19, 2019
 * author: daniel
 *
 */
DeviceManager::~DeviceManager()
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
Deserializer* DeviceManager::get_device(uint16_t id)
{
  device_map_type::iterator iter = _device_map.find(id);
  if (iter != _device_map.end())
    return iter->second;

  Deserializer* device = new Deserializer(id);
  _device_map[id] = device;

  return device;
}

/*
 * \\fn void DeviceManager::stop_all
 *
 * created on: Dec 2, 2019
 * author: daniel
 *
 */
void DeviceManager::stop_all()
{
  for (auto deser : _device_map)
  {
    if (deser.second != nullptr)
    {
      for (size_t cam_id = 0; cam_id < deser.second->num_cameras(); cam_id++)
      {
        Camera* cam = deser.second->get_camera(cam_id);
        if (cam != nullptr)
          cam->stop_streaming();
      }
    }
  }
}

} /* namespace jupiter */
} /* namespace brt */
