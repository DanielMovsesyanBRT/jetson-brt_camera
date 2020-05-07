/*
 * I2CDeviceManager.hpp
 *
 *  Created on: Aug 19, 2019
 *      Author: daniel
 */

#ifndef SOURCE_DEVICE_DEVICE_MANAGER_HPP_
#define SOURCE_DEVICE_DEVICE_MANAGER_HPP_

#include <unordered_map>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>

#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>

#define INVALID_DEVICE_HANDLE               (-1)

#define MAKE_CAM_ID(did,cid)                ((did << 8) | cid)
#define CAM_ID(id)                          (id & 0xff)
#define DES_ID(id)                          ((id >> 8) & 0xff) // deserializer id

#define DRIVER_NAME                         ("/dev/brt_camera")

#define MAX_DATA_SIZE                       (256)
#define CAMERA_NAME_LEN                     (32)

/*
 * \\struct brt_camera_bulk
 *
 * created on: Nov 5, 2019
 *
 */
struct brt_camera_xfer
{
  uint8_t                         _deser_id;
  uint8_t                         _device_addr;
  uint16_t                        _register_address;
  uint8_t                         _register_address_size;

  uint16_t                        _data_size;
  uint8_t                         _data[MAX_DATA_SIZE];
};


/*
 * \\struct brt_camera_name
 *
 * created on: Nov 12, 2019
 *
 */
struct brt_camera_name
{
  uint8_t                         _deser_id;
  uint8_t                         _camera_id;
  char                            _name[CAMERA_NAME_LEN];
};

/*
 * \\struct brt_camera_trigger
 *
 * created on: Nov 12, 2019
 *
 */
struct brt_camera_trigger
{
  int                             _pwm_period; //ms
  int                             _duty_period;
};

/*
 * \\struct brt_board_name
 *
 * created on: Apr 16, 2020
 *
 */
struct brt_board_name
{
  int                             _max_len;
  char*                           _buf;
};

#define BRT_CAMERA_TRIGGER_ONOFF            _IOW('B',  0, int)
#define BRT_CAMERA_TRIGGER_SET_PWM          _IOW('B',  1, struct brt_camera_trigger)

#define BRT_CAMERA_GET_NAME                 _IOR('B',  2, struct brt_camera_name)
#define BRT_CAMERA_WRITE                    _IOW('B',  3, struct brt_camera_xfer)
#define BRT_CAMERA_READ                     _IOR('B',  4, struct brt_camera_xfer)

#define BRT_CAMERA_BOARD_NAME               _IOR('B',  5, struct brt_board_name)


namespace brt
{
namespace jupiter
{


class Deserializer;
/*
 * \\class CameraManager
 *
 * created on: Aug 19, 2019
 *
 */
class DeviceManager
{
  DeviceManager();
  virtual ~DeviceManager();

public:
  static  DeviceManager*          get() {return &_object; }
          Deserializer*           get_device(uint16_t);
          int                     handle() const { return _brt_handle; }
          void                    stop_all();

private:
  static  DeviceManager           _object;

  typedef  std::unordered_map<uint16_t,Deserializer*> device_map_type;
  device_map_type                 _device_map;
  int                             _brt_handle;

};

} /* namespace jupiter */
} /* namespace brt */

#endif /* SOURCE_DEVICE_DEVICE_MANAGER_HPP_ */
