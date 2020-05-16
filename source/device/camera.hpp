/*
 * Camera.hpp
 *
 *  Created on: Nov 8, 2019
 *      Author: daniel
 */

#ifndef SOURCE_DEVICE_CAMERA_HPP_
#define SOURCE_DEVICE_CAMERA_HPP_

#include <atomic>
#include <thread>
#include <string>

#include <sys/ioctl.h>
#include <linux/videodev2.h>

#include <image.hpp>
#include <cuda_debayer.hpp>
//#include "image_processor.hpp"

#define EVENT_STOP                          (1)

namespace brt
{
namespace jupiter
{


enum eCameraGain
{
//###       0 = 1/8x    5 = 4/6x        10 = 6/3x
//###       1 = 2/8x    6 = 4/5x        11 = 7/3x
//###       2 = 2/7x    7 = 5/5x (1)    12 = 7/2x
//###       3 = 3/7x    8 = 5/4x        13 = 8/2x
//###       4 = 3/6x    9 = 6/4x        14 = 8/1x

  eCG_1_DIV_8_X = 0,
  eCG_2_DIV_8_X = 1,
  eCG_2_DIV_7_X = 2,
  eCG_3_DIV_7_X = 3,
  eCG_3_DIV_6_X = 4,
  eCG_4_DIV_6_X = 5,
  eCG_4_DIV_5_X = 6,
  eCG_5_DIV_5_X = 7, // 1x
  eCG_5_DIV_4_X = 8,
  eCG_6_DIV_4_X = 9,
  eCG_6_DIV_3_X = 10,
  eCG_7_DIV_3_X = 11,
  eCG_7_DIV_2_X = 12,
  eCG_8_DIV_2_X = 13,
  eCG_8_DIV_1_X = 14
};


#pragma pack(push, 1)
/*
 * \\struct Intrinsics
 *
 * created on: Jul 31, 2019
 *
 */
struct Intrinsics
{
  double                          _fx,_fy;
  double                          _cx,_cy;
  double                          _k1,_k2;

  double                          _p1,_p2;

  double                          _k3,_k4,_k5,_k6;
};

/*
 * \\struct Extrinsics
 *
 * created on: Jul 31, 2019
 *
 */
struct Extrinsics
{
  double                          _rx,_ry,_rz;
  double                          _tx,_ty,_tz;

  double                          _rot[3][3];
};

/*
 * \\struct CameraParameters
 *
 * created on: Jul 31, 2019
 *
 */
struct CameraParameters
{
  uint16_t                        _version;         // offset 0
  uint8_t                         _reserved1[6];

  char                            _sn[13];          // offset 8
  uint8_t                         _reserved2[11];

  char                            _pn[8];           // offset 32
  uint8_t                         _reserved3[4];

  uint8_t                         _pn_revision;     // offset 44
  uint8_t                         _reserved4[3];

  char                            _mf_sn[11];       // offset 48
  uint8_t                         _reserved5[9];

  char                            _mf_an[10];       // offset 68
  uint8_t                         _reserved6[18];

  char                            _mf_mn[19];       // offset 96
  uint8_t                         _reserved7[5];

  char                            _mf_date_code[4]; // offset 120
  uint8_t                         _reserved8[4];

  uint16_t                        _stereoVersion;   // offset 128
  uint8_t                         _reserved9[6];

  double                          _calib_timestamp; // offset 136
  uint8_t                         _reserved10[4];

  double                          _focal_length_scale; // offset 148
  uint8_t                         _reserved11[4];

  Intrinsics                      _lense_parameters;        // offset 160
  Intrinsics                      _companion_lense_params;  // offset 256

  Extrinsics                      _stereo_params;           // offset 352

  Intrinsics                      _lense_offset;            // offset 480
  Intrinsics                      _companion_lense_offset;  // offset 576

  Extrinsics                      _stereo_offset;           // offset 672
  uint8_t                         _reserved12[48];

  struct
  {
    uint16_t                        _xPosition;               // offset 840
    uint16_t                        _yPosition;               // offset 842

    uint16_t                        _stickerHeight;           // offset 844
    uint16_t                        _stickerWidth;            // offset 846

    uint16_t                        _band_grayValue[6];       // offset 848
    uint16_t                        _focusValueGraySticker;   // offset 860

  }                               _GQ_Gray_Strip_Params;    // offset 840
  uint8_t                         _reserved13[2];

  uint16_t                        _exposureTime;            // offset 864

  uint8_t                         _freeSpace[30];           // offset 866

  float                           _rgmMatrix[4][3];         // offset 896
  uint8_t                         _reserved14[16];

  float                           _lensShading[100];        // offset 960
  uint8_t                         _reserved15[4];

  uint8_t                         _camera_type;             // offset 1364
  uint8_t                         _reserved16[11];

  struct
  {
    uint16_t                        _xPosition;               // offset 1376
    uint16_t                        _yPosition;               // offset 1378
    uint16_t                        _height;                  // offset 1380
    uint16_t                        _width;                   // offset 1382
  }                               _cropped_image;           // offset 1376
  uint8_t                         _reserved17[4];

  uint16_t                        _cameraPosition;          // offset 1388
  uint8_t                         _reserved18[18];

  struct
  {
    uint16_t                        _xPosition;                 // offset 1408
    uint16_t                        _yPosition;                 // offset 1410
    uint16_t                        _stickerHeight;             // offset 1412
    uint16_t                        _stickerWidth;              // offset 1414

    uint16_t                        _redIllimunationValue;        // offset 1416
    uint16_t                        _greenIllimunationValue;      // offset 1418
    uint16_t                        _darkBlueIllimunationValue;   // offset 1420
    uint16_t                        _yellowIllimunationValue;     // offset 1422
    uint16_t                        _blueIllimunationValue;       // offset 1424
    uint16_t                        _magentaIllimunationValue;    // offset 1426

    uint16_t                        _redValue;                    // offset 1428
    uint16_t                        _greenValue;                  // offset 1430
    uint16_t                        _darkBlueValue;               // offset 1432
    uint16_t                        _yellowValue;                 // offset 1434
    uint16_t                        _blueValue;                   // offset 1436
    uint16_t                        _magentaValue;                // offset 1438

    uint16_t                        _redHue;                      // offset 1440
    uint16_t                        _greenHue;                    // offset 1442
    uint16_t                        _darkBlueHue;                 // offset 1444
    uint16_t                        _yellowHue;                   // offset 1446
    uint16_t                        _blueHue;                     // offset 1448
    uint16_t                        _magentaHue;                  // offset 1450

    uint16_t                        _focusValueColorSticker;      // offset 1452

  }                               _GQ_Color_Strip_Parameters; //  offset 1408
  uint8_t                         _reserved19[18];

  struct
  {
    uint16_t                        _x;                         // offset 1472
    uint16_t                        _y;                         // offset 1474

    uint16_t                        _width;                     // offset 1476
    uint16_t                        _height;                    // offset 1478

    uint16_t                        _illumination;              // offset 1480
    uint16_t                        _focus;                     // offset 1482
  }                               _GQ2_Calibration_Target;    //  offset 1472
  uint8_t                         _reserved20[20];

  struct
  {
    float                           _x;                         // offset 1504
    float                           _y;                         // offset 1508
    float                           _z;                         // offset 1512
    float                           _roll;                      // offset 1516
    float                           _pitch;                     // offset 1520
    float                           _yaw;                       // offset 1524

  }                               _GQ2_Camera_Misalignment;   // offset 1504

};

#pragma pack(pop)


class Deserializer;

/*
 * \\class Camera
 *
 * created on: Nov 8, 2019
 *
 */
class Camera : public image::ImageProducer
{
public:
  Camera(Deserializer* owner,int id,const Value::byte_buffer& = Value::byte_buffer());
  virtual ~Camera();

          int                     id() const { return _id; }

          bool                    start_streaming();
          bool                    stop_streaming();

          const v4l2_format*      format() const { return &_fmt; }

          void                    set_exposure(double ms);
          double                  get_exposure();
          double                  get_temperature(int temp_sensor_id);

          void                    read_exposure();
          void                    set_gain(eCameraGain);

          std::string             name() const { return _device_name; }

          image::ImageProducer*   debayer_producer() { return &_ip; }

  virtual const CameraParameters* get_camera_parameters_bin() const { return &_camera_params; }
  virtual void                    get_camera_parameters_json(std::string&);

private:
          bool                    open_device();

          bool                    init_device();
          bool                    uninit_device();

          bool                    init_read();
          bool                    init_mmap();
          bool                    start_capturing();
          bool                    stop_capturing();


          void                    main_loop();
          bool                    read_frame();


          inline int xioctl(int fh, int request, void *arg)
          {
            int r;

            do
            {
              r = ioctl(fh, request, arg);
            }
            while (-1 == r && EINTR == errno);

            return r;
          }
private:
  Deserializer*                   _owner;
  int                             _id;
  bool                            _active;
  std::string                     _device_name;
  int                             _handle;

  std::atomic_bool                _terminate;
  std::atomic_int                 _skip_frames;
  std::thread                     _thread;
  int                             _pipe[2];

  enum io_method
  {
    IO_METHOD_READ, IO_METHOD_MMAP
  }                               _io_method;
  v4l2_format                     _fmt;

  struct buffer
  {
    void *start;
    size_t length;
  }                               *_buffers;
  unsigned int                    _n_buffers;

  Debayer                         _ip;
  CameraParameters                _camera_params;
};

} /* namespace jupiter */
} /* namespace brt */

#endif /* SOURCE_DEVICE_CAMERA_HPP_ */
