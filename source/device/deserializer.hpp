/*
 * I2CDevice.hpp
 *
 *  Created on: Aug 19, 2019
 *      Author: daniel
 */

#ifndef SOURCE_I2CDEVICE_HPP_
#define SOURCE_I2CDEVICE_HPP_

#include <string>
#include <vector>

#include "script_file.hpp"


namespace brt
{
namespace jupiter
{

class Camera;
/*
 * \\class I2CDevice
 *
 * created on: Aug 19, 2019
 *
 */
class Deserializer
{
public:
  Deserializer(uint16_t id);
  virtual ~Deserializer();

          int                     id() const { return _id; }

          bool                    read(uint8_t address, uint16_t offset, size_t offset_size, uint8_t* buffer, size_t size);
          bool                    write(uint8_t address, uint16_t offset, size_t offset_size, const uint8_t* buffer, size_t size);

          bool                    load_script(const char *file_path,const Metadata& extra_args = Metadata());

          bool                    run_script(const char *text);
          bool                    run_macro(const char *macro_name,std::vector<script::Value> arguments = std::vector<script::Value>());
          bool                    run_macro(const char *macro_name,script::Value& result,std::vector<script::Value> arguments = std::vector<script::Value>());

          size_t                  num_cameras() const { return _cameras.size(); }
          Camera*                 get_camera(size_t index) const { return ((index >= num_cameras())?nullptr:_cameras[index]); }
private:
  int                             _handle;
  uint16_t                        _id;
  //std::string                     _i2cname;

  ScriptPtr                       _script;
  std::vector<Camera*>            _cameras;
};

} /* namespace jupiter */
} /* namespace brt */

#endif /* SOURCE_I2CDEVICE_HPP_ */
