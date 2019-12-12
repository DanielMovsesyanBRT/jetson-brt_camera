//
// Created by daniel on 3/28/19.
//

#ifndef I2C_00_I2CACTION_HPP
#define I2C_00_I2CACTION_HPP

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <map>
#include <cstring>

#include <script.hpp>

#include <metadata.hpp>
#include <value.hpp>

namespace brt {

namespace jupiter {

/**
 *
 */
class ActionRead : public script::ScriptAction
{
public:
  ActionRead()
  : _num_bytes(nullptr)
  , _device_address(nullptr)
  , _offset(nullptr)
  , _target(nullptr)
  {  }

  virtual ~ActionRead()
  {
    if (_num_bytes != nullptr)
      delete _num_bytes;

    if (_device_address != nullptr)
      delete _device_address;

    if (_offset != nullptr)
      delete _offset;

    if (_target != nullptr)
      delete _target;
  }

  virtual bool                    do_action(script::Session&);
  virtual bool                    extract(script::ParserEnv&);

private:
  script::Expression*             _num_bytes;
  script::Expression*             _device_address;

  script::Expression*             _offset;
  script::Expression*             _target;
};

/**
 *
 */
class ActionWrite : public script::ScriptAction
{
public:
  ActionWrite() : _device_address(nullptr), _offset(nullptr), _bytes()  {}
  virtual ~ActionWrite()
  {
    if (_device_address != nullptr)
      delete _device_address;

    if (_offset != nullptr)
      delete _offset;

    if (_bytes != nullptr)
      delete _bytes;
  }

  virtual bool                    do_action(script::Session&);
  virtual bool                    extract(script::ParserEnv&);

private:
  script::Expression*             _device_address;
  script::Expression*             _offset;
  script::ExpressionArray*        _bytes;
};


} // jupiter
} // brt

#endif //I2C_00_I2CACTION_HPP
