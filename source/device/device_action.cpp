//
// Created by daniel on 3/28/19.
//

#include "device_action.hpp"
#ifdef HARDWARE
#include "deserializer.hpp"
#include "device_manager.hpp"
#endif

#include <string.h>
#include <ctype.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <parser.hpp>
#include <utils.hpp>

namespace brt {

namespace jupiter {


/**
 *
 * @return
 */
bool ActionRead::do_action(script::Session& session)
{
  if ((_num_bytes == nullptr) ||
      (_device_address == nullptr) ||
      (_offset == nullptr) ||
      (_target == nullptr))
  {
    return false;
  }

  int num_bytes = _num_bytes->evaluate(&session);
  Value offset = _offset->evaluate(&session);

  uint8_t* buffer = new uint8_t[num_bytes];

#ifdef HARDWARE
  int address = _device_address->evaluate(&session);

  Deserializer* device = static_cast<Deserializer*>((void*)session.var("device"));
  if (device == nullptr)
    return false;

  bool result = device->read((uint8_t)address, (int)offset, offset.size(), buffer,num_bytes);
#else
  bool result = true;
#endif
  if (result)
  {
    Value val = _target->evaluate(&session);
    val.set_byte_array(buffer, num_bytes, false);
  }
#ifdef HARDWARE
  else
    std::cerr << "Unable to read from device :0x" << std::hex <<(int)address << std::endl;
#endif

  delete[] buffer;
  return result;
}

/**
 *
 * @param str
 * @return
 */
bool ActionRead::extract(script::ParserEnv& ps)
{
  script::Parser pr;
  script::Expression *expr = pr.parse(ps);

  script::ExpressionArray* exArray = dynamic_cast<script::ExpressionArray*>(expr);
  if ((exArray == nullptr) || (exArray->num_expresions() < 4))
  {
    if (expr != nullptr)
      delete expr;

    throw script::ParserException::create("Invalid Read expression");
  }

  _num_bytes = exArray->get_expression(0);
  _device_address = exArray->get_expression(1);
  _offset = exArray->get_expression(2);
  _target = exArray->get_expression(3);

  exArray->detach(4);
  delete expr;

  return true;
}

/**
 *
 * @return
 */
bool ActionWrite::do_action(script::Session& session)
{
  Value offset = _offset->evaluate(&session);

  if (_bytes == nullptr)
    return false;

  std::vector<uint8_t>  buffer;
  for (size_t index = 0; index < _bytes->num_expresions(); index++)
  {
    Value byte = _bytes->get_expression(index)->evaluate(&session);

    int size = byte.size();
    int byteValue = byte;
    int shift = 8 * (size - 1);

    while (size-- > 0)
    {
      buffer.push_back((byteValue >> shift) & 0xFF);
      shift -= 8;
    }
  }

#ifdef HARDWARE
  int address = _device_address->evaluate(&session);

  Deserializer* device = static_cast<Deserializer*>((void*)session.var("device"));
  if (device == nullptr)
    return false;

  return device->write((uint8_t)address, (int)offset, offset.size(), &buffer.front(),buffer.size());
#else
  return true;
#endif
}

/**
 *
 * @param str
 * @return
 */
bool ActionWrite::extract(script::ParserEnv& ps)
{
  script::Parser pr;
  script::Expression *expr = pr.parse(ps);

  _bytes = dynamic_cast<script::ExpressionArray*>(expr);
  if ((_bytes == nullptr) || (_bytes->num_expresions() < 2))
  {
    delete expr;
    throw script::ParserException::create("Invalid Write expression");
  }

  _device_address = _bytes->get_expression(0);
  _offset = _bytes->get_expression(1);
  _bytes->detach(2);
  if (_bytes->num_expresions() == 0)
  {
    delete _bytes;
    _bytes = nullptr;
  }

  return true;
}


} // jupiter
} // brt
