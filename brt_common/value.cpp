//
// Created by Daniel Movsesyan on 2019-04-06.
//

#include "value.hpp"

#include <iostream>
#include <iomanip>
#include <string.h>

namespace brt {

namespace jupiter {

/**
 *
 * @return
 */
Value& Value::operator++()
{
  switch (_data->type())
  {
  case ValueData::BOOL:
    _data->set(true,_data->size());
    break;

  case ValueData::FLOAT:
    _data->set_float(_data->get_float() + 1.0,_data->size());
    break;

  case ValueData::ULONGLONG:
    _data->set_ull(_data->get_ull() + 1ull,_data->size());
    break;

  case ValueData::PTR:
    _data->set_ptr(reinterpret_cast<void*>(static_cast<std::uintptr_t>(_data->get_ull() + 1ull)),_data->size());
    break;

  default:
    _data->set_int(_data->get_int() + 1,_data->size());
    break;
  }
  return *this;
}


/**
 *
 * @return
 */
Value Value::operator++(int)
{
  Value result = *this;
  switch (_data->type())
  {
  case ValueData::BOOL:
    _data->set(true,_data->size());
    break;

  case ValueData::FLOAT:
    _data->set_float(_data->get_float() + 1.0,_data->size());
    break;

  case ValueData::ULONGLONG:
    _data->set_ull(_data->get_ull() + 1ull,_data->size());
    break;

  case ValueData::PTR:
    _data->set_ptr(reinterpret_cast<void*>(static_cast<std::uintptr_t>(_data->get_ull() + 1ull)),_data->size());
    break;

  default:
    _data->set_int(_data->get_int() + 1,_data->size());
    break;
  }
  return result;
}

/**
 *
 * @return
 */
Value& Value::operator--()
{
  switch (_data->type())
  {
  case ValueData::BOOL:
    _data->set(!_data->get_bool(),_data->size());
    break;

  case ValueData::FLOAT:
    _data->set_float(_data->get_float() - 1.0,_data->size());
    break;

  case ValueData::ULONGLONG:
    _data->set_ull(_data->get_ull() - 1ull,_data->size());
    break;

  case ValueData::PTR:
    _data->set_ptr(reinterpret_cast<void*>(static_cast<std::uintptr_t>(_data->get_ull() - 1ull)),_data->size());
    break;

  default:
    _data->set_int(_data->get_int() - 1,_data->size());
    break;
  }
  return *this;
}

/**
 *
 * @return
 */
Value Value::operator--(int)
{
  Value result = *this;
  switch (_data->type())
  {
  case ValueData::BOOL:
    _data->set(!_data->get_bool(),_data->size());
    break;

  case ValueData::FLOAT:
    _data->set_float(_data->get_float() - 1.0,_data->size());
    break;

  case ValueData::ULONGLONG:
    _data->set_ull(_data->get_ull() - 1ull,_data->size());
    break;

  case ValueData::PTR:
    _data->set_ptr(reinterpret_cast<void*>(static_cast<std::uintptr_t>(_data->get_ull() - 1ull)),_data->size());
    break;

  default:
    _data->set_int(_data->get_int() - 1,_data->size());
    break;
  }
  return result;
}

/**
 *
 * @return
 */
Value Value::operator-()
{
  Value result;
  switch (_data->type())
  {
  case ValueData::BOOL:
    result._data->set_bool(!_data->get_bool(),_data->size());
    break;

  case ValueData::INT:
    result._data->set_int(-_data->get_int(),_data->size());
    break;

  case ValueData::ULONGLONG:
    result._data->set_ull(-_data->get_ull(),_data->size());
    break;

  default:
    result._data->set_float(-_data->get_float(),_data->size());
    break;
  }
  return result;
}

/**
 *
 * @return
 */
Value Value::operator!()
{
  Value result;

  switch (_data->type())
  {
  case ValueData::INT:
    result._data->set_int(!_data->get_int(),_data->size());
    break;

  case ValueData::FLOAT:
    result._data->set_float(!_data->get_float(),_data->size());
    break;

  case ValueData::ULONGLONG:
    result._data->set_ull(!_data->get_ull(),_data->size());
    break;

  default:
    result._data->set_bool(!_data->get_bool(),_data->size());
    break;
  }
  return result;
}

/**
 *
 * @return
 */
Value Value::operator~()
{
  Value result;
  switch (_data->type())
  {
  case ValueData::BOOL:
    result._data->set_bool(~_data->get_bool(),_data->size());
    break;

  case ValueData::ULONGLONG:
    result._data->set_ull(~_data->get_ull(),_data->size());
    break;

  default:
    result._data->set_int(~_data->get_int(),_data->size());
    break;
  }
  return result;
}



/**
 *
 * @param val
 * @return
 */
Value Value::operator+(const Value& val) const
{
  Value result;

  switch (_data->type())
  {
  case ValueData::BOOL:
    result._data->set_bool((_data->get_int() + val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::INT:
    result._data->set_int(_data->get_int() + val._data->get_int(),_data->size());
    break;

  case ValueData::FLOAT:
    result._data->set_float(_data->get_float() + val._data->get_float(),_data->size());
    break;

  case ValueData::ULONGLONG:
    result._data->set_ull(_data->get_ull() + val._data->get_ull(),_data->size());
    break;

  case ValueData::PTR:
    result._data->set_ptr(reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(_data->get_ptr()) + val._data->get_int()),_data->size());
    break;

  case ValueData::STRING:
    result._data->set_string((_data->get_string() + val._data->get_string()).c_str());
    break;

  case ValueData::BYTEARRAY:
    {
      ValueData::byte_buffer new_buf(_data->get_byte_array());
      ValueData::byte_buffer val_buff = val._data->get_byte_array();
      new_buf.insert(new_buf.end(), val_buff.begin(), val_buff.end());
      result._data->set_byte_array(new_buf.data(), new_buf.size());
    }
    break;

  default:
    if (val._data->type() == ValueData::FLOAT)
      result._data->set_float(_data->get_float() + val._data->get_float());
    else
      result._data->set_int(_data->get_int() + val._data->get_int());

    break;
  }
  return result;
}

/**
 *
 * @param val
 * @return
 */
Value Value::operator-(const Value& val) const
{
  Value result;
  switch (_data->type())
  {
  case ValueData::BOOL:
    result._data->set_bool((_data->get_int() - val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::INT:
    result._data->set_int(_data->get_int() - val._data->get_int(),_data->size());
    break;

  case ValueData::FLOAT:
    result._data->set_float(_data->get_float() - val._data->get_float(),_data->size());
    break;

  case ValueData::ULONGLONG:
    result._data->set_ull(_data->get_ull() - val._data->get_ull(),_data->size());
    break;

  case ValueData::PTR:
    result._data->set_ptr(reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(_data->get_ptr()) - val._data->get_int()),_data->size());
    break;

  default:
    if (val._data->type() == ValueData::FLOAT)
      result._data->set_float(_data->get_float() - val._data->get_float(),_data->size());
    else
      result._data->set_int(_data->get_int() - val._data->get_int(),_data->size());
    break;
  }

  return result;
}

/**
 *
 * @param val
 * @return
 */
Value Value::operator/(const Value& val) const
{
  Value result;
  switch (_data->type())
  {
  case ValueData::BOOL:
    result._data->set_bool((_data->get_int() / val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::INT:
    result._data->set_int(_data->get_int() / val._data->get_int(),_data->size());
    break;

  case ValueData::ULONGLONG:
    result._data->set_ull(_data->get_ull() / val._data->get_ull(),_data->size());
    break;

  default:
    result._data->set_float(_data->get_float() / val._data->get_float(),_data->size());
    break;
  }

  return result;
}

/**
 *
 * @param val
 * @return
 */
Value Value::operator%(const Value& val) const
{
  Value result;
  switch (_data->type())
  {
  case ValueData::BOOL:
    result._data->set_bool((_data->get_int() % val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::ULONGLONG:
    result._data->set_ull(_data->get_ull() % val._data->get_ull(),_data->size());
    break;

  default:
    result._data->set_int(_data->get_int() % val._data->get_int(),_data->size());
    break;
  }
  return result;
}


/**
 *
 * @param val
 * @return
 */
Value Value::operator*(const Value& val) const
{
  Value result;
  switch (_data->type())
  {
  case ValueData::BOOL:
    result._data->set_bool((_data->get_int() * val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::INT:
    result._data->set_int(_data->get_int() * val._data->get_int(),_data->size());
    break;

  case ValueData::FLOAT:
    result._data->set_float(_data->get_float() * val._data->get_float(),_data->size());
    break;

  case ValueData::ULONGLONG:
    result._data->set_ull(_data->get_ull() * val._data->get_ull(),_data->size());
    break;

  default:
    if (val._data->type() == ValueData::FLOAT)
      result._data->set_float(_data->get_float() * val._data->get_float(),_data->size());
    else
      result._data->set_int(_data->get_int() * val._data->get_int(),_data->size());
    break;
  }

  return result;
}

/**
 *
 * @param val
 * @return
 */
Value Value::operator|(const Value& val) const
{
  Value result;
  switch (_data->type())
  {
  case ValueData::BOOL:
    result._data->set_bool((_data->get_int() | val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::ULONGLONG:
    result._data->set_ull(_data->get_ull() | val._data->get_ull(),_data->size());
    break;

  default:
    result._data->set_int(_data->get_int() | val._data->get_int(),_data->size());
    break;
  }

  return result;
}

/**
 *
 * @param val
 * @return
 */
Value Value::operator&(const Value& val) const
{
  Value result;
  switch (_data->type())
  {
  case ValueData::BOOL:
    result._data->set_bool((_data->get_int() & val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::ULONGLONG:
    result._data->set_ull(_data->get_ull() & val._data->get_ull(),_data->size());
    break;

  default:
    result._data->set_int(_data->get_int() & val._data->get_int(),_data->size());
    break;
  }

  return result;
}

/**
 *
 * @param val
 * @return
 */
Value Value::operator^(const Value& val) const
{
  Value result;
  switch (_data->type())
  {
  case ValueData::BOOL:
    result._data->set_bool((_data->get_int() ^ val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::ULONGLONG:
    result._data->set_ull(_data->get_ull() ^ val._data->get_ull(),_data->size());
    break;

  default:
    result._data->set_int(_data->get_int() ^ val._data->get_int(),_data->size());
    break;
  }

  return result;
}

/**
 *
 * @param val
 * @return
 */
Value Value::operator>>(const Value& val) const
{
  Value result;
  switch (_data->type())
  {
  case ValueData::BOOL:
    result._data->set_bool((_data->get_int() >> val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::ULONGLONG:
    result._data->set_ull(_data->get_ull() >> val._data->get_ull(),_data->size());
    break;

  default:
    result._data->set_int(_data->get_int() >> val._data->get_int(),_data->size());
    break;
  }

  return result;
}

/**
 *
 * @param val
 * @return
 */
Value Value::operator<<(const Value& val) const
{
  Value result;
  switch (_data->type())
  {
  case ValueData::BOOL:
    result._data->set_bool((_data->get_int() << val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::ULONGLONG:
    result._data->set_ull(_data->get_ull() << val._data->get_ull(),_data->size());
    break;

  default:
    result._data->set_int(_data->get_int() << val._data->get_int(),_data->size());
    break;
  }

  return result;
}


/**
 *
 * @param val
 * @return
 */
Value& Value::operator+=(const Value& val)
{
  switch (_data->type())
  {
  case ValueData::BOOL:
    _data->set_bool((_data->get_int() + val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::INT:
    _data->set_int(_data->get_int() + val._data->get_int(),_data->size());
    break;

  case ValueData::FLOAT:
    _data->set_float(_data->get_float() + val._data->get_float(),_data->size());
    break;

  case ValueData::ULONGLONG:
    _data->set_ull(_data->get_ull() + val._data->get_ull(),_data->size());
    break;

  case ValueData::PTR:
    _data->set_ptr(reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(_data->get_ptr()) + val._data->get_int()),_data->size());
    break;

  case ValueData::STRING:
    _data->set_string((_data->get_string() + val._data->get_string()).c_str());
    break;

  case ValueData::BYTEARRAY:
    {
      ValueData::byte_buffer new_buf(_data->get_byte_array());
      ValueData::byte_buffer val_buff = val._data->get_byte_array();
      new_buf.insert(new_buf.end(), val_buff.begin(), val_buff.end());
      _data->set_byte_array(new_buf.data(), new_buf.size());
    }
    break;

  default:
    break;
  }

  return *this;
}

/**
 *
 * @param val
 * @return
 */
Value& Value::operator-=(const Value& val)
{
  switch (_data->type())
  {
  case ValueData::BOOL:
    _data->set_bool((_data->get_int() - val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::INT:
    _data->set_int(_data->get_int() - val._data->get_int(),_data->size());
    break;

  case ValueData::FLOAT:
    _data->set_float(_data->get_float() - val._data->get_float(),_data->size());
    break;

  case ValueData::ULONGLONG:
    _data->set_ull(_data->get_ull() - val._data->get_ull(),_data->size());
    break;

  case ValueData::PTR:
    _data->set_ptr(reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(_data->get_ptr()) - val._data->get_int()),_data->size());
    break;

  default:
    if (val._data->type() == ValueData::FLOAT)
      _data->set_float(_data->get_float() - val._data->get_float(),_data->size());
    else
      _data->set_int(_data->get_int() - val._data->get_int(),_data->size());
    break;
  }

  return *this;
}

/**
 *
 * @param val
 * @return
 */
Value& Value::operator/=(const Value& val)
{
  switch (_data->type())
  {
  case ValueData::BOOL:
    _data->set_bool((_data->get_int() / val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::INT:
    _data->set_int(_data->get_int() / val._data->get_int(),_data->size());
    break;

  case ValueData::ULONGLONG:
    _data->set_ull(_data->get_ull() / val._data->get_ull(),_data->size());
    break;

  default:
    _data->set_float(_data->get_float() / val._data->get_float(),_data->size());
    break;
  }

  return *this;
}

/**
 *
 * @param val
 * @return
 */
Value& Value::operator%=(const Value& val)
{
  switch (_data->type())
  {
  case ValueData::BOOL:
    _data->set_bool((_data->get_int() % val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::ULONGLONG:
    _data->set_ull(_data->get_ull() % val._data->get_ull(),_data->size());
    break;

  default:
    _data->set_int(_data->get_int() % val._data->get_int(),_data->size());
    break;
  }

  return *this;
}


/**
 *
 * @param val
 * @return
 */
Value& Value::operator*=(const Value& val)
{
  switch (_data->type())
  {
  case ValueData::BOOL:
    _data->set_bool((_data->get_int() * val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::INT:
    _data->set_int(_data->get_int() * val._data->get_int(),_data->size());
    break;

  case ValueData::FLOAT:
    _data->set_float(_data->get_float() * val._data->get_float(),_data->size());
    break;

  case ValueData::ULONGLONG:
    _data->set_ull(_data->get_ull() * val._data->get_ull(),_data->size());
    break;

  default:
    if (val._data->type() == ValueData::FLOAT)
      _data->set_float(_data->get_float() * val._data->get_float(),_data->size());
    else
      _data->set_int(_data->get_int() * val._data->get_int(),_data->size());
    break;
  }

  return *this;
}

/**
 *
 * @param val
 * @return
 */
Value& Value::operator|=(const Value& val)
{
  switch (_data->type())
  {
  case ValueData::BOOL:
    _data->set_bool((_data->get_int() | val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::ULONGLONG:
    _data->set_ull(_data->get_ull() | val._data->get_ull(),_data->size());
    break;

  default:
    _data->set_int(_data->get_int() | val._data->get_int(),_data->size());
    break;
  }

  return *this;
}

/**
 *
 * @param val
 * @return
 */
Value& Value::operator&=(const Value& val)
{
  switch (_data->type())
  {
  case ValueData::BOOL:
    _data->set_bool((_data->get_int() & val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::ULONGLONG:
    _data->set_ull(_data->get_ull() & val._data->get_ull(),_data->size());
    break;

  default:
    _data->set_int(_data->get_int() & val._data->get_int(),_data->size());
    break;
  }

  return *this;
}

/**
 *
 * @param val
 * @return
 */
Value& Value::operator^=(const Value& val)
{
  switch (_data->type())
  {
  case ValueData::BOOL:
    _data->set_bool((_data->get_int() ^ val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::ULONGLONG:
    _data->set_ull(_data->get_ull() ^ val._data->get_ull(),_data->size());
    break;

  default:
    _data->set_int(_data->get_int() ^ val._data->get_int(),_data->size());
    break;
  }

  return *this;
}


/**
 *
 * @param val
 * @return
 */
Value& Value::operator>>=(const Value& val)
{
  switch (_data->type())
  {
  case ValueData::BOOL:
    _data->set_bool((_data->get_int() >> val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::ULONGLONG:
    _data->set_ull(_data->get_ull() >> val._data->get_ull(),_data->size());
    break;

  default:
    _data->set_int(_data->get_int() >> val._data->get_int(),_data->size());
    break;
  }

  return *this;
}

/**
 *
 * @param val
 * @return
 */
Value& Value::operator<<=(const Value& val)
{
  switch (_data->type())
  {
  case ValueData::BOOL:
    _data->set_bool((_data->get_int() << val._data->get_int()) != 0,_data->size());
    break;

  case ValueData::ULONGLONG:
    _data->set_ull(_data->get_ull() << val._data->get_ull(),_data->size());
    break;

  default:
    _data->set_int(_data->get_int() << val._data->get_int(),_data->size());
    break;
  }
  return *this;
}


/*
 * \\fn Value Value Value::operator=
 *
 * created on: Jul 31, 2019
 * author: daniel
 *
 */
Value& Value::operator=(const Value& val)
{
  *_data = *val._data;
  return *this;
}

/**
 *
 * @param val
 * @return
 */
Value Value::operator==(const Value& val) const
{
  Value result;

  if ((_data->type() == ValueData::NIL) && (val._data->type() == ValueData::NIL))
  {
    result._data->set_bool(memcmp(&val._data->at(0), &_data->at(0),
              std::min(val._data->size(), _data->size()) == 0));
  }
  else
  {
    switch (_data->type())
    {
    case ValueData::BOOL:
      result._data->set_bool(_data->get_bool() == val._data->get_bool(),_data->size());
      break;

    case ValueData::INT:
      result._data->set_bool(_data->get_int() == val._data->get_int(),_data->size());
      break;

    case ValueData::FLOAT:
      result._data->set_bool(_data->get_float() == val._data->get_float(),_data->size());
      break;

    case ValueData::ULONGLONG:
      result._data->set_bool(_data->get_ull() == val._data->get_ull(),_data->size());
      break;

    case ValueData::PTR:
      result._data->set_bool(_data->get_ptr() == val._data->get_ptr(),_data->size());
      break;

    case ValueData::STRING:
      result._data->set_bool(_data->get_string() == val._data->get_string());
      break;

    case ValueData::BYTEARRAY:
      if (_data->size() != val._data->size())
        result._data->set_bool(false);
      else
        result._data->set_bool((_data->size() == 0) ? true : (memcmp(_data->get_byte_array().data(), val._data->get_byte_array().data(), _data->size()) == 0));
      break;

    default:
      result._data->set_bool((bool)(val == *this), size());
      break;
    }
  }

  return result;
}

/**
 *
 * @param val
 * @return
 */
Value Value::operator!=(const Value& val) const
{
  Value result;
  if ((_data->type() == ValueData::NIL) && (val._data->type() == ValueData::NIL))
  {
    result._data->set_bool(memcmp(&val._data->at(0), &_data->at(0),
              std::min(val._data->size(), _data->size()) != 0));
  }
  else
  {
    switch (_data->type())
    {
    case ValueData::BOOL:
      result._data->set_bool(_data->get_bool() != val._data->get_bool(),_data->size());
      break;

    case ValueData::INT:
      result._data->set_bool(_data->get_int() != val._data->get_int(),_data->size());
      break;

    case ValueData::FLOAT:
      result._data->set_bool(_data->get_float() != val._data->get_float(),_data->size());
      break;

    case ValueData::ULONGLONG:
      result._data->set_bool(_data->get_ull() != val._data->get_ull(),_data->size());
      break;

    case ValueData::PTR:
      result._data->set_bool(_data->get_ptr() != val._data->get_ptr(),_data->size());
      break;

    case ValueData::STRING:
      result._data->set_bool(_data->get_string() != val._data->get_string());
      break;

    case ValueData::BYTEARRAY:
      if (_data->size() != val._data->size())
        result._data->set_bool(true);
      else
        result._data->set_bool((_data->size() == 0) ? false : (memcmp(_data->get_byte_array().data(), val._data->get_byte_array().data(), _data->size()) != 0));
      break;

    default:
      result._data->set_bool((bool)(val != *this), size());
      break;
    }
  }

  return result;
}

/**
 *
 * @param val
 * @return
 */
Value Value::operator>(const Value& val) const
{
  Value result;

  if ((_data->type() == ValueData::NIL) && (val._data->type() == ValueData::NIL))
  {
    result._data->set_bool(memcmp(&val._data->at(0), &_data->at(0),
              std::min(val._data->size(), _data->size()) > 0));
  }
  else
  {
    switch (_data->type())
    {
    case ValueData::BOOL:
      result._data->set_bool(_data->get_bool() > val._data->get_bool(),_data->size());
      break;

    case ValueData::INT:
      result._data->set_bool(_data->get_int() > val._data->get_int(),_data->size());
      break;

    case ValueData::FLOAT:
      result._data->set_bool(_data->get_float() > val._data->get_float(),_data->size());
      break;

    case ValueData::ULONGLONG:
      result._data->set_bool(_data->get_ull() > val._data->get_ull(),_data->size());
      break;

    case ValueData::PTR:
      result._data->set_bool(_data->get_ptr() > val._data->get_ptr(),_data->size());
      break;

    case ValueData::STRING:
      result._data->set_bool(_data->get_string().compare(val._data->get_string()) > 0);
      break;

    case ValueData::BYTEARRAY:
      if (_data->size() != val._data->size())
        result._data->set_bool(_data->size() > val._data->size());
      else
        result._data->set_bool((_data->size() == 0) ? true : (memcmp(_data->get_byte_array().data(), val._data->get_byte_array().data(), _data->size()) > 0));
      break;

    default:
      result._data->set_bool((bool)(val < *this), size());
      break;
    }
  }
  return result;
}

/**
 *
 * @param val
 * @return
 */
Value Value::operator<(const Value& val) const
{
  Value result;

  if ((_data->type() == ValueData::NIL) && (val._data->type() == ValueData::NIL))
  {
    result._data->set_bool(memcmp(&val._data->at(0), &_data->at(0),
              std::min(val._data->size(), _data->size()) < 0));
  }
  else
  {
    switch (_data->type())
    {
    case ValueData::BOOL:
      result._data->set_bool(_data->get_bool() < val._data->get_bool(),_data->size());
      break;

    case ValueData::INT:
      result._data->set_bool(_data->get_int() < val._data->get_int(),_data->size());
      break;

    case ValueData::FLOAT:
      result._data->set_bool(_data->get_float() < val._data->get_float(),_data->size());
      break;

    case ValueData::ULONGLONG:
      result._data->set_bool(_data->get_ull() < val._data->get_ull(),_data->size());
      break;

    case ValueData::PTR:
      result._data->set_bool(_data->get_ptr() < val._data->get_ptr(),_data->size());
      break;

    case ValueData::STRING:
      result._data->set_bool(_data->get_string().compare(val._data->get_string()) < 0);
      break;

    case ValueData::BYTEARRAY:
      if (_data->size() != val._data->size())
        result._data->set_bool(false);
      else
        result._data->set_bool((_data->size() == 0) ? true : (memcmp(_data->get_byte_array().data(), val._data->get_byte_array().data(), _data->size()) < 0));
      break;

    default:
      result._data->set_bool((bool)(val > *this), size());
      break;
    }
  }
  return result;
}

/**
 *
 * @param val
 * @return
 */
Value Value::operator>=(const Value& val) const
{
  Value result;

  if ((_data->type() == ValueData::NIL) && (val._data->type() == ValueData::NIL))
  {
    result._data->set_bool(memcmp(&val._data->at(0), &_data->at(0),
              std::min(val._data->size(), _data->size()) >= 0));
  }
  else
  {
    switch (_data->type())
    {
    case ValueData::BOOL:
      result._data->set_bool(_data->get_bool() >= val._data->get_bool(),_data->size());
      break;

    case ValueData::INT:
      result._data->set_bool(_data->get_int() >= val._data->get_int(),_data->size());
      break;

    case ValueData::FLOAT:
      result._data->set_bool(_data->get_float() >= val._data->get_float(),_data->size());
      break;

    case ValueData::ULONGLONG:
      result._data->set_bool(_data->get_ull() >= val._data->get_ull(),_data->size());
      break;

    case ValueData::PTR:
      result._data->set_bool(_data->get_ptr() >= val._data->get_ptr(),_data->size());
      break;

    case ValueData::STRING:
      result._data->set_bool(_data->get_string().compare(val._data->get_string()) >= 0);
      break;

    case ValueData::BYTEARRAY:
      if (_data->size() != val._data->size())
        result._data->set_bool(false);
      else
        result._data->set_bool((_data->size() == 0) ? true : (memcmp(_data->get_byte_array().data(), val._data->get_byte_array().data(), _data->size()) >= 0));
      break;

    default:
      result._data->set_bool((bool)(val <= *this), size());
      break;
    }
  }
  return result;
}

/**
 *
 * @param val
 * @return
 */
Value Value::operator<=(const Value& val) const
{
  Value result;

  if ((_data->type() == ValueData::NIL) && (val._data->type() == ValueData::NIL))
  {
    result._data->set_bool(memcmp(&val._data->at(0), &_data->at(0),
              std::min(val._data->size(), _data->size()) <= 0));
  }
  else
  {
    switch (_data->type())
    {
    case ValueData::BOOL:
      result._data->set_bool(_data->get_bool() <= val._data->get_bool(),_data->size());
      break;

    case ValueData::INT:
      result._data->set_bool(_data->get_int() <= val._data->get_int(),_data->size());
      break;

    case ValueData::FLOAT:
      result._data->set_bool(_data->get_float() <= val._data->get_float(),_data->size());
      break;

    case ValueData::ULONGLONG:
      result._data->set_bool(_data->get_ull() <= val._data->get_ull(),_data->size());
      break;

    case ValueData::PTR:
      result._data->set_bool(_data->get_ptr() <= val._data->get_ptr(),_data->size());
      break;

    case ValueData::STRING:
      result._data->set_bool(_data->get_string().compare(val._data->get_string()) <= 0);
      break;

    case ValueData::BYTEARRAY:
      if (_data->size() != val._data->size())
        result._data->set_bool(false);
      else
        result._data->set_bool((_data->size() == 0) ? true : (memcmp(_data->get_byte_array().data(), val._data->get_byte_array().data(), _data->size()) <= 0));
      break;

    default:
      result._data->set_bool((bool)(val >= *this), size());
      break;
    }
  }
  return result;
}

/**
 *
 * @param val
 * @return
 */
Value Value::operator&&(const Value& val) const
{
  Value result;
  switch (_data->type())
  {
  case ValueData::INT:
    result._data->set_bool(_data->get_int() && val._data->get_int(),_data->size());
    break;

  case ValueData::ULONGLONG:
    result._data->set_bool(_data->get_ull() && val._data->get_ull(),_data->size());
    break;

  default:
    result._data->set_bool(_data->get_bool() && val._data->get_bool(),_data->size());
    break;
  }

  return result;
}

/**
 *
 * @param val
 * @return
 */
Value Value::operator||(const Value& val) const
{
  Value result;
  switch (_data->type())
  {
  case ValueData::INT:
    result._data->set_bool(_data->get_int() || val._data->get_int(),_data->size());
    break;

  case ValueData::ULONGLONG:
    result._data->set_bool(_data->get_ull() || val._data->get_ull(),_data->size());
    break;

  default:
    result._data->set_bool(_data->get_bool() || val._data->get_bool(),_data->size());
    break;
  }

  return result;
}

/*
 * \\fn Value& Value::at
 *
 * created on: Jul 2, 2019
 * author: daniel
 *
 */
Value Value::at(size_t index)
{
  return _data->at(index);
}

/*
 * \\fn Value Value::sub_array
 *
 * created on: Jul 31, 2019
 * author: daniel
 *
 */
Value Value::sub_array(int start, int length)
{
  Value result;

  _data->extract(*result._data, start, length);
  return result;
}

} // jupiter
} // brt
