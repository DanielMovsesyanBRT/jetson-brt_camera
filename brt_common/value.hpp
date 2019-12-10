//
// Created by Daniel Movsesyan on 2019-04-06.
//

#ifndef EXPRESSIONS_VALUE_HPP
#define EXPRESSIONS_VALUE_HPP

#include <string>
#include <vector>
#include <cassert>

#include <value_data.hpp>

namespace brt {

namespace jupiter {

/**
 *
 */
class Value
{
public:
  Value()
  : _data(new ValueData)
  {   }

  Value(const Value& val)
  : _data(val._data)
  {
    assert(_data != nullptr);
    _data->add_ref();
  }

  Value(ValueData& vd)
  : _data(&vd)
  {
    _data->add_ref();
  }

  virtual ~Value()
  {
    if (_data != nullptr)
      _data->release();
  }

  operator bool() const { return _data->get_bool(); }
  operator int() const  { return _data->get_int(); }
  operator double() const { return _data->get_float(); }
  operator std::string() const { return _data->get_string(); }
  operator ValueData::byte_buffer() const { return _data->get_byte_array(); }
  operator ValueData&() { return *_data; }


          /*
           * \\fn set
           *
           * created on: Jul 30, 2019
           * author: daniel
           *
           */
          template<typename T>
          Value&                 set(T value,size_t size = (size_t)-1)
          {
            assert(_data != nullptr);
            if (size == (size_t)-1)
              _data->set<T>(value);
            else
              _data->set<T>(value,size);

            return *this;
          }

          /*
           *
           */
          Value&                  set_byte_array(const uint8_t* value,size_t size,bool little_endian = true)
          {
            assert(_data != nullptr);
            _data->set_byte_array(value,size,little_endian);

            return *this;
          }


          size_t                  size() const { return _data->size(); }

          // Unary Operators
          Value&                  operator++();
          Value                   operator++(int);
          Value&                  operator--();
          Value                   operator--(int);
          Value                   operator-();
          Value                   operator!();
          Value                   operator~();

          // Arithmetic
          Value                   operator+(const Value& val) const;
          Value                   operator-(const Value& val) const;
          Value                   operator/(const Value& val) const;
          Value                   operator%(const Value& val) const;
          Value                   operator*(const Value& val) const;
          Value                   operator|(const Value& val) const;
          Value                   operator&(const Value& val) const;
          Value                   operator^(const Value& val) const;
          Value                   operator>>(const Value& val) const;
          Value                   operator<<(const Value& val) const;

          Value&                  operator+=(const Value& val);
          Value&                  operator-=(const Value& val);
          Value&                  operator/=(const Value& val);
          Value&                  operator%=(const Value& val);
          Value&                  operator*=(const Value& val);
          Value&                  operator|=(const Value& val);
          Value&                  operator&=(const Value& val);
          Value&                  operator^=(const Value& val);
          Value&                  operator>>=(const Value& val);
          Value&                  operator<<=(const Value& val);
          Value&                  operator=(const Value& val);

          // Logic
          Value                   operator==(const Value& val) const;
          Value                   operator!=(const Value& val) const;
          Value                   operator>(const Value& val) const;
          Value                   operator<(const Value& val) const;
          Value                   operator>=(const Value& val) const;
          Value                   operator<=(const Value& val) const;
          Value                   operator&&(const Value& val) const;
          Value                   operator||(const Value& val) const;

          Value                   at(size_t index);
          Value                   sub_array(int start, int length);

private:
  ValueData*                      _data;
};

} // jupiter
} // brt

#endif //EXPRESSIONS_VALUE_HPP
