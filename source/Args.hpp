/*
 * Args.hpp
 *
 *  Created on: May 8, 2019
 *      Author: daniel
 */

#ifndef JUPITER_IPP_INCLUDE_ARGS_HPP_
#define JUPITER_IPP_INCLUDE_ARGS_HPP_

#include <map>
#include <set>
#include <vector>

#include "Metadata.hpp"


namespace brt
{
namespace jupiter
{



/*
 * \\class Args
 *
 * created on: May 8, 2019
 *
 */
class Args
{
  Args();
  virtual ~Args();

public:
  static Args&                    get() { return _obj; }
  Args&                           init(const char* name,const char* default_value,const char* help,bool mandatory = false);
  Args&                           init(const char* name,const char* help,bool mandatory = false);

  bool                            parse(int argc,const char **argv);

  std::string                     value(const char* name) const;
  std::vector<std::string>        values(const char* name) const;

  size_t                          exist(const char* name) const;

  std::string                     help() const;

  Metadata                        get_as_metadata() const;

private:
  static  Args                    _obj;

  struct Value
  {
    Value(const char* value = nullptr, const char* description = nullptr,bool mandatory = false)
    : _value(value != nullptr ? value:"")
    , _description(description != nullptr ? description:"")
    , _mandatory(mandatory)
    { }

    std::string                     _value;
    std::string                     _description;
    bool                            _mandatory;
  };

  typedef std::multimap<std::string,Value>   _map;
  _map                              _value_map;
};

} /* namespace jupiter */
} /* namespace brt */

using arguments = brt::jupiter::Args;

#endif /* JUPITER_IPP_INCLUDE_ARGS_HPP_ */
