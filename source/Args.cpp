/*
 * Args.cpp
 *
 *  Created on: May 8, 2019
 *      Author: daniel
 */

#include "Args.hpp"
#include <string.h>

#include <sstream>
#include <algorithm>

namespace brt
{
namespace jupiter
{

Args Args::_obj;

/*
 * \\fn Constructor Args::Args
 *
 * created on: May 8, 2019
 * author: daniel
 *
 */
Args::Args()
: _value_map()
{
}

/*
 * \\fn Destructor Args::~Args
 *
 * created on: May 8, 2019
 * author: daniel
 *
 */
Args::~Args()
{
}

/*
 * \\fn Args& Args::init
 *
 * created on: May 8, 2019
 * author: daniel
 *
 */
Args& Args::init(const char* name,const char* default_value,const char* help,bool mandatory /*= false*/)
{
  _value_map.insert(_map::value_type(name,Value(default_value,help,mandatory)));
  return *this;
}

/*
 * \\fn Args& Args::init
 *
 * created on: May 8, 2019
 * author: daniel
 *
 */
Args& Args::init(const char* name,const char* help,bool mandatory /*= false*/)
{
  _value_map.insert(_map::value_type(name,Value(nullptr,help,mandatory)));
  return *this;
}
/*
 * \\fn bool Args& Args::init
 *
 * created on: May 8, 2019
 * author: daniel
 *
 */
bool Args::parse(int argc,const char **argv)
{
  for (int index = 1; index < argc; index++)
  {
    if (strstr(argv[index],"--") == argv[index])
    {
      const char *eq = strstr(argv[index],"=");
      std::string value;
      std::string name;
      if (eq == nullptr)
      {
        name = argv[index];
        value = "1";
      }
      else
      {
        name = std::string (argv[index],eq - argv[index]);
        value = (eq + 1);
      }
      _value_map.insert(_map::value_type(name,Value(value.c_str())));
    }

    else if (strstr(argv[index],"-") == argv[index])
    {
      _value_map.insert(_map::value_type(argv[index],Value("1")));
    }
    else
    {
      _value_map.insert(_map::value_type(argv[index],Value("true")));
    }
  }

  std::map<std::string,Value>::const_iterator iter = _value_map.begin();
  while (iter != _value_map.end())
  {
    if (iter->second._mandatory && iter->second._value.empty())
      return false;

    iter++;
  }
  return true;
}

/*
 * \\fn std::string Args::value
 *
 * created on: May 8, 2019
 * author: daniel
 *
 */
std::string Args::value(const char* name) const
{
  std::map<std::string,Value>::const_iterator iter = _value_map.find(name);
  if (iter == _value_map.end())
    return std::string();

  return iter->second._value;
}

/*
 * \\fn std::vector<std::string> Args::values
 *
 * created on: Jun 24, 2019
 * author: daniel
 *
 */
std::vector<std::string> Args::values(const char* name) const
{
  std::vector<std::string> result;
  auto range = _value_map.equal_range(name);
  std::for_each(range.first, range.second, [&](const _map::value_type& val)
  {
    result.push_back(val.second._value);
  });

  return result;
}

/*
 * \\fn bool Args::flag
 *
 * created on: May 8, 2019
 * author: daniel
 *
 */
size_t Args::exist(const char* name) const
{
  return _value_map.count(name);
}

/*
 * \\fn std::string Args::help
 *
 * created on: May 8, 2019
 * author: daniel
 *
 */
std::string Args::help() const
{
  std::ostringstream  out;
  out << "Arguments are:" << std::endl;

  std::map<std::string,Value>::const_iterator iter = _value_map.begin();
  while (iter != _value_map.end())
  {
    if (!iter->second._description.empty())
    {
      out << "\t " << iter->first << " - " << iter->second._description << (iter->second._mandatory?" [MANDATORY] ":"") << std::endl;
    }
    iter++;
  }

  return out.str();
}

/*
 * \\fn Metadata Args::get_as_metadata
 *
 * created on: May 17, 2019
 * author: daniel
 *
 */
Metadata Args::get_as_metadata() const
{
  Metadata result;
  for (auto values : _value_map)
  {
    std::string key = values.first;
    while (key.size() > 0 && key[0]=='-')
      key.erase(key.begin());

    result.set(key.c_str(), values.second._value.c_str());
  }

  return result;
}

} /* namespace jupiter */
} /* namespace brt */
