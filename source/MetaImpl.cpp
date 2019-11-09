/*
 * Meta.cpp
 *
 *  Created on: Sep 6, 2019
 *      Author: dmovsesyan
 */

#include "MetaImpl.hpp"
#include "MetaImpl.hpp"

#include <string.h>
#include <sstream>

namespace brt
{
namespace jupiter
{


/*
 * \\fn void MetaImpl::add
 *
 * created on: May 17, 2019
 * author: daniel
 *
 */
void MetaImpl::add(const char *semicolon_separated_list)
{
  if (semicolon_separated_list != nullptr)
  {
    std::vector<std::string>    sub_strings;

    const char *cur_ptr = semicolon_separated_list;
    while ((cur_ptr != nullptr) && (*cur_ptr != '\0'))
    {
      const char *semi_colon = strchr(cur_ptr,';');
      if (semi_colon != nullptr)
      {
        sub_strings.push_back(std::string(cur_ptr,semi_colon - cur_ptr));
        cur_ptr = semi_colon + 1;
      }
      else
      {
        sub_strings.push_back(std::string(cur_ptr));
        cur_ptr = nullptr;
      }
    }

    for(std::string param : sub_strings)
    {
      std::string key,value;
      size_t equal = param.find('=');
      if (equal != std::string::npos)
      {
        set(param.substr(0, equal).c_str(),param.substr(equal + 1).c_str());
      }
      else
      {
        set(param.c_str(),true);
      }
    }
  }
}

/*
 * \\fn void MetaImpl::add
 *
 * created on: May 17, 2019
 * author: daniel
 *
 */
void MetaImpl::add(const MetaImpl* params)
{
  if ((_meta_data != nullptr) && (params->_meta_data != nullptr))
  {
    for (auto pair : (*params->_meta_data))
      _meta_data->insert(pair);
  }
}


/*
 * \\fn std::string std::string MetaImpl::to_string
 *
 * created on: May 17, 2019
 * author: daniel
 *
 */
std::string MetaImpl::to_string() const
{
  std::stringstream result;
  if (_meta_data != nullptr)
  {
    for (auto pair : (*_meta_data))
      result << pair.first << "=" << pair.second.get_string() << ";";
  }

  return result.str();
}

/*
 * \\fn std::string MetaImpl::to_json
 *
 * created on: Aug 12, 2019
 * author: daniel
 *
 */
std::string MetaImpl::to_json(bool nicely_formatted /*= true*/) const
{
  std::stringstream result;
  result << "{";

  bool first_arg = true;
  for (auto pair : (*_meta_data))
  {
    if (!first_arg)
      result << ",";

    if (nicely_formatted)
      result << std::endl << "\t";

    result << "\"" << pair.first << "\": " << pair.second.get_string();
    first_arg = false;
  }

  if (nicely_formatted)
    result << std::endl;
  result << "}";

  return result.str();
}

} /* namespace jupiter */
} /* namespace brt */
