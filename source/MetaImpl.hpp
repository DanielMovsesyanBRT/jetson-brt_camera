/*
 * Meta.hpp
 *
 *  Created on: Sep 6, 2019
 *      Author: dmovsesyan
 */

#ifndef UTILS_METAIMPL_HPP_
#define UTILS_METAIMPL_HPP_

#include <unordered_set>
#include <stdint.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <typeinfo>

#include <ValueData.hpp>
#include <regex>

namespace brt
{
namespace jupiter
{

class Metadata;

class MetaImpl
{
public:
  /*
   * \\fn Constructor MetaImpl
   *
   * created on: Jul 30, 2019
   * author: daniel
   *
   */
  MetaImpl()
  : _meta_data(nullptr)
  , _own_data(false)
  {
    _meta_data = new script::value_database;
    _own_data = true;
  }

  /*
   * \\fn Constructor MetaImpl
   *
   * created on: Jul 30, 2019
   * author: daniel
   *
   */
  MetaImpl(const char *semicolon_separated_list)
  : _meta_data(nullptr)
  , _own_data(false)
  {
    _meta_data = new script::value_database;
    _own_data = true;

    add(semicolon_separated_list);
  }


  /*
   * \\fn destructor ~MetaImpl
   *
   * created on: Jul 30, 2019
   * author: daniel
   *
   */
  virtual ~MetaImpl()
  {
    if (_own_data && (_meta_data == nullptr))
      delete _meta_data;
  }

  /*
   * \\fn Metadata& set
   *
   * created on: Jul 30, 2019
   * author: daniel
   *
   */
  template<typename T>
  void                      set(const char* key,T value)
  {
    if (_meta_data != nullptr)
      (*_meta_data)[key].set<T>(value);
  }

  /*
   * \\fn T get
   *
   * created on: Jul 30, 2019
   * author: daniel
   *
   */

  template<typename T> T    get(const char* key,T default_value) const
  {
    if (_meta_data == nullptr)
      return default_value;

    script::value_database::const_iterator iter = _meta_data->find(key);
    if (iter == _meta_data->end())
      return default_value;

    return iter->second.get<T>();
  }

  /*
   * \\fn ValueData value
   *
   * created on: Jul 30, 2019
   * author: daniel
   *
   */
  script::ValueData&          value(const char* key)
  {
    return (*_meta_data)[key];
  }

  /*
   * \\fn std::vector<std::string>  matching_keys(const char *regex)
   *
   * created on: Sep 12, 2019
   * author: daniel
   *
   */
  std::vector<std::string>  matching_keys(const char *regex) const
  {
    const std::regex re(regex);
    std::vector<std::string> result;

    for (auto pair : (*_meta_data))
    {
      if (std::regex_match(pair.first, re))
        result.push_back(pair.first);
    }
    return result;
  }

        bool                      exist(const char *key) const { return (_meta_data != nullptr) ? (_meta_data->find(key) != _meta_data->end()) : false; }
        void                      erase(const char *key) { if (_meta_data != nullptr) _meta_data->erase(key); }

        void                      copy_metadata(const MetaImpl* from)
        {
          if (_meta_data != nullptr)
            *(_meta_data) = (*from->_meta_data);
        }

        void                      add(const char *semicolon_separated_list);
        void                      add(const MetaImpl*);
        std::string               to_string() const;
        std::string               to_json(bool nicely_formatted = true) const;

private:
 script::value_database*          _meta_data;
 bool                             _own_data;
};

} /* namespace jupiter */
} /* namespace brt */

#endif /* UTILS_METAIMPL_HPP_ */
