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
#include <regex>

#include <assert.h>

#include "value_data.hpp"

namespace brt
{
namespace jupiter
{

class Metadata;

typedef std::unordered_map<std::string,ValueData*> value_db;

/**
 * \class value_database
 *
 * \brief <description goes here>
 */
class value_database
{
public:
  /**
   * \fn  constructor value_database
   *
   * \brief <description goes here>
   */
  value_database()
  {  }

  /**
   * \fn  destructor value_database
   *
   * \brief <description goes here>
   */
  virtual ~value_database()
  {
    for (auto val : _db)
    {
      if (val.second != nullptr)
        val.second->release();
    }
  }


  /**
   * \fn  value
   *
   * @param  key : const char* 
   * @return  ValueData*
   * \brief <description goes here>
   */
  ValueData*  value(const char* key)
  {
    ValueData* result = nullptr;
    auto iter = _db.find(key);
    if (iter != _db.end())
      result = iter->second;
    else
    {
      auto pr = _db.insert({key, new ValueData});
      result = pr.first->second;
    }
    assert(result != nullptr);
    return result;
  }

  /**
   * \fn  find
   *
   * @param  key : const char* 
   * @return  ValueData*
   * \brief <description goes here>
   */
  ValueData*  find(const char* key)
  {
    auto iter = _db.find(key);
    if (iter != _db.end())
      return iter->second;
      
    return nullptr;
  }

  /**
   * \fn  matching_keys
   *
   * @param  *regex : const char 
   * @return  std::vector<std::string
   * \brief <description goes here>
   */
  std::vector<std::string>  matching_keys(const char *regex) const
  {
    const std::regex re(regex);
    std::vector<std::string> result;

    for (auto pair : _db)
    {
      if (std::regex_match(pair.first, re))
        result.push_back(pair.first);
    }
    return result;
  }

  /**
   * \fn  erase
   *
   * @param  *key : const char 
   * \brief <description goes here>
   */
  void  erase(const char *key)
  {
    auto iter = _db.find(key);
    if (iter != _db.end())
    {
      assert(iter->second != nullptr);
      iter->second->release();
      _db.erase(iter);
    }
  }

  /**
   * \fn  add
   *
   * @param  other : const value_database* 
   * \brief <description goes here>
   */
  void add(const value_database* other)
  {
      for (auto pair : other->_db)
        _db.insert({pair.first, new ValueData(*pair.second)});
  }


  /**
   * \fn  operator
   *
   * @param  rvalue : const value_database& 
   * @return  value_database
   * \brief <description goes here>
   */
  value_database& operator=(const value_database& rvalue)
  {
    for (auto val : _db)
    {
      if (val.second != nullptr)
        val.second->release();
    }
    _db.clear();

    for (auto val : rvalue._db)
    {
      if (val.second != nullptr)
      {
        val.second->add_ref();
        _db.insert({val.first, val.second});
      }
    }

    return *this;
  }


  value_db::iterator begin() { return _db.begin(); }
  value_db::const_iterator begin() const { return _db.begin(); }
  value_db::iterator end() { return _db.end(); }
  value_db::const_iterator end() const { return _db.end(); }
  

private:
  value_db                        _db;
};

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
    _meta_data = new value_database;
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
    _meta_data = new value_database;
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
    if (_own_data && (_meta_data != nullptr))
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
      _meta_data->value(key)->set<T>(value);
  }

  /*
   * \\fn Metadata& add
   *
   * created on: Jul 30, 2019
   * author: daniel
   *
   */
  template<typename T>
  void                      add(const char* key,T value)
  {
    if (_meta_data != nullptr)
    {
      ValueData* vl = _meta_data->find(key);
      if (vl == nullptr)
        _meta_data->value(key)->set<T>(value);
      else
        vl->at(vl->length())->set<T>(value);
    }
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

    ValueData* vl = _meta_data->find(key);
    if (vl == nullptr)
      return default_value;

    return vl->get<T>();
  }

  /*
   * \\fn ValueData value
   *
   * created on: Jul 30, 2019
   * author: daniel
   *
   */
  ValueData*          value(const char* key)
  {
    return _meta_data->value(key);
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
    return _meta_data->matching_keys(regex);
  }

        void                      parse(int argc,char** argv,const char* default_arg_name);
        bool                      exist(const char *key) const { return (_meta_data != nullptr) ? (_meta_data->find(key) != nullptr) : false; }
        void                      erase(const char *key) { if (_meta_data != nullptr) _meta_data->erase(key); }

        void                      copy_key(const char* to, const char* from, const MetaImpl* meta);
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
 value_database*                  _meta_data;
 bool                             _own_data;
};

} /* namespace jupiter */
} /* namespace brt */

#endif /* UTILS_METAIMPL_HPP_ */
