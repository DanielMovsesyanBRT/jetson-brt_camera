/*
 * dynamic_library.hpp
 *
 *  Created on: Mar 13, 2020
 *      Author: daniel
 */

#ifndef BRT_COMMON_DYNAMIC_LIBRARY_HPP_
#define BRT_COMMON_DYNAMIC_LIBRARY_HPP_

#include <string>
#include <dlfcn.h>
#include <unordered_map>
#include <functional>
#include <iostream>

namespace brt
{
namespace jupiter
{

/*
 * \\class DynamicLibrary
 *
 * created on: Mar 13, 2020
 *
 */
template<typename T>
class DynamicLibrary
{
protected:
  DynamicLibrary(const char* libname)
  : _lib_name(libname)
  {
    _handle = dlopen(_lib_name.c_str(), RTLD_LAZY);
  }

  virtual ~DynamicLibrary()
  {
    if (_handle != nullptr)
      dlclose(_handle);
  }

public:
  /*
   * \\fn T& get
   *
   * created on: Mar 13, 2020
   * author: daniel
   *
   */

  static  T&                      get()
  {
    static T    obj;
    return obj;
  }

  /*
   * \\fn operator bool
   *
   * created on: Mar 16, 2020
   * author: daniel
   *
   */
  operator bool()
  {
    return (_handle != nullptr);
  }

  /*
   * \\fn void* find_function
   *
   * created on: Mar 13, 2020
   * author: daniel
   *
   */
  void*                           find_function(const char* f_name)
  {
    if (_handle == nullptr)
      return nullptr;

    auto iter = _functions.find(f_name);
    if (iter != _functions.end())
      return iter->second;

    void* fn = dlsym(_handle, f_name);
    if (fn == nullptr)
    {
      std::cerr << "Cannot find function:" << f_name << "in " << _lib_name << std::endl;
      return nullptr;
    }

    _functions[f_name] = fn;
    return fn;
  }

  /*
   * \\fn Ret call
   *
   * created on: Mar 13, 2020
   * author: daniel
   *
   */

  template<typename Ret, typename... D>
  Ret call(const char *f_name,D... args)
  {
    void* fn_void = find_function(f_name);
    if (fn_void == nullptr)
      return (Ret)0;

    std::function<Ret(D...)> fn = (Ret(*)(D...))fn_void;
    return fn(args...);
  }


private:
  std::string                     _lib_name;
  void*                           _handle;
  std::unordered_map<std::string, void*>
                                  _functions;
};




} /* namespace jupiter */
} /* namespace brt */

#endif /* BRT_COMMON_DYNAMIC_LIBRARY_HPP_ */
