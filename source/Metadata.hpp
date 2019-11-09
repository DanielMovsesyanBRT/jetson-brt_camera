//
// Created by Daniel Movsesyan on 2019-06-19.
//

#ifndef MOTEC_BU_METADATA_HPP
#define MOTEC_BU_METADATA_HPP

#include <unordered_set>
#include <stdint.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <typeinfo>



namespace brt {

namespace jupiter {

class MetaImpl;

/**
 *
 */
class Metadata
{
public:
  typedef std::vector<uint8_t>    byte_buffer;

  /*
   * \\fn Constructor Metadata
   *
   * created on: Jul 30, 2019
   * author: daniel
   *
   */
  Metadata() : _impl(nullptr) 
  { 
    init(); 
  }

  /*
   * \\fn Constructor Metadata
   *
   * created on: Jul 30, 2019
   * author: daniel
   *
   */
  Metadata(const char *semicolon_separated_list) : _impl(nullptr)
  {
    init();
    add(semicolon_separated_list);
  }


  /*
   * \\fn Constructor Metadata
   *
   * created on: Jul 30, 2019
   * author: daniel
   *
   */
  Metadata(const Metadata& params) : _impl(nullptr)
  {
    init();
    copy_metadata(&params);
  }
  virtual ~Metadata();

  /*
   * \\struct default_arg template
   *
   * created on: Jul 30, 2019
   *
   */
  template<typename T>  struct default_arg { static T get() { return static_cast<T>(0); } };

  template<typename T> Metadata&  set(const char* key,T value);
  template<typename T> T          get(const char* key,T default_value = default_arg<T>::get()) const;
  template<typename T> T          get_at(const char* key,size_t index, const T& default_value = default_arg<T>::get());

        bool                      exist(const char *key) const;
        void                      erase(const char *key);
        
        std::vector<std::string>  matching_keys(const char *regex) const;
        

        Metadata&                 copy_metadata(const Metadata *from);
        Metadata&                 add(const char *semicolon_separated_list);

        Metadata&                 operator=(const Metadata&);
        Metadata&                 operator+=(const Metadata&);
        Metadata&                 operator+=(const char *semicolon_separated_list);
        bool                      operator()(const char *key) const;

        std::string               to_string() const;
        std::string               to_json(bool nicely_formatted = true) const;
private:
        void                      init();

protected:
  MetaImpl*                       _impl;
};

template<> struct Metadata::default_arg<void*> { static void* get() { return nullptr; } };
template<> struct Metadata::default_arg<std::string> { static std::string get() { return std::string(); } };

template<> struct Metadata::default_arg<Metadata::byte_buffer>
    { static Metadata::byte_buffer get() { return Metadata::byte_buffer(); } };

} // jupiter
} // brt

extern brt::jupiter::Metadata _context;

#endif //MOTEC_BU_METADATA_HPP
