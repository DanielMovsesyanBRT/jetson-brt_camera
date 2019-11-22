//
// Created by Daniel Movsesyan on 2019-04-19.
//

#ifndef MOTEC_BU_UTILS_HPP
#define MOTEC_BU_UTILS_HPP


#include <string>
#include <memory>
#include <atomic>
#include <vector>

#include <stdlib.h>
#include <stdint.h>

#define DEFAULT_FIFO_SIZE                   (32)

namespace brt {

namespace jupiter {

/**
 *
 */
template<typename T,int S = DEFAULT_FIFO_SIZE>
class FIFO
{
public:
  FIFO()
  : _write(0)
  , _read(0)
  {
    for (size_t index = 0;index < S;index++)
      _elem[index] = T(0);
  }

  virtual ~FIFO() {}

  /**
   *
   * @param elem
   */
  bool                            push(T elem)
  {
    if (available() == 0)
      return false;

    uint32_t wr = _write.load();
    _elem[wr] = elem;

    if (++wr >= S)
      wr = 0;

    _write.store(wr);
    return true;
  }

  /**
   *
   * @param elem
   * @return
   */
  bool                            pop(T& elem)
  {
    uint32_t wr = _write.load();
    uint32_t rd = _read.load();

    if (rd == wr)
      return false;

    elem = _elem[rd];
    if (++rd >= S)
      rd = 0;

    _read.store(rd);
    return true;
  }

  /**
   *
   * @return
   */
  uint32_t                        available() const
  {
    uint32_t wr = _write.load();
    uint32_t rd = _read.load();

    if (rd > wr)
      return rd - wr - 1;

    return S - wr + rd - 1;
  }

  /**
   *
   * @return
   */
  uint32_t                        num_elements() const
  {
    uint32_t wr = _write.load();
    uint32_t rd = _read.load();

    if (rd > wr)
      return S - rd + wr;

    return wr - rd;
  }

private:
  std::atomic_uint_fast32_t       _write;
  std::atomic_uint_fast32_t       _read;
  T                               _elem[S];
};


/*
 * \\enum DisplayType
 *
 * created on: Nov 19, 2019
 *
 */
enum DisplayType
{
  eLocalDisplays = 1,
  eRemoteDisplay = 2,
  eAllDisplays = eLocalDisplays | eRemoteDisplay
};

/**
 *
 */
class Utils
{
public:
  Utils() {}
  virtual ~Utils() {}

  template<typename ... Args>
  static std::string              string_format( const std::string& format, Args ... arguments)
  {
    size_t size = snprintf( nullptr, 0, format.c_str(), arguments ... ) + 1; // Extra space for '\0'
    std::unique_ptr<char[]> buf( new char[ size ] );
    snprintf( buf.get(), size, format.c_str(), arguments ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
  }

  static const char*              stristr(const char* src,const char* dst,size_t len = (size_t)-1);
  static size_t                   stristr(const std::string& src,const char* dst,size_t len = (size_t)-1);
  static double                   frame_rate(const char* fr_string);

  static  std::vector<std::string>
                                  enumerate_displays(DisplayType = eAllDisplays);

};

} // jupiter
} // brt


#endif //MOTEC_BU_UTILS_HPP
