//
// Created by Daniel Movsesyan on 2019-04-19.
//

#include <cstdio>
#include "Utils.hpp"

#undef _lengthof
#define _lengthof(x)        (sizeof(x)/sizeof(x[0]))

namespace brt {

namespace jupiter {


/**
 *
 * @param src
 * @param dst
 * @param len
 * @return
 */
const char* Utils::stristr(const char* src,const char* dst,size_t len /*= (size_t)-1*/)
{
  const char* _dst = dst;
  while (*src && *_dst && ( (len != (size_t)-1) ? len-- : true))
  {
    if (tolower(*_dst) == tolower(*src))
    {
      if (*(++_dst) == '\0')
        return src - (_dst - dst - 1);
    }
    else
      _dst = dst; // reset Destination pointer

    src++;
  }

  return nullptr;
}


/**
 *
 * @param src
 * @param dst
 * @param len
 * @return
 */
size_t Utils::stristr(const std::string& src,const char* dst,size_t len /*= (size_t)-1*/)
{
  const char *result = stristr(src.c_str(),dst,len);
  if (result == nullptr)
    return std::string::npos;

  return 0;
}

/*
 * \\fn double Utils::frame_rate
 *
 * created on: Sep 17, 2019
 * author: daniel
 *
 */
double Utils::frame_rate(const char* fr_string)
{
  char *end;
  double frame_rate = strtod(fr_string, &end), multiplier = 1.0;

  if (*end != '\0')
  {
    if (stristr(end, "fps") != nullptr)
      multiplier = 1.0;
    else if (stristr(end, "fpms") != nullptr)
      multiplier = 1e3;
    else if (stristr(end, "fpm") != nullptr)
      multiplier = 1.0 / 60.0;
    else if (stristr(end, "fpu") != nullptr)
      multiplier = 1e6;
  }

  return frame_rate = 1.0 / (frame_rate * multiplier);
}

} // jupiter
} // brt
