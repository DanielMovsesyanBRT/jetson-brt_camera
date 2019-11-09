//
// Created by Daniel Movsesyan on 2019-06-14.
//

#ifndef MOTEC_BU_PARSERSTRING_HPP
#define MOTEC_BU_PARSERSTRING_HPP

#include <cstdio>
#include <string>
#include <memory>

namespace brt {

namespace jupiter {

namespace script {


/*
 * \\class ParserException
 *
 * created on: Jul 9, 2019
 *
 */
class ParserException
{
private:
  ParserException(const char *text)
  : _text(text)
  {  }

  virtual ~ParserException() {}

public:
  static  ParserException*        create(const char *text)
  {
    return new ParserException(text);
  }

          void                    release() { delete this; }
          const char*             text() const { return _text.c_str(); }

private:
  std::string                     _text;
};


/*
 * \\class ParserString
 *
 * created on: Jul 9, 2019
 *
 */
class ParserString
{
public:
  ParserString(char* buffer);
  ParserString(const char* buffer);
  virtual ~ParserString();

          char*                   detach();
          const char*             next_token(const char *delimeters);
          bool                    eos() const { return _eos; }

          operator const char*()  { return &_cur_line[_offset]; }
          const char*             c_str() const { return &_cur_line[_offset]; }
          const char&             operator[](size_t index) { return _cur_line[_offset + index]; }

          const char*             operator++() { _offset++; return c_str(); }
          const char*             operator++(int) { const char* res = c_str(); _offset++; return res; }

          const char*             operator--() { _offset--; return c_str(); }
          const char*             operator--(int) { const char* res = c_str(); _offset--; return res; }

          size_t                  find(char);
          size_t                  find(const char *);

          size_t                  rfind(char);
          size_t                  rfind(const char *);

          void                    trim_l(size_t len);
          void                    trim_r(size_t pos);

          void                    trim_l(const char* caracters = nullptr);
          void                    trim_r(const char* caracters = nullptr);

          size_t                  line_num() const { return _line_counter; }


  static const size_t             npos = (size_t)-1;

private:
  char*                           _buffer;
  char*                           _saveptr;

  char*                           _cur_line;
  size_t                          _offset;

  bool                            _eos; // end of string
  size_t                          _line_counter;

  bool                            _own_buffer;
};

template<typename ... Args>
static std::string              string_format( const std::string& format, Args ... arguments )
{
  size_t size = snprintf( nullptr, 0, format.c_str(), arguments ... ) + 1; // Extra space for '\0'
  std::unique_ptr<char[]> buf( new char[ size ] );
  snprintf( buf.get(), size, format.c_str(), arguments ... );
  return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}


} // script
} // jupiter
} // brt

#endif //MOTEC_BU_PARSERSTRING_HPP
