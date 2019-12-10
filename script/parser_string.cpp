//
// Created by Daniel Movsesyan on 2019-06-14.
//

#include "parser_string.hpp"

#include <cctype>
#include <string.h>

namespace brt {

namespace jupiter {

namespace script {

/**
 *
 */
ParserString::ParserString(char* buffer)
: _buffer(buffer)
, _saveptr(nullptr)
, _cur_line(nullptr)
, _offset(0)
, _eos(false)
, _line_counter(0)
, _own_buffer(false)
{
}

/**
 *
 * @param buffer
 */
ParserString::ParserString(const char* buffer)
: _buffer(nullptr)
, _saveptr(nullptr)
, _cur_line(nullptr)
, _offset(0)
, _eos(false)
, _line_counter(0)
, _own_buffer(true)
{
  if (buffer != nullptr)
  {
    _buffer = new char [strlen(buffer) + 1];
    strcpy(_buffer,buffer);
  }
}

/**
 *
 */
ParserString::~ParserString()
{
  if (_own_buffer && (_buffer != nullptr))
    delete[] _buffer;
}

/**
 *
 * @return
 */
char* ParserString::detach()
{
  char* result = _buffer;
  _buffer = nullptr;
  return result;
}


/**
 *
 * @param tok_array
 * @return
 */
const char* ParserString::next_token(const char *delimeters)
{
  if (_eos)
    return nullptr;

  _cur_line = nullptr;
  _offset = 0;

  size_t del_pos;
  if (_saveptr == nullptr)
  {
    // Counting Empty lines as well
    while ((*_buffer != '\0') && ((del_pos = strspn(_buffer, delimeters)) > 0))
    {
      _line_counter += del_pos;
      _buffer += del_pos;
    }

    _cur_line = strtok_r(_buffer,delimeters,&_saveptr);
  }
  else
  {
    // Counting Empty lines as well
    while ((*_saveptr != '\0') && ((del_pos = strspn(_saveptr, delimeters)) > 0))
    {
      _line_counter += del_pos;
      _saveptr += del_pos;
    }

    _cur_line = strtok_r(nullptr,delimeters,&_saveptr);
  }

  if (_cur_line == nullptr)
    _eos = true;
  else
    _line_counter++;

  return _cur_line;
}

/**
 *
 * @return
 */
size_t ParserString::find(char character)
{
  if (_cur_line == nullptr)
    return npos;

  const char* where = strchr(c_str(), character);
  if (where == nullptr)
    return npos;

  return (where - c_str());
}

/**
 *
 * @param str
 * @return
 */
size_t ParserString::find(const char *str)
{
  if ((_cur_line == nullptr) || (str == nullptr))
    return npos;

  const char* where = strstr(c_str(), str);
  if (where == nullptr)
    return npos;

  return (where - c_str());
}

/**
 *
 * @return
 */
size_t ParserString::rfind(char character)
{
  if (_cur_line == nullptr)
    return npos;

  const char *last_char = c_str();
  while (*(last_char + 1) != '\0')
    last_char++;

  while (*last_char != character)
  {
    if (last_char == c_str())
      return npos;

    last_char--;
  }

  return (*last_char == character) ? (last_char - c_str()) : npos;
}

/**
 *
 * @return
 */
size_t ParserString::rfind(const char *str)
{
  if ((_cur_line == nullptr) || (str == nullptr))
    return npos;

  // Remember last position of argument string
  const char* last_str = str;
  while (*(last_str + 1) != '\0')
    last_str++;
  const char* cur_ptr = last_str;

  // Remember last position of source string
  const char *last_char = c_str();
  while (*(last_char + 1) != '\0')
    last_char++;

  while (last_char >= c_str())
  {
    if (*cur_ptr == *last_char)
    {
      if (cur_ptr == str)
        break;
      cur_ptr--;
    }
    else
      cur_ptr = last_str;

    last_char--;
  }

  return (cur_ptr == str) ? (last_char - c_str()) : npos;
}



/**
 *
 * @param pos
 * @param len
 */
void ParserString::trim_l(size_t len)
{
  if (_cur_line == nullptr)
    return;

  size_t full_length = strlen(c_str());
  if (len > full_length)
    _offset += full_length;

  else
    _offset += len;
}


/**
 *
 * @param pos
 */
void ParserString::trim_r(size_t pos)
{
  if (_cur_line == nullptr)
    return;

  size_t full_length = strlen(c_str());
  if (pos < full_length)
    _cur_line[_offset + pos] = '\0';
}

/**
 *
 * @param characters
 */
void ParserString::trim_l(const char* characters /*= nullptr*/)
{
  if (_cur_line == nullptr)
    return;

  if (characters == nullptr) // default spaces
  {
    while (isspace(_cur_line[_offset]) && (_cur_line[_offset] != '\0'))
      _offset++;
  }
  else
  {
    while ((strchr(characters,_cur_line[_offset]) != nullptr) && (_cur_line[_offset] != '\0'))
      _offset++;
  }
}

/**
 *
 * @param characters
 */
void ParserString::trim_r(const char* characters /*= nullptr*/)
{
  if (_cur_line == nullptr)
    return;

  char *last_char = &_cur_line[_offset];
  while (*(last_char + 1) != '\0')
    last_char++;

  if (characters == nullptr) // default spaces
  {
    while (isspace(*last_char) && (last_char > &_cur_line[_offset]))
      last_char--;
  }
  else
  {
    while ((strchr(characters,*last_char) != nullptr) && (last_char > &_cur_line[_offset]))
      last_char--;
  }

  *(last_char + 1) = '\0';
}

} // script
} // jupiter
} // brt
