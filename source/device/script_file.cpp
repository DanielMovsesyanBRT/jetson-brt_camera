//
// Created by daniel on 3/28/19.
//

#include "script_file.hpp"

#include <fstream>
#include <iostream>

#include <parser_string.hpp>
#include <utils.hpp>
#include "device_action.hpp"

namespace brt {

namespace jupiter {

/**
 *
 * @param file_name
 */
ScriptFile::ScriptFile(const char *file_name)
: _file_path(file_name)
, _busy(false)
{
}


/**
 *
 *
 */
ScriptFile::~ScriptFile()
{
}


/*
 * \\fn bool ScriptFile::load
 *
 * created on: Jul 25, 2019
 * author: daniel
 *
 */
bool ScriptFile::load()
{
  std::fstream istr(_file_path.c_str(),istr.in);
  if (!istr.is_open())
    return false;

  size_t length = istr.seekg(0, std::ios_base::end).tellg();
  istr.seekg(0);

  if (length == 0)
    return false;

  char* file_buffer = new char[length + 1];
  istr.read(file_buffer, length);
  file_buffer[length] = '\0';

  script::ScriptParser parser;
  _script = parser.parse_script(file_buffer);

  return !_script.empty();
}


/*
 * \\fn script::ScriptAction* ScriptFile::ActionCreator::get_action
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
script::ScriptAction* ScriptFile::ActionCreator::get_action(const char* action)
{
  if (action == nullptr)
    return nullptr;

  switch (action[0])
  {
  case 'r':
  case 'R':
    return new ActionRead();

  case 'w':
  case 'W':
    return new ActionWrite();

  default:
    break;
  }

  return nullptr;
}


/*
 * \\fn bool ScriptFile::run
 *
 * created on: Jul 2, 2019
 * author: daniel
 *
 */
bool ScriptFile::run(Metadata mt /*= Metadata()*/)
{
  if (!is_loaded())
    return false;

  bool expected = false;
  if (!_busy.compare_exchange_strong(expected, true))
    return false;

  _script.run(mt);

  _busy.store(false);
  return true;
}

/*
 * \\fn bool ScriptFile::run
 *
 * created on: Oct 30, 2019
 * author: daniel
 *
 */
bool ScriptFile::run(const char *text,Metadata mt /*= Metadata()*/)
{
  script::ScriptParser parser;
  script::Script commands = parser.parse_script(text);

  if (commands.empty())
    return false;

  bool expected = false;
  if (!_busy.compare_exchange_strong(expected, true))
    return false;

  _script.run(commands, mt);

  _busy.store(false);
  return true;
}

/*
 * \\fn bool ScriptFile::run_macro
 *
 * created on: Jul 2, 2019
 * author: daniel
 *
 */
Value ScriptFile::run_macro(const char *macro_name, std::vector<Value> arguments, Metadata mt /*= Metadata()*/)
{
  bool expected = false;
  if (!_busy.compare_exchange_strong(expected, true))
    return false;

  Value result = _script.run_macro(macro_name, arguments, mt);
  _busy.store(false);

  return result;
}


} // jupiter
} // brt
