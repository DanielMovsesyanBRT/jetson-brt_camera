//
// Created by daniel on 3/28/19.
//

#include "script_file.hpp"

#include <fstream>
#include <iostream>

#include <parser_string.hpp>
#include <utils.hpp>
#include "script_action.hpp"

namespace brt {

namespace jupiter {

/**
 *
 * @param file_name
 */
ScriptFile::ScriptFile(const char *file_name)
: Session(nullptr)
, _file_path(file_name)
, _busy(false)
{
}


/**
 *
 *
 */
ScriptFile::~ScriptFile()
{
  while (_action_list.size() > 0)
  {
    delete _action_list.front();
    _action_list.erase(_action_list.begin());
  }
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

  // Parse
  ParserEnv ps(file_buffer);

  try
  {
    while (ps.next_token("\n") != nullptr)
    {
      ScriptAction* action = ScriptAction::read_line(ps);
      if (action != nullptr)
        _action_list.push_back(action);
    }

    Session::initialize(_action_list);
  }
  catch(script::ParserException* pe)
  {
    std::cerr << Utils::string_format("%s at line:%d", pe->text(), ps.line_num()) << std::endl;
    pe->release();

    while (_action_list.size() > 0)
    {
      delete _action_list.front();
      _action_list.erase(_action_list.begin());
    }
  }
  delete[] file_buffer;

  return !_action_list.empty();
}

/*
 * \\fn bool ScriptFile::run
 *
 * created on: Jul 2, 2019
 * author: daniel
 *
 */
bool ScriptFile::run()
{
  if (!is_loaded())
    return false;

  bool expected = false;
  if (!_busy.compare_exchange_strong(expected, true))
    return false;

  Session::run(_action_list);

  _busy.store(false);
  return true;
}

/*
 * \\fn bool ScriptFile::run(const char *text)
 *
 * created on: Oct 30, 2019
 * author: daniel
 *
 */
bool ScriptFile::run(const char *text)
{
  ParserEnv ps(text);

  std::vector<ScriptAction*> action_list;

  try
  {
    while (ps.next_token("\n") != nullptr)
    {
      ScriptAction* action = ScriptAction::read_line(ps);
      if (action != nullptr)
        action_list.push_back(action);
    }
  }
  catch(script::ParserException* pe)
  {
    std::cerr << Utils::string_format("%s at line:%d", pe->text(), ps.line_num()) << std::endl;
    pe->release();

    while (action_list.size() > 0)
    {
      delete action_list.front();
      action_list.erase(action_list.begin());
    }

    return false;
  }

  bool expected = false;
  if (!_busy.compare_exchange_strong(expected, true))
    return false;

  set_verbose(true);
  Session::run(action_list);
  set_verbose(false);

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
bool ScriptFile::run_macro(const char *macro_name, script::Value& val, std::vector<script::Value> arguments /*= std::vector<script::Value>()*/)
{
  bool expected = false;
  if (!_busy.compare_exchange_strong(expected, true))
    return false;

  ActionMacro::SessionObject* macro_obj = dynamic_cast<ActionMacro::SessionObject*>(object(macro_name));
  if ((macro_obj == nullptr) || (macro_obj->get() == nullptr))
    return false;

  std::vector<ValueData> values;
  for (size_t index = 0; index < arguments.size(); index++)
    values.push_back(arguments[index]);

  bool result = macro_obj->get()->run_macro(*this,values);
  if (exist(macro_name))
    val = var(macro_name);

  _busy.store(false);


  return result;
}


} // jupiter
} // brt
