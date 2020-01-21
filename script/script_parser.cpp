/*
 * script_parser.cpp
 *
 *  Created on: Dec 10, 2019
 *      Author: daniel
 */

#include "script_parser.hpp"
#include <utils.hpp>

#include <iostream>


namespace brt
{
namespace jupiter
{
namespace script
{

/*
 * \\fn Constructor ScriptParser::ScriptParser()
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
ScriptParser::ScriptParser()
{
}

/*
 * \\fn Destructor ScriptParser::~ScriptParser
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
ScriptParser::~ScriptParser()
{
}

/*
 * \\fn ScriptActionList ScriptParser::parse_script
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
Script ScriptParser::parse_script(const char* text,CreatorContainer cc /*= CreatorContainer()*/)
{
  ParserEnv ps(text, this, cc);
  Script al;

  try
  {
    while (ps.next_token("\n") != nullptr)
    {
      ScriptAction* action = read_line(ps);
      if (action != nullptr)
        al.add(Script::ActionPtr(action));
    }

    al.load_objects();
  }
  catch(script::ParserException* pe)
  {
    std::cerr << Utils::string_format("%s at line:%d", pe->text(), ps.line_num()) << std::endl;
    pe->release();
    return Script();
  }

  return al;
}

/*
 * \\fn ScriptAction* ScriptParser::read_line
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
ScriptAction* ScriptParser::read_line(ParserEnv& ps)
{
  // Remove comments
  size_t hash_tag = ps.find('#');
  if (hash_tag != ParserString::npos)
    ps.trim_r(hash_tag);

  ScriptAction *result = nullptr;

  // Trim left space
  ps.trim_l();
  if (Utils::stristr(ps,"macro") == ps.c_str())
  {
    if (ps.env().exist("macro"))
      throw ParserException::create("Invalid macro expression");

    result = new ActionMacro();
    ps.word_right();
  }
  else if (Utils::stristr(ps,"endm") == ps.c_str())
  {
    result = (ScriptAction*)ps.env().get<void*>("macro");
    if (result == nullptr)
      throw script::ParserException::create("Mismatched macro expression");

    ps.word_right();
    return result;
  }
  // Loop
  else if (Utils::stristr(ps,"loop") == ps.c_str())
  {
    result = new ActionLoop();
    ps.word_right();
  }
  else if (Utils::stristr(ps,"endloop") == ps.c_str())
  {
    result = (ScriptAction*)ps.env().get<void*>("loop");
    if (result == nullptr)
      throw script::ParserException::create("Mismatched loop");

    ps.word_right();
    return result;
  }
  // If
  else if (Utils::stristr(ps,"if") == ps.c_str())
  {
    result = new ActionIf();
    ps.word_right();
  }
  else if ((Utils::stristr(ps,"elif") == ps.c_str()) ||
           (Utils::stristr(ps,"endif") == ps.c_str()) ||
           (Utils::stristr(ps,"else") == ps.c_str()) )
  {
    result = (ScriptAction*)ps.env().get<void*>("if");
    if (result == nullptr)
      throw ParserException::create("Mismatched if statement");

    return result;
  }
  else
  {
    switch (ps[0])
    {
    case 'd':
    case 'D':
      result = new ActionDelay();
      break;

    case 'e':
    case 'E':
      result = new ActionEcho();
      break;

    case 's':
    case 'S':
      result = new ActionExpression();
      break;

    case 'x':
    case 'X':
      result = new ActionRunMacro();
      break;

    case 'b':
    case 'B':
      result = new ActionBreak();
      break;

    default:
      break;
    }

    if (result == nullptr)
      result = ps.cc().create_action(ps);

    if (result != nullptr)
      ps.word_right();
  }

  // Although there is no way that the result will be null at this point
  // it is certainly good idea to check its condition...
  if (result == nullptr)
    return nullptr;

  if (!result->extract(ps))
  {
    delete result;
    throw script::ParserException::create(Utils::string_format("Incorrect string %s",ps.c_str()).c_str());
  }

  return result;
}

} /* namespace script */
} /* namespace jupiter */
} /* namespace brt */
