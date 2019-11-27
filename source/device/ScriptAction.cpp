//
// Created by daniel on 3/28/19.
//

#include "ScriptAction.hpp"
#ifdef HARDWARE
#include "Deserializer.hpp"
#include "DeviceManager.hpp"
#endif

#include <string.h>
#include <ctype.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include "Parser.hpp"
#include "Utils.hpp"
#include "MetaImpl.hpp"

namespace brt {

namespace jupiter {


#define MAX_LINE_SIZE                       (1024)

size_t ScriptAction::_line_number = 0;

/*
 * \\fn script::Value& Session::var
 *
 * created on: Jul 30, 2019
 * author: daniel
 *
 */
script::ValueData& Session::var(std::string name)
{
  // check local
  if (_global_session != nullptr)
  {
    // This is local stack - first we check if we have
    // local variable with this name
    if (script::Session::var_exist(name))
      return script::Session::var(name);

    // Now check if global session has it
    if (_global_session->var_exist(name))
      return _global_session->var(name);

    // None of the above - create local variable
    return script::Session::var(name);
  }

  // This is Global Scope
  //return Metadata::value(name.c_str());
  return _impl->value(name.c_str());
}

/*
 * \\fn bool Session::var_exist
 *
 * created on: Jul 30, 2019
 * author: daniel
 *
 */
bool Session::var_exist(std::string name)
{
  if (script::Session::var_exist(name))
    return true;

  return Metadata::exist(name.c_str());
}



/*
 * \\fn void Session::initialize
 *
 * created on: Jul 30, 2019
 * author: daniel
 *
 */
void Session::initialize(std::vector<ScriptAction*>& action_array)
{
  // Fill all macros
  for (ScriptAction* action : action_array)
  {
    ActionMacro* macro = dynamic_cast<ActionMacro*>(action);
    if (macro != nullptr)
      object(macro->name()) = new ActionMacro::SessionObject(macro);
  }
}

/*
 * \\fn void Session::run
 *
 * created on: Jul 30, 2019
 * author: daniel
 *
 */
void Session::run(std::vector<ScriptAction*>& action_array)
{
  // Run loop
  for (ScriptAction* action : action_array)
    action->do_action(*this);
}

/**
 *
 * @param str
 * @return
 */
ScriptAction* ScriptAction::read_line(ParserEnv& ps)
{
  // Remove comments
  size_t hash_tag = ps.find('#');
  if (hash_tag != script::ParserString::npos)
    ps.trim_r(hash_tag);

  ScriptAction *result = nullptr;

  // Trim left space
  ps.trim_l();

  const char* helper = ps;
  if (Utils::stristr(ps,"macro") == ps.c_str())
  {
    if (ps.env().exist("macro"))
      throw script::ParserException::create("Invalid macro expression");

    result = new ActionMacro();
    ps.trim_l(5);
  }
  else if (Utils::stristr(ps,"endm") == ps.c_str())
  {
    result = (ScriptAction*)ps.env().get<void*>("macro");
    if (result == nullptr)
      throw script::ParserException::create("Mismatched macro expression");

    ps.trim_l(4);
    return result;
  }
  else if (Utils::stristr(ps,"loop") == ps.c_str())
  {
    result = new ActionLoop();
    ps.trim_l(4);
  }
  else if (Utils::stristr(ps,"endloop") == ps.c_str())
  {
    result = (ScriptAction*)ps.env().get<void*>("loop");
    if (result == nullptr)
      throw script::ParserException::create("Mismatched loop");

    ps.trim_l(4);
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

    case 'r':
    case 'R':
      result = new ActionRead();
      break;

    case 'w':
    case 'W':
      result = new ActionWrite();
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
      return nullptr;
    }
    ps++;
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

/**
 *
 * @return
 */
bool ActionDelay::do_action(Session& session)
{
  if (_useconds == nullptr)
    return false;

  double ms = (double)_useconds->evaluate(&session) * _multiplier;

  if (session.verbose())
    std::cout << std::dec << "Delay " << ms / 1E3 << "ms" << std::endl;

  usleep(ms);

  return true;
}

/**
 *
 * @param str
 * @return
 */
bool ActionDelay::extract(ParserEnv& ps)
{
  // Trim left space
  ps.trim_l();

  size_t rev_index = 0;

  if ((rev_index = ps.rfind("ms")) != script::ParserString::npos)
    _multiplier = 1000;

  else if ((rev_index = ps.rfind("s")) != script::ParserString::npos)
    _multiplier = 1e6;

  else if ((rev_index = ps.rfind("us")) != script::ParserString::npos)
    _multiplier = 1;

  else if ((rev_index = ps.rfind("m")) != script::ParserString::npos)
    _multiplier = 60 * 1e6;

  else if ((rev_index = ps.rfind("min")) != script::ParserString::npos)
    _multiplier = 60 * 1e6;

  else
    _multiplier = 1000;

  if (rev_index != script::ParserString::npos)
    ps.trim_r(rev_index);

  script::Parser pr;
  _useconds = pr.parse(ps);

  return true;
}


/**
 *
 * @return
 */
bool ActionRead::do_action(Session& session)
{
  if ((_num_bytes == nullptr) ||
      (_device_address == nullptr) ||
      (_offset == nullptr) ||
      (_target == nullptr))
  {
    return false;
  }

  int num_bytes = _num_bytes->evaluate(&session);
  int address = _device_address->evaluate(&session);
  script::Value offset = _offset->evaluate(&session);

  uint8_t* buffer = new uint8_t[num_bytes];

#ifdef HARDWARE
  Deserializer* device = static_cast<Deserializer*>(session.get<void*>("device",nullptr));
  if (device == nullptr)
    return false;

  bool result = device->read((uint8_t)address, (int)offset, offset.size(), buffer,num_bytes);
#else
  bool result = true;
#endif
  if (result)
  {
    script::Value val = _target->evaluate(&session);
    val.set_byte_array(buffer, num_bytes, false);

    if (session.verbose())
    {
      std::cout << std::hex << std::uppercase;
      std::cout << "Read from device 0x" << (int) address << ":  ";
      std::cout << "0x" << std::setw(num_bytes * 2) << std::setfill('0') << (int) offset;
      std::cout << " = 0x" << std::setw(2) << std::setfill('0') << (int)val;// (int) buffer[index];

      std::cout << std::endl;
    }
  }
#ifdef HARDWARE
  else
    std::cerr << "Unable to read from device :0x" << std::hex <<(int)address << std::endl;
#endif

  delete[] buffer;
  return result;
}

/**
 *
 * @param str
 * @return
 */
bool ActionRead::extract(ParserEnv& ps)
{
  script::Parser pr;
  script::Expression *expr = pr.parse(ps);

  script::ExpressionArray* exArray = dynamic_cast<script::ExpressionArray*>(expr);
  if ((exArray == nullptr) || (exArray->num_expresions() < 4))
  {
    if (expr != nullptr)
      delete expr;

    throw script::ParserException::create("Invalid Read expression");
  }

  _num_bytes = exArray->get_expression(0);
  _device_address = exArray->get_expression(1);
  _offset = exArray->get_expression(2);
  _target = exArray->get_expression(3);

  exArray->detach(4);
  delete expr;

  return true;
}

/**
 *
 * @return
 */
bool ActionWrite::do_action(Session& session)
{
  int address = _device_address->evaluate(&session);
  script::Value offset = _offset->evaluate(&session);

  std::vector<uint8_t>  buffer;
  for (size_t index = 0; index < _bytes.size(); index++)
  {
    script::Value byte = _bytes[index]->evaluate(&session);

    int size = byte.size();
    int byteValue = byte;
    int shift = 8 * (size - 1);

    while (size-- > 0)
    {
      buffer.push_back((byteValue >> shift) & 0xFF);
      shift -= 8;
    }
  }

  if (session.verbose())
  {
    std::cout << std::hex << std::uppercase;
    std::cout << "Write to device 0x" << (int) address << ":  ";
    std::cout << "0x" << std::setw(offset.size() * 2) << std::setfill('0') << (int) offset;

    for (size_t index = 0; index < buffer.size(); index++)
      std::cout << " 0x" << std::setw(2) << std::setfill('0') << (int) buffer[index];

    std::cout << std::endl;
  }
#ifdef HARDWARE

  Deserializer* device = static_cast<Deserializer*>(session.get<void*>("device",nullptr));
  if (device == nullptr)
    return false;

  return device->write((uint8_t)address, (int)offset, offset.size(), &buffer.front(),buffer.size());
#else
  return true;
#endif
}


/**
 *
 * @param str
 * @return
 */
bool ActionWrite::extract(ParserEnv& ps)
{
  script::Parser pr;
  script::Expression *expr = pr.parse(ps);

  script::ExpressionArray* exArray = dynamic_cast<script::ExpressionArray*>(expr);
  if ((exArray == nullptr) || (exArray->num_expresions() < 2))
  {
    if (expr != nullptr)
      delete expr;

    throw script::ParserException::create("Invalid Write expression");
  }

  _device_address = exArray->get_expression(0);
  _offset = exArray->get_expression(1);

  for (size_t index = 2; index < exArray->num_expresions(); index++)
    _bytes.push_back(exArray->get_expression(index));

  exArray->detach();
  delete expr;

  return true;
}

/**
 *
 * @return
 */
bool ActionEcho::do_action(Session& session)
{
  if (_text == nullptr)
    return false;

  script::Value text = _text->evaluate(&session);

  std::cout << (std::string)text;
  std::cout << std::endl;
  return true;
}


/**
 *
 * @param str
 * @return
 */
bool ActionEcho::extract(ParserEnv& ps)
{
  script::Parser pr;
  _text = pr.parse(ps);

  return true;
}

/**
 *
 * @param session
 * @return
 */
bool ActionExpression::do_action(Session& session)
{
  _expr->evaluate(&session);
  return true;
}


/**
 *
 * @param ps
 * @return
 */
bool ActionExpression::extract(ParserEnv& ps)
{
  script::Parser pr;
  _expr = pr.parse(ps);

  return true;
}




/**
 *
 * @param session
 * @param val_array
 * @return
 */
bool ActionMacro::run_macro(Session& session,const std::vector<script::ValueData>& val_array)
{
  Session stackSession(session.global() != nullptr ? session.global() : &session,
                        session.verbose());

  stackSession.copy_metadata(&session);

  for (size_t index = 0; index < std::min(val_array.size(),_arguments.size()); index++)
    stackSession.var(_arguments[index]) = val_array[index];


  std::vector<ScriptAction*>::iterator iter;
  for (iter = _action_array.begin();iter != _action_array.end();++iter)
    (*iter)->do_action(stackSession);

  if (stackSession.var_exist("_return"))
    session.var(_name.c_str()) = stackSession.var("_return");

  return true;
}



/**
 *
 * @param str
 * @return
 */
bool ActionMacro::extract(ParserEnv& ps)
{
  const char* const_str = ps.c_str();
  std::string token = script::Parser::get_token(const_str);
  if (token.empty() || (!isalpha(token[0]) && (token[0] != '_')))
    return false;

  _name = token;
  token = script::Parser::get_token(const_str);
  if (!token.empty())
  {
    if (token == "(")
      token = script::Parser::get_token(const_str);

    while (!token.empty() && token != ")")
    {
      if (isalpha(token[0]) || (token[0] == '_'))
        _arguments.push_back(token);

      token = script::Parser::get_token(const_str);
    }
  }

  ps.env().set("macro",(void*)this);

  while (ps.next_token("\n") != nullptr)
  {
    ScriptAction* action = ScriptAction::read_line(ps);
    if (action == nullptr)
      continue;

    if (action == this)
    {
      ps.env().erase("macro");
      return true;
    }

    _action_array.push_back(action);
  }

  throw script::ParserException::create("Endless macro");
}

/*
 * \\fn bool ISCActionLoop::do_action
 *
 * created on: Jul 9, 2019
 * author: daniel
 *
 */
bool ActionLoop::do_action(Session& session)
{
  if (_condition == nullptr)
    return false;

  while ((bool)_condition->evaluate(&session))
  {
    std::vector<ScriptAction*>::iterator iter;
    for (iter = _action_array.begin();iter != _action_array.end();++iter)
    {
      if (!(*iter)->do_action(session))
        return false;
    }
  }

  return true;
}

/*
 * \\fn bool ISCActionLoop::extract
 *
 * created on: Jul 9, 2019
 * author: daniel
 *
 */
bool ActionLoop::extract(ParserEnv& ps)
{
  script::Parser pr;
  _condition = pr.parse(ps);

  void* previous_loop = ps.env().get<void*>("loop");
  ps.env().set("loop", (void*)this);

  while (ps.next_token("\n") != nullptr)
  {
    ScriptAction* action = ScriptAction::read_line(ps);
    if (action == nullptr)
      continue;

    if (action == this)
    {
      if (previous_loop != nullptr)
        ps.env().set("loop", previous_loop);
      else
        ps.env().erase("loop");

      return true;
    }

    _action_array.push_back(action);
  }
  throw script::ParserException::create("Endless loop");
}


/**
 *
 * @param session
 * @return
 */
bool ActionRunMacro::do_action(Session& session)
{
  ActionMacro::SessionObject* macro_obj = dynamic_cast<ActionMacro::SessionObject*>(session.object(_name));
  if ((macro_obj == nullptr) || (macro_obj->get() == nullptr))
    return false;

  std::vector<script::ValueData> values;
  for (size_t index = 0; index < _arguments.size(); index++)
    values.push_back(_arguments[index]->evaluate(&session));


  return macro_obj->get()->run_macro(session,values);
}

/**
 *
 * @param str
 * @return
 */
bool ActionRunMacro::extract(ParserEnv& ps)
{
  const char* buf = ps;
  _name = script::Parser::get_token(buf);
  if (_name.empty())
    throw script::ParserException::create("Incomplete macro execution");
    //return false;

  script::Parser pr;
  script::Expression *arg = pr.parse(buf);
  script::ExpressionArray* exArray = dynamic_cast<script::ExpressionArray*>(arg);
  if (exArray != nullptr)
  {
    for (size_t index = 0; index < exArray->num_expresions(); index++)
      _arguments.push_back(exArray->get_expression(index));
  }
  else if (arg != nullptr)
    _arguments.push_back(arg);

  return true;
}

/*
 * \\fn bool ISCActionBreak::do_action
 *
 * created on: Jul 29, 2019
 * author: daniel
 *
 */
bool ActionBreak::do_action(Session& session)
{
  if (_condition != nullptr)
  {
    bool condition = _condition->evaluate(&session);
    if (condition)
    {
      /// Place for breakpoint
      condition =  false;
    }
  }
  return true;
}

/*
 * \\fn bool ISCActionBreak::extract
 *
 * created on: Jul 29, 2019
 * author: daniel
 *
 */
bool ActionBreak::extract(ParserEnv& ps)
{
  script::Parser pr;
  _condition = pr.parse(ps);

  return true;
}


} // jupiter
} // brt
