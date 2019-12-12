/*
 * script_action.cpp
 *
 *  Created on: Dec 10, 2019
 *      Author: daniel
 */

#include "script.hpp"
#include "script_parser.hpp"
#include "parser.hpp"
#include "session.hpp"

#include <utils.hpp>

#include <iostream>
#include <fstream>
#include <iostream>

#include <unistd.h>


namespace brt
{
namespace jupiter
{
namespace script
{
/*
 * \\fn Constructor ScriptAction::ScriptAction
 *
 * created on: Dec 10, 2019
 * author: daniel
 *
 */
ScriptAction::ScriptAction()
{

}

/*
 * \\fn Destructor ScriptAction::~ScriptAction
 *
 * created on: Dec 10, 2019
 * author: daniel
 *
 */
ScriptAction::~ScriptAction()
{

}

/*
 * \\fn Constructor ScriptActionList::ScriptActionList
 *
 * created on: Dec 10, 2019
 * author: daniel
 *
 */
Script::Script()
 : _data(nullptr)
{

}

/*
 * \\fn Constructor ScriptActionList::ScriptActionList
 *
 * created on: Dec 10, 2019
 * author: daniel
 *
 */
Script::Script(const Script& list)
: _data(list._data)
{

}

/*
 * \\fn Destructor ScriptActionList::~ScriptActionList
 *
 * created on: Dec 10, 2019
 * author: daniel
 *
 */
Script::~Script()
{

}

/*
 * \\fn bool Script::load_from_file
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
bool Script::load_from_file(const char* file_name,CreatorContainer cc /*= CreatorContainer*/)
{
  std::fstream istr(file_name,istr.in);
  if (!istr.is_open())
    return false;

  size_t length = istr.seekg(0, std::ios_base::end).tellg();
  istr.seekg(0);

  if (length == 0)
    return false;

  char* file_buffer = new char[length + 1];
  istr.read(file_buffer, length);
  file_buffer[length] = '\0';

  ScriptParser parser;
  *this += parser.parse_script(file_buffer,cc);

  return !empty();
}

/*
 * \\fn Script::load
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
bool Script::load(const char* text,CreatorContainer cc /*= CreatorContainer*/)
{
  ScriptParser parser;
  *this += parser.parse_script(text,cc);

  return !empty();
}

/*
 * \\fn void ScriptActionList::add
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
void Script::add(ActionPtr sa)
{
  if (!_data)
    _data.reset(new SAData);

  std::unique_lock<std::mutex> l(_data->_mutex);
  _data->_array.push_back(sa);
}


/*
 * \\fn Script::load_objects name
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
void Script::load_objects()
{
  if (!_data)
    return;

  std::unique_lock<std::mutex> l(_data->_mutex);
  for (auto action : _data->_array)
  {
    ActionMacro::MacroPtr macro = std::dynamic_pointer_cast<ActionMacro>(action);
    if (macro)
      _data->_session.object(macro->name()).reset(new ActionMacro::SessionObject(macro));
  }
}


/*
 * \\fn void Script::run
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
void Script::run(Metadata meta /*= Metadata()*/)
{
  if (!_data)
    return;

  std::unique_lock<std::mutex> l(_data->_mutex);

  _data->_session += meta;
  for (auto action : _data->_array)
    action->do_action(_data->_session);

}

/*
 * \\fn Script::run
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
void Script::run(Script other_script,Metadata meta /*= Metadata()*/)
{
  if (!_data || !other_script._data)
    return;

  std::lock(_data->_mutex, other_script._data->_mutex);
  std::lock_guard<std::mutex> lk1(_data->_mutex, std::adopt_lock);
  std::lock_guard<std::mutex> lk2(other_script._data->_mutex, std::adopt_lock);

  _data->_session += meta;
  for (auto action : other_script._data->_array)
    action->do_action(_data->_session);
}


/*
 * \\fn Value Script::run_macro
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
Value Script::run_macro(const char* macro_name,std::vector<Value> arguments,Metadata meta /*= Metadata()*/)
{
  if (!_data)
    return Value();

  std::lock_guard<std::mutex> lk1(_data->_mutex);

  std::shared_ptr<ActionMacro::SessionObject> macro_obj =
      std::dynamic_pointer_cast<ActionMacro::SessionObject>(_data->_session.object(macro_name));

  if (!macro_obj)
    return Value();

  Session sess(&_data->_session);
  sess += meta;

  bool result = macro_obj->get()->run_macro(sess,arguments);
  if (sess.exist(macro_name))
    return sess.var(macro_name);

  return Value();
}

/*
 * \\fn Value Script::get
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
Value Script::get(const char* var_name)
{
  if (!_data)
    return Value();

  std::lock_guard<std::mutex> lk1(_data->_mutex);
  if (_data->_session.exist(var_name))
    return _data->_session.value(var_name);

  return Value();
}

/*
 * \\fn void Script::set
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
void Script::set(Metadata meta)
{
  if (!_data)
    return;

  std::lock_guard<std::mutex> lk1(_data->_mutex);
  _data->_session += meta;
}

/*
 * \\fn void Script::set
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
void Script::set(const char* var_name,Value val)
{
  if (!_data)
    return;

  std::lock_guard<std::mutex> lk1(_data->_mutex);
  _data->_session.value(var_name) = val;
}

/*
 * \\fn Script& Script::operator+=
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
Script& Script::operator+=(const Script& script)
{
  if (!script._data)
    return *this;

  if (!_data)
    _data.reset(new SAData);

  std::lock(_data->_mutex, script._data->_mutex);
  std::lock_guard<std::mutex> lk1(_data->_mutex, std::adopt_lock);
  std::lock_guard<std::mutex> lk2(script._data->_mutex, std::adopt_lock);

  for (auto action : script._data->_array)
    _data->_array.push_back(action);

  _data->_session += script._data->_session;
  return *this;
}

/*
 * \\fn ScriptActionList::size
 *
 * created on: Dec 10, 2019
 * author: daniel
 *
 */
size_t Script::size() const
{
  if (_data == nullptr)
    return 0;

  std::unique_lock<std::mutex> l(_data->_mutex);
  return _data->_array.size();
}

/*
 * \\fn bool Script::empty
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
bool Script::empty() const
{
  if (_data == nullptr)
    return true;

  std::unique_lock<std::mutex> l(_data->_mutex);
  return _data->_array.empty();
}

/*
 * \\fn ActionPtr ScriptActionList::operator
 *
 * created on: Dec 10, 2019
 * author: daniel
 *
 */
Script::ActionPtr Script::operator[](size_t index)
{
  if (index >= size())
    return Script::ActionPtr();

  std::unique_lock<std::mutex> l(_data->_mutex);
  return _data->_array[index];
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

  if ((rev_index = ps.rfind("ms")) != ParserString::npos)
    _multiplier = 1000;

  else if ((rev_index = ps.rfind("s")) != ParserString::npos)
    _multiplier = 1e6;

  else if ((rev_index = ps.rfind("us")) != ParserString::npos)
    _multiplier = 1;

  else if ((rev_index = ps.rfind("m")) != ParserString::npos)
    _multiplier = 60 * 1e6;

  else if ((rev_index = ps.rfind("min")) != ParserString::npos)
    _multiplier = 60 * 1e6;

  else
    _multiplier = 1000;

  if (rev_index != ParserString::npos)
    ps.trim_r(rev_index);

  Parser pr;
  _useconds = pr.parse(ps);

  return true;
}

/*
 * \\fn bool ActionEcho::do_action
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
bool ActionEcho::do_action(Session& session)
{
  if (_text == nullptr)
    return false;

  Value text = _text->evaluate(&session);

  std::cout << (std::string)text;
  std::cout << std::endl;
  return true;
}


/*
 * \\fn bool ActionEcho::extract
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
bool ActionEcho::extract(ParserEnv& ps)
{
  Parser pr;
  _text = pr.parse(ps);

  return true;
}

/*
 * \\fn bool ActionExpression::do_action
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
bool ActionExpression::do_action(Session& session)
{
  _expr->evaluate(&session);
  return true;
}

/*
 * \\fn bool ActionExpression::extract
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
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
bool ActionMacro::run_macro(Session& session,const std::vector<Value>& val_array)
{
  Session stackSession(session.global());

  for (size_t index = 0; index < std::min(val_array.size(),_arguments.size()); index++)
    stackSession.var(_arguments[index]) = val_array[index];

  for (auto action : _block)
    action->do_action(stackSession);

  if (stackSession.exist("_return"))
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
  std::string token = Parser::get_token(const_str);
  if (token.empty() || (!isalpha(token[0]) && (token[0] != '_')))
    return false;

  _name = token;
  token = Parser::get_token(const_str);
  if (!token.empty())
  {
    if (token == "(")
      token = Parser::get_token(const_str);

    while (!token.empty() && token != ")")
    {
      if (isalpha(token[0]) || (token[0] == '_'))
        _arguments.push_back(token);

      token = Parser::get_token(const_str);
    }
  }

  ps.env().set("macro",(void*)this);

  while (ps.next_token("\n") != nullptr)
  {
    ScriptAction* action = ps.parser()->read_line(ps);
    if (action == nullptr)
      continue;

    if (action == this)
    {
      ps.env().erase("macro");
      return true;
    }

    _block.add(Script::ActionPtr(action));
  }

  throw ParserException::create("Endless macro");
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
    for (auto action : _block)
    {
      if (!action->do_action(session))
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
  Parser pr;
  _condition = pr.parse(ps);

  void* previous_loop = ps.env().get<void*>("loop");
  ps.env().set("loop", (void*)this);

  while (ps.next_token("\n") != nullptr)
  {
    ScriptAction* action = ps.parser()->read_line(ps);
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

    _block.add(Script::ActionPtr(action));
  }
  throw ParserException::create("Endless loop");
}

/*
 * \\fn ActionIf::do_action
 *
 * created on: Nov 28, 2019
 * author: daniel
 *
 */
bool ActionIf::do_action(Session& session)
{
  for (size_t index = 0; index < _statement.size(); index++)
  {
    if ((_statement[index]->_condition == nullptr) ||
        ((bool)_statement[index]->_condition->evaluate(&session)))
    {
      for (auto action : _statement[index]->_block)
        action->do_action(session);

      break;
    }
  }

  return true;
}

/*
 * \\fn ActionIf::extract
 *
 * created on: Nov 28, 2019
 * author: daniel
 *
 */
bool ActionIf::extract(ParserEnv& ps)
{
  Parser pr;
  _statement.push_back(new ConditionAction);
  ConditionAction *ca = _statement[_statement.size() - 1];
  ca->_condition = pr.parse(ps);

  void* previous_if = ps.env().get<void*>("if");
  ps.env().set("if", (void*)this);

  while (ps.next_token("\n") != nullptr)
  {
    ScriptAction* action = ps.parser()->read_line(ps);
    if (action == nullptr)
      continue;

    if (action == this)
    {
      if (Utils::stristr(ps,"endif") == ps.c_str())
      {
        ps.trim_l(5);
        if (previous_if != nullptr)
          ps.env().set("if", previous_if);
        else
          ps.env().erase("if");

        return true;
      }
      else if (Utils::stristr(ps,"elif") == ps.c_str())
      {
        ps.trim_l(4);

        _statement.push_back(new ConditionAction);
        ca = _statement[_statement.size() - 1];
        ca->_condition = pr.parse(ps);
        continue;
      }
      else if (Utils::stristr(ps,"else") == ps.c_str())
      {
        ps.trim_l(4);

        _statement.push_back(new ConditionAction);
        ca = _statement[_statement.size() - 1];
        continue;
      }
      else
        throw ParserException::create("Undefined if statement");
    }

    ca->_block.add(Script::ActionPtr(action));
  }

  throw ParserException::create("Endless if statement");
}

/**
 *
 * @param session
 * @return
 */
bool ActionRunMacro::do_action(Session& session)
{
  std::shared_ptr<ActionMacro::SessionObject> macro_obj =
      std::dynamic_pointer_cast<ActionMacro::SessionObject>(session.object(_name));

  if (!macro_obj)
    return false;

  if (_arguments == nullptr)
    return false;

  std::vector<Value> values;
  for (size_t index = 0; index < _arguments->num_expresions(); index++)
    values.push_back(_arguments->get_expression(index)->evaluate(&session));

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
  _name = Parser::get_token(buf);
  if (_name.empty())
    throw ParserException::create("Incomplete macro execution");

  Parser pr;
  Expression *arg = pr.parse(buf);
  _arguments = dynamic_cast<ExpressionArray*>(arg);

  if (_arguments == nullptr)
  {
    _arguments = new ExpressionArray;
    _arguments->add_expression(arg);
  }

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
  Parser pr;
  _condition = pr.parse(ps);

  return true;
}

} /* namespace script */
} /* namespace jupiter */
} /* namespace brt */
