//
// Created by daniel on 3/28/19.
//

#ifndef I2C_00_I2CACTION_HPP
#define I2C_00_I2CACTION_HPP

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <map>
#include <cstring>

#include <script.hpp>

#include <metadata.hpp>
#include <value.hpp>

namespace brt {

namespace jupiter {

//class DeviceAction;


///*
// * \\class Session
// *
// * created on: Jul 30, 2019
// *
// */
//class Session : public script::Session
//{
//public:
//  Session(Session* global_session,
//          bool verbose = false)
//  : _verbose(verbose)
//  , _global_session(global_session)
//  { }
//
//  virtual ~Session() {}
//
//          void                    set_verbose(bool verbose) { _verbose = verbose; }
//
//          bool                    verbose() const { return _verbose; }
//          Session*                global() const { return _global_session; }
//
//  virtual script::SessionObject*& object(std::string name)
//          {
//            if (global() != nullptr)
//              return global()->object(name);
//
//            return script::Session::object(name);
//          }
//
//          void                    initialize(std::vector<DeviceAction*>& action_array);
//          void                    run(std::vector<DeviceAction*>&);
//
//  virtual Value                   var(std::string name);
//          void                    to_meta(std::string name);
//
//private:
//  bool                            _verbose;
//  Session*                        _global_session;
//};
//

///*
// * \\class ParserEnv
// *
// * created on: Jul 9, 2019
// *
// */
//class ParserEnv : public script::ParserString
//{
//public:
//  ParserEnv(char* buffer) : ParserString(buffer), _env() {}
//  ParserEnv(const char* buffer) : ParserString(buffer), _env() {}
//
//  virtual ~ParserEnv() {}
//          Metadata&               env() { return _env; }
//private:
//  Metadata                        _env;
//};
//
//
///**
// *
// */
//class DeviceAction
//{
//protected:
//  DeviceAction() {}
//public:
//  virtual ~DeviceAction() {}
//
//  static  DeviceAction*           read_line(ParserEnv&);
//  virtual bool                    do_action(Session&) = 0;
//
//protected:
//  virtual bool                    extract(ParserEnv&) = 0;
//
//private:
//  static size_t                   _line_number;
//};
//
//
//
///**
// *
// */
//class ActionDelay : public DeviceAction
//{
//public:
//  ActionDelay()
//  : _useconds(nullptr)
//  , _multiplier(1)
//  { }
//
//  virtual ~ActionDelay()
//  {
//    if (_useconds != nullptr)
//      delete _useconds;
//  }
//
//  virtual bool                    do_action(Session&);
//protected:
//  virtual bool                    extract(ParserEnv&);
//
//private:
//  script::Expression*             _useconds;
//  uint32_t                        _multiplier;
//};
//
/**
 *
 */
class ActionRead : public script::ScriptAction
{
public:
  ActionRead()
  : _num_bytes(nullptr)
  , _device_address(nullptr)
  , _offset(nullptr)
  , _target(nullptr)
  {  }

  virtual ~ActionRead()
  {
    if (_num_bytes != nullptr)
      delete _num_bytes;

    if (_device_address != nullptr)
      delete _device_address;

    if (_offset != nullptr)
      delete _offset;

    if (_target != nullptr)
      delete _target;
  }

  virtual bool                    do_action(script::Session&);
  virtual bool                    extract(script::ParserEnv&);
  virtual script::ScriptAction*   get_copy();

private:
  script::Expression*             _num_bytes;
  script::Expression*             _device_address;

  script::Expression*             _offset;
  script::Expression*             _target;
};

/**
 *
 */
class ActionWrite : public script::ScriptAction
{
public:
  ActionWrite() : _device_address(nullptr), _offset(nullptr), _bytes()  {}
  virtual ~ActionWrite()
  {
    if (_device_address != nullptr)
      delete _device_address;

    if (_offset != nullptr)
      delete _offset;

    if (_bytes != nullptr)
      delete _bytes;
  }

  virtual bool                    do_action(script::Session&);
  virtual bool                    extract(script::ParserEnv&);
  virtual script::ScriptAction*   get_copy();

private:
  script::Expression*             _device_address;
  script::Expression*             _offset;
  script::ExpressionArray*        _bytes;
};


///**
// *
// */
//class ActionEcho : public DeviceAction
//{
//public:
//  ActionEcho() : _text(nullptr) {}
//  virtual ~ActionEcho()
//  {
//    if (_text != nullptr)
//      delete _text;
//  }
//
//  virtual bool                    do_action(Session&);
//protected:
//  virtual bool                    extract(ParserEnv&);
//
//private:
//  script::Expression*             _text;
//};
//
///**
// *
// */
//class ActionExpression : public DeviceAction
//{
//public:
//  ActionExpression() : _expr(nullptr) {}
//  virtual ~ActionExpression()
//  {
//    if (_expr != nullptr)
//      delete _expr;
//  }
//
//  virtual bool                    do_action(Session&);
//protected:
//  virtual bool                    extract(ParserEnv&);
//
//private:
//  script::Expression*             _expr;
//};
//
//
///**
// *
// */
//class ActionMacro : public DeviceAction
//{
//public:
//  ActionMacro() : _action_array() {}
//  virtual ~ActionMacro()
//  {
//    while (!_action_array.empty())
//    {
//      delete _action_array.front();
//      _action_array.erase(_action_array.begin());
//    }
//  }
//
//  /**
//   *
//   */
//  class SessionObject : public script::SessionObject
//  {
//  public:
//    SessionObject(ActionMacro* macro) : _macro(macro) {}
//    virtual ~SessionObject() {}
//
//            ActionMacro*        get() const { return _macro; }
//
//  private:
//    ActionMacro*                 _macro;
//  };
//
//          std::string             name() const { return _name; }
//          bool                    run_macro(Session&,const std::vector<Value>&);
//
//  virtual bool                    do_action(Session&) { return true; }
//
//protected:
//  virtual bool                    extract(ParserEnv&);
//
//private:
//  std::vector<DeviceAction*>      _action_array;
//  std::vector<std::string>        _arguments;
//  std::string                     _name;
//};
//
//
///*
// * \\class ActionLoop
// *
// * created on: Jul 9, 2019
// *
// */
//class ActionLoop : public DeviceAction
//{
//public:
//  ActionLoop() : _condition(nullptr) {}
//  virtual ~ActionLoop()
//  {
//    while (!_action_array.empty())
//    {
//      delete _action_array.front();
//      _action_array.erase(_action_array.begin());
//    }
//
//    if (_condition != nullptr)
//      delete _condition;
//  }
//
//  virtual bool                    do_action(Session&);
//protected:
//  virtual bool                    extract(ParserEnv&);
//
//private:
//  std::vector<DeviceAction*>      _action_array;
//  script::Expression*             _condition;
//};
//
///*
// * \\class ActionIf
// *
// * created on: Jul 9, 2019
// *
// */
//class ActionIf : public DeviceAction
//{
//public:
//  ActionIf() : _statement() {}
//  virtual ~ActionIf()
//  {
//    while (!_statement.empty())
//    {
//      delete _statement.front();
//      _statement.erase(_statement.begin());
//    }
//  }
//
//  virtual bool                    do_action(Session&);
//protected:
//  virtual bool                    extract(ParserEnv&);
//
//private:
//  struct ConditionAction
//  {
//    ConditionAction() : _condition(nullptr), _action_array() {}
//    ~ConditionAction()
//    {
//      if (_condition != nullptr)
//        delete _condition;
//
//      while (!_action_array.empty())
//      {
//        delete _action_array.front();
//        _action_array.erase(_action_array.begin());
//      }
//    }
//
//    script::Expression*             _condition;
//    std::vector<DeviceAction*>      _action_array;
//  };
//
//  std::vector<ConditionAction*>   _statement;
//};
//
///**
// *
// */
//class ActionRunMacro : public DeviceAction
//{
//public:
//  ActionRunMacro() : _arguments() {}
//  virtual ~ActionRunMacro()
//  {
//    while (!_arguments.empty())
//    {
//      delete _arguments.front();
//      _arguments.erase(_arguments.begin());
//    }
//  }
//
//  virtual bool                    do_action(Session&);
//protected:
//  virtual bool                    extract(ParserEnv&);
//
//private:
//  std::vector<script::Expression*>
//                                  _arguments;
//  std::string                     _name;
//};
//
//
///*
// * \\class ISCActionBreak
// *
// * created on: Jul 29, 2019
// *
// */
//class ActionBreak : public DeviceAction
//{
//public:
//  ActionBreak() : _condition(nullptr) {}
//  virtual ~ActionBreak()
//  {
//    if (_condition != nullptr)
//      delete _condition;
//  }
//
//  virtual bool                    do_action(Session&);
//protected:
//  virtual bool                    extract(ParserEnv&);
//
//private:
//  script::Expression*             _condition;
//};
//

} // jupiter
} // brt

#endif //I2C_00_I2CACTION_HPP
