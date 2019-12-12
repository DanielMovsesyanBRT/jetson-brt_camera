/*
 * script_action.hpp
 *
 *  Created on: Dec 10, 2019
 *      Author: daniel
 */

#ifndef SCRIPT_SCRIPT_HPP_
#define SCRIPT_SCRIPT_HPP_

#include <vector>
#include <atomic>
#include <mutex>
#include <string>
#include <set>

#include "session.hpp"
#include "expression.hpp"

namespace brt
{
namespace jupiter
{
namespace script
{

class ParserEnv;

/*
 * \\class ScriptAction
 *
 * created on: Dec 10, 2019
 *
 */
class ScriptAction
{
public:
  ScriptAction();
  virtual ~ScriptAction();

  virtual bool                    do_action(Session&) = 0;
  virtual bool                    extract(ParserEnv&) = 0;
  virtual ScriptAction*           get_copy() = 0;
};

/*
 * \\struct iActionInterface
 *
 * created on: Dec 11, 2019
 *
 */
struct iActionInterface
{
  virtual ~iActionInterface() {}
  virtual ScriptAction*           create_action(const char* action) = 0;
};

/*
 * \\class CreatorContainer
 *
 * created on: Dec 11, 2019
 *
 */
class CreatorContainer
{
public:
  CreatorContainer(iActionInterface* iface = nullptr)
  {
    if (iface != nullptr)
      _creators.insert(iface);
  }

  virtual ~CreatorContainer() {}

  CreatorContainer& add(iActionInterface* iface)
  {
    _creators.insert(iface);
    return *this;
  }

  ScriptAction*           create_action(const char* action)
  {
    for (auto creator : _creators)
    {
      ScriptAction* sa = creator->create_action(action);
      if (sa != nullptr)
        return sa;
    }

    return nullptr;
  }

private:
  std::set<iActionInterface*>     _creators;
};

/*
 * \\class Script
 *
 * created on: Dec 10, 2019
 *
 */
class Script
{
private:
  struct SAData
  {
    mutable std::mutex              _mutex;
    std::vector<ScriptAction*>      _array;
    std::atomic_int_fast32_t        _reference;
    Session                         _session;
  };

public:

  template<class T>
  class Iter
  {
  public:
    Iter(Script *owner, size_t index = 0) : _owner(owner), _index(index)
    {
      _owner->addref();
    }

    virtual ~Iter()
    {
      _owner->release();
    }

    T operator*()
    {
      if (_owner->_data == nullptr || _index >= _owner->_data->_array.size())
        return nullptr;

      return _owner->_data->_array.at(_index);
    }

    Iter& operator++()  { _index++; return *this; }
    bool operator==(const Iter& rval) { return _index == rval._index; }
    bool operator!=(const Iter& rval) { return _index != rval._index; }

  private:
    Script*                         _owner;
    size_t                          _index;
  };

  typedef Iter<ScriptAction*>     iterator;

  Script();
  Script(const Script&);
  virtual ~Script();

          bool                    load_from_file(const char* file_name,CreatorContainer = CreatorContainer());
          bool                    load(const char* text,CreatorContainer = CreatorContainer());

          void                    add(ScriptAction*);
          void                    load_objects();

          void                    run(Metadata meta = Metadata());
          void                    run(Script,Metadata meta = Metadata());
          Value                   run_macro(const char* macro_name,std::vector<Value> arguments,Metadata meta = Metadata());

          Value                   get(const char* var_name);
          void                    set(Metadata meta);
          void                    set(const char* var_name,Value val);

          Script&                 operator=(const Script&);
          Script&                 operator+=(const Script&);
          ScriptAction*           operator[](size_t index);

          size_t                  size() const;
          bool                    empty() const;

          iterator                begin() { return iterator(this, 0); }
          iterator                end() { return iterator(this, size()); }

private:
          void                    release();
          void                    addref();

  SAData*                         _data;
};



/*
 * \\class ActionDelay
 *
 * created on: Dec 11, 2019
 *
 */
class ActionDelay : public ScriptAction
{
public:
  ActionDelay()
  : _useconds(nullptr)
  , _multiplier(1)
  { }

  virtual ~ActionDelay()
  {
    if (_useconds != nullptr)
      delete _useconds;
  }

  virtual bool                    do_action(Session&);
  virtual bool                    extract(ParserEnv&);
  virtual ScriptAction*           get_copy();

private:
  Expression*                     _useconds;
  uint32_t                        _multiplier;
};


/*
 * \\class ActionEcho
 *
 * created on: Dec 11, 2019
 *
 */
class ActionEcho : public ScriptAction
{
public:
  ActionEcho() : _text(nullptr) {}
  virtual ~ActionEcho()
  {
    if (_text != nullptr)
      delete _text;
  }

  virtual bool                    do_action(Session&);
  virtual bool                    extract(ParserEnv&);
  virtual ScriptAction*           get_copy();

private:
  Expression*                     _text;
};


/*
 * \\class ActionExpression
 *
 * created on: Dec 11, 2019
 *
 */
class ActionExpression : public ScriptAction
{
public:
  ActionExpression() : _expr(nullptr) {}
  virtual ~ActionExpression()
  {
    if (_expr != nullptr)
      delete _expr;
  }

  virtual bool                    do_action(Session&);
  virtual bool                    extract(ParserEnv&);
  virtual ScriptAction*           get_copy();

private:
  Expression*                     _expr;
};


/*
 * \\class ActionMacro
 *
 * created on: Dec 11, 2019
 *
 */
class ActionMacro : public ScriptAction
{
public:
  ActionMacro() : _block() {}
  virtual ~ActionMacro() {}

  /**
   *
   */
  class SessionObject : public script::SessionObject
  {
  public:
    SessionObject(ActionMacro* macro) : _macro(macro) {}
    virtual ~SessionObject() {}

            ActionMacro*        get() const { return _macro; }

  private:
    ActionMacro*                 _macro;
  };

          std::string             name() const { return _name; }
          bool                    run_macro(Session&,const std::vector<Value>&);

  virtual bool                    do_action(Session&) { return true; }
  virtual bool                    extract(ParserEnv&);
  virtual ScriptAction*           get_copy();

private:
  Script                          _block;
  std::vector<std::string>        _arguments;
  std::string                     _name;
};


/*
 * \\class ActionLoop
 *
 * created on: Jul 9, 2019
 *
 */
class ActionLoop : public ScriptAction
{
public:
  ActionLoop() : _block(), _condition(nullptr) {}
  virtual ~ActionLoop()
  {
    if (_condition != nullptr)
      delete _condition;
  }

  virtual bool                    do_action(Session&);
  virtual bool                    extract(ParserEnv&);
  virtual ScriptAction*           get_copy();

private:
  Script                          _block;
  Expression*                     _condition;
};

/*
 * \\class ActionIf
 *
 * created on: Jul 9, 2019
 *
 */
class ActionIf : public ScriptAction
{
public:
  ActionIf() : _statement() {}
  virtual ~ActionIf()
  {
    while (!_statement.empty())
    {
      delete _statement.front();
      _statement.erase(_statement.begin());
    }
  }

  virtual bool                    do_action(Session&);
  virtual bool                    extract(ParserEnv&);
  virtual ScriptAction*           get_copy();

private:
  struct ConditionAction
  {
    ConditionAction() : _condition(nullptr), _block() {}
    ConditionAction(const ConditionAction& ca)
    {
      _condition = (ca._condition != nullptr) ? ca._condition->create_copy() : nullptr;
      _block += ca._block;
    }

    ~ConditionAction()
    {
      if (_condition != nullptr)
        delete _condition;
    }

    Expression*                     _condition;
    Script                          _block;
  };

  std::vector<ConditionAction*>   _statement;
};

/**
 *
 */
class ActionRunMacro : public ScriptAction
{
public:
  ActionRunMacro() : _arguments(nullptr) {}
  virtual ~ActionRunMacro()
  {
    if (_arguments != nullptr)
      delete _arguments;
  }

  virtual bool                    do_action(Session&);
  virtual bool                    extract(ParserEnv&);
  virtual ScriptAction*           get_copy();

private:
  ExpressionArray*                _arguments;
  std::string                     _name;
};


/*
 * \\class ISCActionBreak
 *
 * created on: Jul 29, 2019
 *
 */
class ActionBreak : public ScriptAction
{
public:
  ActionBreak() : _condition(nullptr) {}
  virtual ~ActionBreak()
  {
    if (_condition != nullptr)
      delete _condition;
  }

  virtual bool                    do_action(Session&);
  virtual bool                    extract(ParserEnv&);
  virtual ScriptAction*           get_copy();

private:
  Expression*                     _condition;
};



} /* namespace script */
} /* namespace jupiter */
} /* namespace brt */

#endif /* SCRIPT_SCRIPT_HPP_ */
