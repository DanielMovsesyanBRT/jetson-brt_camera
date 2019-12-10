//
// Created by Daniel Movsesyan on 2019-04-06.
//

#ifndef EXPRESSIONS_EXPRESSION_HPP
#define EXPRESSIONS_EXPRESSION_HPP

#include <map>
#include <vector>


#include <metadata.hpp>
#include "value.hpp"


namespace brt {

namespace jupiter {

namespace script {

/**
 *
 */
class SessionObject
{
public:
  SessionObject() {}
  virtual ~SessionObject() {}
};

/**
 *
 */
class Session : public Metadata
{
public:
  Session() {}

  virtual ~Session()
  {
    while (!_objects.empty())
    {
      delete (*_objects.begin()).second;
      _objects.erase(_objects.begin());
    }
  }

  virtual ValueData&              var(std::string name) { return value(name.c_str()); }
  virtual SessionObject*&         object(std::string name) { return _objects[name]; }

  virtual bool                    object_exist(std::string name) const { return _objects.find(name) != _objects.end(); }

private:
  std::map<std::string,SessionObject*>
                                  _objects;
};


/**
 *
 */
class Expression
{
public:
  Expression() : _result() {}
  virtual ~Expression() {}

  virtual Value                   evaluate(Session*) { return _result; }

protected:
  Value                           _result;
};

/**
 *
 */
class ExpressionArray : public Expression
{
public:
  ExpressionArray() : _expressions() {}
  virtual ~ExpressionArray()
  {
    while(!_expressions.empty())
    {
      delete _expressions.front();
      _expressions.erase(_expressions.begin());
    }
  }

  virtual Value                   evaluate(Session*);

          void                    add_expression(Expression* expr)  { _expressions.push_back(expr); }

          size_t                  num_expresions() const { return _expressions.size(); }
          Expression*             get_expression(size_t index) const { return (index >= _expressions.size()) ? nullptr : _expressions[index]; }

          // Detach array from first num expressions
          void                    detach(size_t num = (size_t)-1)
          {
            if (num >= _expressions.size())
              _expressions.clear();
            else
            {
              while (num-- > 0)
                _expressions.erase(_expressions.begin());
            }
          }

protected:
  std::vector<Expression*>        _expressions;
};


/**
 *
 */
class Constant : public Expression
{
public:
  Constant (const char *text,bool string = false);
  virtual ~Constant() {}

  virtual Value                   evaluate(Session*) { _result = _value; return _result; }

private:
  Value                           _value;
};


/**
 *
 */
class Variable : public Expression
{
public:
  Variable (const char *text);
  virtual ~Variable() {}

  virtual Value                   evaluate(Session*);

private:
  std::string                     _varname;
};

/**
 *
 */
class UnaryExpression : public Expression
{
public:
  UnaryExpression(char op,Expression* expr) : _op(op), _expr(expr) {}
  virtual ~UnaryExpression()
  {
    if (_expr != nullptr)
      delete _expr;
  }

  virtual Value                   evaluate(Session*);

private:
  char                            _op;
  Expression*                     _expr;
};


/**
 *
 */
class IncrDecrExpression :  public Expression
{
public:
  enum OP
  {
    PRE_INCR,
    POST_INCR,
    PRE_DECR,
    POST_DECR
  };


  IncrDecrExpression(OP op, Expression *expr) : _op(op), _expr(expr) {}
  virtual ~IncrDecrExpression()
  {
    if (_expr != nullptr)
      delete _expr;
  }

  virtual Value                   evaluate(Session*);

private:
  OP                              _op;
  Expression*                     _expr;
};

/**
 *
 */
class BinaryExpression : public Expression
{
public:
  BinaryExpression(Expression* lvalue,Expression* rvalue,std::string op);
  virtual ~BinaryExpression()
  {
    if (_lvalue != nullptr)
      delete _lvalue;

    if (_rvalue != nullptr)
      delete _rvalue;
  }

  struct Operation
  {
    Operation() : _operation(""), _priority(1000) {}
    Operation(const Operation& oper) : _operation(oper._operation), _priority(oper._priority) {}
    Operation(std::string _oper,int priority) : _operation(_oper), _priority(priority) {}

    bool                            noop() const { return _operation.empty(); }

    std::string                     _operation;
    int                             _priority;
  };

  virtual Value                   evaluate(Session*);

public:
  static const Operation          get_operation(std::string token);

private:
  Expression*                     _lvalue;
  Expression*                     _rvalue;
  int                             _op; // index from array
};

/*
 * \\class IndexExpression
 *
 * created on: Jul 2, 2019
 *
 */
class IndexExpression : public Expression
{
public:
  IndexExpression(Expression* value,Expression* index)
  : _value(value)
  , _index(index)
  {  }

  virtual ~IndexExpression()
  {
    if (_value != nullptr)
      delete _value;

    if (_index != nullptr)
      delete _index;
  }

  virtual Value                   evaluate(Session*);

private:
  Expression*                     _value;
  Expression*                     _index;
};

/*
 * \\class LogicalExpression
 *
 * created on: Jul 8, 2019
 *
 */
class LogicalExpression : public Expression
{
public:
  LogicalExpression(Expression* condition,
                    Expression* positive,
                    Expression* negative)
  : _condition(condition)
  , _positive(positive)
  , _negative(negative)
  { }

  virtual ~LogicalExpression()
  {
    if (_condition != nullptr)
      delete _condition;

    if(_positive != nullptr)
      delete _positive;

    if (_negative != nullptr)
      delete _negative;
  }

  virtual Value                   evaluate(Session*);

private:
  Expression*                     _condition;
  Expression*                     _positive;
  Expression*                     _negative;
};

} // script
} // jupiter
} // brt

#endif //EXPRESSIONS_EXPRESSION_HPP
