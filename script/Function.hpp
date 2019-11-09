/*
 * Function.hpp
 *
 *  Created on: Jul 29, 2019
 *      Author: daniel
 */

#ifndef SCRIPT_FUNCTION_HPP_
#define SCRIPT_FUNCTION_HPP_

#include "Expression.hpp"


namespace brt
{
namespace jupiter
{
namespace script
{

/*
 * \\class Function
 *
 * created on: Jul 29, 2019
 *
 */
class Function: public Expression
{
public:
  Function() : _arguments() {}
  virtual ~Function()
  {
    while (_arguments.size() > 0)
    {
      delete (_arguments.front());
      _arguments.erase(_arguments.begin());
    }
  }

          Value                   arg(size_t id,Session* session) { return _arguments[id]->evaluate(session); }
          size_t                  num_args() const { return _arguments.size(); }

          void                    add_argument(Expression* expr) { _arguments.push_back(expr); }

  static  Function*               create_function(const char*);

private:
  std::vector<Expression*>        _arguments;

};

/*
 * \\class FunctionConvertToBool
 *
 * created on: Jul 29, 2019
 *
 */
class FunctionConvertToBool : public Function
{
public:
  FunctionConvertToBool() {}
  virtual ~FunctionConvertToBool() {}

  virtual Value                   evaluate(Session*);
};

/*
 * \\class FunctionConvertToInt
 *
 * created on: Jul 29, 2019
 *
 */
class FunctionConvertToInt : public Function
{
public:
  FunctionConvertToInt() {}
  virtual ~FunctionConvertToInt() {}

  virtual Value                   evaluate(Session*);
};

/*
 * \\class FunctionConvertToReal
 *
 * created on: Jul 29, 2019
 *
 */
class FunctionConvertToReal : public Function
{
public:
  FunctionConvertToReal() {}
  virtual ~FunctionConvertToReal() {}

  virtual Value                   evaluate(Session*);
};

/*
 * \\class FunctionConvertToString
 *
 * created on: Jul 29, 2019
 *
 */
class FunctionConvertToString : public Function
{
public:
  FunctionConvertToString() {}
  virtual ~FunctionConvertToString() {}

  virtual Value                   evaluate(Session*);
};

/*
 * \\class FunctionConvertToBuff
 *
 * created on: Jul 29, 2019
 *
 */
class FunctionConvertToBuff : public Function
{
public:
  FunctionConvertToBuff() {}
  virtual ~FunctionConvertToBuff() {}

  virtual Value                   evaluate(Session*);
};

/*
 * \\class FunctionDec
 *
 * created on: Jul 29, 2019
 *
 */
class FunctionDec : public Function
{
public:
  FunctionDec() {}
  virtual ~FunctionDec() {}

  virtual Value                   evaluate(Session*);
};

/*
 * \\class FunctionHex
 *
 * created on: Jul 29, 2019
 *
 */
class FunctionHex : public Function
{
public:
  FunctionHex() {}
  virtual ~FunctionHex() {}

  virtual Value                   evaluate(Session*);
};

/*
 * \\class FunctionSubArray
 *
 * created on: Jul 31, 2019
 *
 */
class FunctionSubArray : public Function
{
public:
  FunctionSubArray() {}
  virtual ~FunctionSubArray() {}

  virtual Value                   evaluate(Session*);
};

} /* namespace script */
} /* namespace jupiter */
} /* namespace brt */

#endif /* SCRIPT_FUNCTION_HPP_ */
