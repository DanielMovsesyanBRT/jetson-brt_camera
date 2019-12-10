//
// Created by Daniel Movsesyan on 2019-04-06.
//

#include "expression.hpp"

#include <string.h>
#include "parser.hpp"

#undef _lengthof
#define _lengthof(x)    (sizeof(x)/sizeof(x[0]))

namespace brt {

namespace jupiter {

namespace script {

/**
 *
 * @param text
 * @param string
 */
Constant::Constant (const char *text,bool string /*= false*/)
{
  if (!string)
  {
    std::string token = toupper(text);
    if ((token == "TRUE") || (token == "FALSE"))
    {
      _value.set(token == "TRUE");
    }
    else if (((strchr(text, '.') != nullptr) ||
         (strchr(text, 'E') != nullptr) ||
         (strchr(text, 'e') != nullptr)) &&
        (strstr(text, "0x") == nullptr) &&
        (strstr(text, "0X") == nullptr))
    {
      _value.set(strtod(text, nullptr));
    }
    else
    {
      // Check presence of suffix
      const char *suffix = strchr(text, 's');
      if (suffix == nullptr)
        suffix = strchr(text, 'S');

      if (suffix != nullptr)
      {
        size_t size = strtoul(suffix + 1,nullptr,0);
        _value.set((int)strtol(text, nullptr, 0), size);
      }
      else
      {
        char *end;
        int val = strtol(text, &end, 0);
        if ((strstr(text, "0x") == text) || (strstr(text, "0X") == text))
          _value.set(val,(end - text - 2) / 2);
        else
        {
          size_t size = (val == 0) ? (sizeof(int) * 8) : __builtin_clz(val);
          size = ((sizeof(int) * 8) - size) / 8 + 1;
          _value.set(val,size);
        }
      }
    }
  }
  else
    _value.set(text);
}

/**
 *
 * @param session
 * @return
 */
Value ExpressionArray::evaluate(Session* session)
{
  for (Expression* expr : _expressions)
    _result = expr->evaluate(session);

  return _result;
}



/**
 *
 * @param text
 */
Variable::Variable (const char *text)
{
  if (text[0] == '$')
    _varname = &text[1];
  else
    _varname = text;
}

/**
 *
 * @param session
 * @return
 */
Value Variable::evaluate(Session* session)
{
  return Value(session->var(_varname));
}

/**
 *
 * @param session
 * @return
 */
Value UnaryExpression::evaluate(Session* session)
{
  _result = _expr->evaluate(session);
  switch (_op)
  {
  case '-':
    _result = -_result;
    break;

  case '!':
    _result = !_result;
    break;

  case '~':
    _result = ~_result;
    break;

  default:
    break;
  }

  return _result;
}

/**
 *
 * @param session
 * @return
 */
Value IncrDecrExpression::evaluate(Session* session)
{
  switch(_op)
  {
  case PRE_INCR:
    _result = ++_expr->evaluate(session);
    break;

  case POST_INCR:
    _result = _expr->evaluate(session)++;
    break;

  case PRE_DECR:
    _result = --_expr->evaluate(session);
    break;

  case POST_DECR:
    _result = _expr->evaluate(session)--;
    break;
  }
  return _result;
}



const BinaryExpression::Operation opers[] =
{
  {"*",     0 },
  {"/",     0 },
  {"%",     0 },

  {"-",     1 },
  {"+",     1 },

  {">>",    2 },
  {"<<",    2 },

  {">",     3 },
  {"<",     3 },
  {">=",    3 },
  {"<=",    3 },

  {"==",    4 },
  {"!=",    4 },

  {"&",     5 },
  {"^",     6 },
  {"|",     7 },

  {"&&",    8 },
  {"||",    9 },

  {"=",    10 },
};

/**
 *
 * @param lvalue
 * @param rvalue
 * @param op
 */
BinaryExpression::BinaryExpression(Expression* lvalue,Expression* rvalue,std::string op)
: _lvalue(lvalue)
, _rvalue(rvalue)
, _op(-1)
{
  for (size_t index = 0; index < _lengthof(opers); index++)
  {
    if (opers[index]._operation == op)
    {
      _op = index;
      break;
    }
  }

}

/**
 *
 * @param token
 * @return
 */
const BinaryExpression::Operation BinaryExpression::get_operation(std::string token)
{
  for (size_t index = 0; index < _lengthof(opers); index++)
  {
    if (token == opers[index]._operation)
      return opers[index];
  }

  return Operation(token,1000);
}

/**
 *
 * @param session
 * @return
 */
Value BinaryExpression::evaluate(Session* session)
{
  switch (_op)
  {
  case 0: //  {"*",     0 },
    _result = _lvalue->evaluate(session) * _rvalue->evaluate(session);
    break;

  case 1: //  {"/",     0 },
    _result = _lvalue->evaluate(session) / _rvalue->evaluate(session);
    break;
  
  case 2: // {"%",     0 },
    _result = _lvalue->evaluate(session) % _rvalue->evaluate(session);
    break;

  case 3: // {"-",     1 },
    _result = _lvalue->evaluate(session) - _rvalue->evaluate(session);
    break;

  case 4: // {"+",     1 },
    _result = _lvalue->evaluate(session) + _rvalue->evaluate(session);
    break;

  case 5: // {">>",    2 },
    _result = _lvalue->evaluate(session) >> _rvalue->evaluate(session);
    break;

  case 6: // {"<<",    2 },
    _result = _lvalue->evaluate(session) << _rvalue->evaluate(session);
    break;

  case 7: // {">",     3 },
    _result = _lvalue->evaluate(session) > _rvalue->evaluate(session);
    break;

  case 8: // {"<",     3 },
    _result = _lvalue->evaluate(session) < _rvalue->evaluate(session);
    break;

  case 9: // {">=",    3 },
    _result = _lvalue->evaluate(session) >= _rvalue->evaluate(session);
    break;

  case 10: // {"<=",    3 },
    _result = _lvalue->evaluate(session) <= _rvalue->evaluate(session);
    break;

  case 11: // {"==",    4 },
    _result = _lvalue->evaluate(session) == _rvalue->evaluate(session);
    break;

  case 12: // {"!=",    4 },
    _result = _lvalue->evaluate(session) != _rvalue->evaluate(session);
    break;

  case 13: // {"&",     5 },
    _result = _lvalue->evaluate(session) & _rvalue->evaluate(session);
    break;

  case 14: // {"^",     6 },
    _result = _lvalue->evaluate(session) ^ _rvalue->evaluate(session);
    break;

  case 15: // {"|",     7 },
    _result = _lvalue->evaluate(session) | _rvalue->evaluate(session);
    break;

  case 16: // {"&&",    8 },
    _result = _lvalue->evaluate(session) && _rvalue->evaluate(session);
    break;

  case 17: // {"||",    9 },
    _result = _lvalue->evaluate(session) || _rvalue->evaluate(session);
    break;

  case 18: // {"=",    10 },
    _result = _lvalue->evaluate(session) = _rvalue->evaluate(session);
    break;

  default:
    break;
  }
  return _result;
}


/*
 * \\fn Value& IndexExpression::evaluate
 *
 * created on: Jul 2, 2019
 * author: daniel
 *
 */
Value IndexExpression::evaluate(Session* session)
{
  return _value->evaluate(session).at((int)_index->evaluate(session));
}


/*
 * \\fn Value& LogicalExpression::evaluate
 *
 * created on: Jul 8, 2019
 * author: daniel
 *
 */
Value LogicalExpression::evaluate(Session* session)
{
  _result = Value();
  if (_condition != nullptr)
  {
    if ((bool)_condition->evaluate(session))
    {
      if (_positive != nullptr)
        _result = _positive->evaluate(session);
    }
    else
    {
      if (_negative != nullptr)
        _result = _negative->evaluate(session);
    }
  }

  return _result;
}

} // script
} // jupiter
} // brt

