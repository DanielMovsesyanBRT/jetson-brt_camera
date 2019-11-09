//
// Created by Daniel Movsesyan on 2019-04-06.
//

#include "Parser.hpp"
#include "Function.hpp"
#include <ctype.h>


#undef _lengthof
#define _lengthof(x)  (sizeof(x)/sizeof(x[0]))

namespace brt {

namespace jupiter {

namespace script {

/**
 *
 * @return
 */
std::string toupper(const char *text)
{
  std::string result;
  while (*text != '\0')
  result += ::toupper(*text++);

  return result;
}

/**
 *
 */
Parser::Parser()
{
}

/**
 *
 */
Parser::~Parser()
{
}

/**
 *
 * @param text
 * @return
 */
Expression* Parser::parse(const char*& text)
{
  Expression* result = nullptr;
  Expression* val = nullptr;

  do
  {
    BinaryExpression::Operation oper;
    val = get_expression(text, oper);
    if (val != nullptr)
    {
      if (result != nullptr)
      {
        ExpressionArray* exar = dynamic_cast<ExpressionArray*>(result);
        if (exar == nullptr)
        {
          exar = new ExpressionArray;
          exar->add_expression(result);
          result = exar;
        }
        exar->add_expression(val);
      }
      else
        result =  val;
    }
  }
  while (val != nullptr);

  return result;
}

/**
 *
 * @param c
 * @return
 */
bool Parser::isoperator(char c)
{
  const char operators[] = {'*','/','%','=','>','<','!','|','&','^','~'};
  for (size_t index = 0; index < _lengthof(operators);index++)
  {
    if (operators[index] == c)
      return true;
  }
  return false;
}

/**
 *
 * @param token
 * @return
 */
bool Parser::isconstant(std::string& token)
{
  if (token.empty())
    return false;

  return isdigit(token[0]);
}

/**
 *
 * @param token
 * @return
 */
bool Parser::isvariable(std::string& token)
{
  if (token.empty())
    return false;

  // normal variables start with $
  return (token[0] == '$');
}

/*
 * \\fn bool Parser::isfunction
 *
 * created on: Jul 29, 2019
 * author: daniel
 *
 */
bool Parser::isfunction(std::string& token)
{
  if (token.empty())
    return false;

  // normal variables start with $
  return (isalpha(token[0]));
}

/**
 *
 * @param text
 * @param token
 * @return
 */

std::string Parser::get_token(const char*& text)
{
  std::string token;
  while (isspace(*text) && (*text!='\0'))
    text++;

  enum Category
  {
    eCAT_NONE,
    eCAT_VARIABLE,
    eCAT_CTRL_TOKEN,
    eCAT_NUMBER,
    eCAT_PLUS,
    eCAT_MINUS,
    eCAT_OPERATOR,
    eCAT_BRACKET,
    eCAT_QUOTE,
    eCAT_COMMA,
    eCAT_QUESTION,
    eCAT_COLON
  } category = eCAT_NONE;

  bool terminate_loop = false;
  while (!terminate_loop && !isspace(*text) && (*text!='\0'))
  {
    switch(category)
    {
    case eCAT_NONE:
      if (*text == '$')
        category = eCAT_VARIABLE;
      else if (isalpha(*text))
        category = eCAT_CTRL_TOKEN;
      else if (isdigit(*text))
        category = eCAT_NUMBER;
      else if (*text == '+')
        category = eCAT_PLUS;
      else if (*text == '-')
        category = eCAT_MINUS;
      else if (isoperator(*text))
        category = eCAT_OPERATOR;
      else if (*text=='[' || *text == '(' || *text == '[' ||
               *text==']' || *text == ')' || *text == ']')
        category = eCAT_BRACKET;
      else if (*text==',')
        category = eCAT_COMMA;
      else if (*text == '"' || *text == '\'')
        category = eCAT_QUOTE;
      else if (*text=='?')
        category = eCAT_QUESTION;
      else if (*text==':')
        category = eCAT_COLON;
      else
        terminate_loop = true;

      token = *text;
      break;

    case eCAT_VARIABLE:
    case eCAT_CTRL_TOKEN:
      if (!isalnum(*text) && (*text != '_'))
        terminate_loop = true;
      else
        token += *text;
      break;

    case eCAT_NUMBER:
      if (!isdigit(*text) &&
          ((*text < 'A') || (*text > 'F')) &&
          ((*text < 'a') || (*text > 'f')) &&
          (*text != 'x') && (*text != 'X') &&
          (*text != 's') && (*text != 'S') && // for size
          (*text != '.'))
        terminate_loop = true;
      else
        token += *text;
      break;

    case eCAT_PLUS:
      if ((*text == '+') || (*text == '='))
      {
        token += *text;
        text++;
      }
      terminate_loop = true;
      break;

    case eCAT_MINUS:
      if ((*text == '-') || (*text == '='))
      {
        token += *text;
        text++;
      }
      terminate_loop = true;
      break;

    case eCAT_OPERATOR:
      if ((*text == '=') ||
          ((*text == '<') && (token[0] == '<')) ||
          ((*text == '>') && (token[0] == '>')))
      {
        token += *text;
        text++;
      }
      terminate_loop = true;
      break;

    case eCAT_BRACKET:
    case eCAT_QUOTE:
    case eCAT_COMMA:
    case eCAT_QUESTION:
    case eCAT_COLON:
      terminate_loop = true;
      break;
    }

    if (!terminate_loop)
      text++;
  }
  return token;
}

/**
 *
 * @param text
 * @param value
 * @param oper
 * @return
 */
Expression* Parser::get_expression(const char*& text,
                                   BinaryExpression::Operation& oper,
                                   Expression* lvalue /*= nullptr*/)
{
  Expression *rvalue = get_value(text);
  if (rvalue == nullptr)
    return lvalue;

  std::string token = get_token(text);
  if ((token == "++") || ((token == "--")))
  {
    IncrDecrExpression::OP incr_oper;
    if (token == "--")
      incr_oper = IncrDecrExpression::POST_DECR;
    else// if (token == "++")
      incr_oper = IncrDecrExpression::POST_INCR;

    rvalue = new IncrDecrExpression(incr_oper, rvalue);
    token = get_token(text);
  }
  else if (token == "[")
  {
    BinaryExpression::Operation oper;
    Expression* index = get_expression(text, oper);
    if (index == nullptr)
      return nullptr;

    rvalue = new IndexExpression(rvalue,index);
    token = get_token(text);
  }
  else if (token == "?")
  {
    // Extracting Logical expression
    const char* sub_text = text;
    Expression* positive = get_value(sub_text);
    if (positive != nullptr)
      text = sub_text;

    Expression* negative = nullptr;
    token = get_token(text);
    if (token == ":")
      negative = get_value(text);

    rvalue = new LogicalExpression(rvalue,positive,negative);
    token = get_token(text);
  }

  BinaryExpression::Operation next_oper = BinaryExpression::get_operation(token);
  if (token.empty() || next_oper.noop() || (token == ")") || (token == "}") || (token == "]") || (token == ","))
  {
    rvalue = (lvalue != nullptr) ? new BinaryExpression(lvalue, rvalue, oper._operation) : rvalue;
    oper = next_oper;
    return rvalue;
  }

  while (next_oper._priority < oper._priority)
    rvalue = get_expression(text,next_oper,rvalue);

  lvalue = (lvalue != nullptr) ? new BinaryExpression(lvalue, rvalue, oper._operation) : rvalue;
  oper = next_oper;

  return lvalue;
}

/**
 *
 * @param text
 * @return
 */
Expression* Parser::get_value(const char *&text)
{
  Expression* result = nullptr;
  std::string token;

  /**
   *
   *
   * Extract LVALUE
   */
  token = get_token(text);
  if (token.empty())
    return nullptr;

  else if (isconstant(token) || (toupper(token.c_str()) == "FALSE") || (toupper(token.c_str()) == "TRUE"))
  {
    result = new Constant(token.c_str());
  }
  else if (isvariable(token))
  {
    result = new Variable(token.c_str());
  }
  else if (isfunction(token))
  {
    result = get_function(token, text);
  }
  else if (token == "(")
  {
    BinaryExpression::Operation oper;
    result = get_expression(text,oper);
  }

  else if (token == "-" || token == "!" || token == "~")
  {
    Expression* val = get_value(text);
    if (val == nullptr)
      return nullptr;

    result = new UnaryExpression(token[0],val);
  }
  else if (token == "--" || token == "++")
  {
    Expression* val = get_value(text);
    if (val == nullptr)
      return nullptr;

    IncrDecrExpression::OP oper;
    if (token == "--")
      oper = IncrDecrExpression::PRE_DECR;
    else// if (token == "++")
      oper = IncrDecrExpression::PRE_INCR;

    result = new IncrDecrExpression(oper, val);
  }
  else if (token == "'" || token == "\"" )
  {
    std::string str;
    while (*text != '"' && *text != '\'' && *text != '\0')
    {
      if (*text == '\\')
      {
        text++;
        switch (*text)
        {
        case 'n':
          str += '\n';
          break;

        case 'r':
          str += '\r';
          break;

        case 't':
          str += '\t';
          break;

        default:
          str += *text;
        }
        text++;
      }
      else
        str += *text++;
    }

    if (*text != '\0')
      text++;

    result = new Constant(str.c_str(),true);
  }
  else
    throw ParserException::create((string_format("Invalid token \"%s\"",token.c_str())).c_str());

  return result;
}

/*
 * \\fn Expression* Parser::get_function
 *
 * created on: Jul 29, 2019
 * author: daniel
 *
 */
Expression* Parser::get_function(const std::string& name, const char *&text)
{
  Function* fn = Function::create_function(name.c_str());
  if (fn == nullptr)
    throw ParserException::create((string_format("Invalid function \"%s\"",name)).c_str());

  const char *local_text = text;
  std::string token = get_token(local_text);
  if (token == "(")
  {
    text = local_text;
    do
    {
      BinaryExpression::Operation oper;
      Expression* expr = get_expression(text, oper);
      if (expr == nullptr)
        break;

      fn->add_argument(expr);
      if (oper._operation == ")")
        break;
    }
    while (true);
  }

  return fn;
}

} // script
} // jupiter
} // brt
