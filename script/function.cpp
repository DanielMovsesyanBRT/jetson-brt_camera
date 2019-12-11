/*
 * Function.cpp
 *
 *  Created on: Jul 29, 2019
 *      Author: daniel
 */

#include "function.hpp"

#include <string.h>
#include <iostream>
#include <iomanip>
#include "parser_string.hpp"

namespace brt
{
namespace jupiter
{
namespace script
{


/*
 * \\fn Function* Function::create_function
 *
 * created on: Jul 29, 2019
 * author: daniel
 *
 */
Function* Function::create_function(const char* name)
{
  if (strcmp(name,"bool") == 0)
    return new FunctionConvertToBool;

  else if (strcmp(name,"int") == 0)
    return new FunctionConvertToInt;

  else if (strcmp(name,"real") == 0)
    return new FunctionConvertToReal;

  else if (strcmp(name,"str") == 0)
    return new FunctionConvertToString;

  else if (strcmp(name,"buf") == 0)
    return new FunctionConvertToBuff;

  else if (strcmp(name,"dec") == 0)
    return new FunctionDec;

  else if (strcmp(name,"hex") == 0)
    return new FunctionHex;

  else if (strcmp(name,"sub_array") == 0)
    return new FunctionSubArray;

  return nullptr;
}

/*
 * \\fn Value& FunctionConvertToBool::evaluate
 *
 * created on: Jul 29, 2019
 * author: daniel
 *
 */
Value FunctionConvertToBool::evaluate(Session* session)
{
  if (num_args() > 0)
  {
    Value argument = arg(0,session);
    _result.set((bool)argument,argument.size());
  }

  return _result;
}

/*
 * \\fn Value& FunctionConvertToInt::evaluate
 *
 * created on: Jul 29, 2019
 * author: daniel
 *
 */
Value FunctionConvertToInt::evaluate(Session* session)
{
  if (num_args() > 0)
  {
    Value argument = arg(0,session);
    size_t size = argument.size();
    if (num_args() >= 2)
      size = (int)arg(1,session);

    _result.set((int)argument, size);
  }

  return _result;
}

/*
 * \\fn Value& FunctionConvertToReal::evaluate
 *
 * created on: Jul 29, 2019
 * author: daniel
 *
 */
Value FunctionConvertToReal::evaluate(Session* session)
{
  if (num_args() > 0)
  {
    Value argument = arg(0,session);
    _result.set((double)argument,argument.size());
  }

  return _result;
}

/*
 * \\fn Value& FunctionConvertToString::evaluate
 *
 * created on: Jul 29, 2019
 * author: daniel
 *
 */
Value FunctionConvertToString::evaluate(Session* session)
{
  if (num_args() > 0)
    _result.set( ((std::string)arg(0,session)).c_str());

  return _result;
}

/*
 * \\fn Value& FunctionConvertToBuff::evaluate
 *
 * created on: Jul 29, 2019
 * author: daniel
 *
 */
Value FunctionConvertToBuff::evaluate(Session* session)
{
  if (num_args() > 0)
  {
    Value::byte_buffer buff = arg(0,session);
    _result.set(buff);
  }

  return _result;
}

/*
 * \\fn Value& FunctionDec::evaluate
 *
 * created on: Jul 29, 2019
 * author: daniel
 *
 */
Value FunctionDec::evaluate(Session* session)
{
  if (num_args() > 0)
  {
    int val = arg(0,session);
    std::stringstream stream;
    stream << std::dec << std::uppercase << val;
    _result.set(stream.str().c_str());
  }

  return _result;
}

/*
 * \\fn Value& FunctionHex::evaluate
 *
 * created on: Jul 29, 2019
 * author: daniel
 *
 */
Value FunctionHex::evaluate(Session* session)
{
  if (num_args() > 0)
  {
    int val = arg(0,session);
    std::stringstream stream;
    stream << std::hex << std::uppercase << val;
    _result.set(stream.str().c_str());
  }

  return _result;
}

/*
 * \\fn Value FunctionSubArray::evaluate
 *
 * created on: Jul 31, 2019
 * author: daniel
 *
 */
Value FunctionSubArray::evaluate(Session* session)
{
  if (num_args() > 0)
  {
    Value buffer = arg(0,session);
    int start = 0;
    int length = static_cast<int>(buffer.size());
    if (num_args() >= 2)
      start = (int)arg(1,session);

    if (num_args() >= 3)
      length = (int)arg(2,session);

    _result = buffer.sub_array(start, length);
  }

  return _result;
}


} /* namespace script */
} /* namespace jupiter */
} /* namespace brt */
