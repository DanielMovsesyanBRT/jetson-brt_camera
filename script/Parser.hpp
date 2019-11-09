//
// Created by Daniel Movsesyan on 2019-04-06.
//

#ifndef EXPRESSIONS_PARSER_HPP
#define EXPRESSIONS_PARSER_HPP

#include "Expression.hpp"
#include "ParserString.hpp"
#include <string>
#include <deque>
#include <utility>

namespace brt {

namespace jupiter {

namespace script {




/**
 *
 */
class Parser
{
public:
  Parser();
  virtual ~Parser();

          Expression*             parse(const ParserString& str)
          {
            const char *const_str = str.c_str();
            return parse(const_str);
          }

          Expression*             parse(const char*& text);

  static  std::string             get_token(const char*& text);
  static  bool                    isoperator(char c);
  static  bool                    isconstant(std::string& token);
  static  bool                    isvariable(std::string& token);
  static  bool                    isfunction(std::string& token);

private:
          Expression*             get_expression(const char*& text,
                                          BinaryExpression::Operation& oper,
                                          Expression* lvalue = nullptr);


          Expression*             get_value(const char *&text);
          Expression*             get_function(const std::string& name,const char *&text);
};

std::string toupper(const char *text);

} // script
} // jupiter
} // brt

#endif //EXPRESSIONS_PARSER_HPP
