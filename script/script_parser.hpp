/*
 * script_parser.hpp
 *
 *  Created on: Dec 10, 2019
 *      Author: daniel
 */

#ifndef SCRIPT_SCRIPT_PARSER_HPP_
#define SCRIPT_SCRIPT_PARSER_HPP_

#include "parser_string.hpp"
#include <metadata.hpp>

#include <mutex>
#include <unordered_set>
#include "script.hpp"

namespace brt
{
namespace jupiter
{
namespace script
{

class ScriptParser;
/*
 * \\class ParserEnv
 *
 * created on: Jul 9, 2019
 *
 */
class ParserEnv : public ParserString
{
public:
  ParserEnv(char* buffer,ScriptParser* parser) : ParserString(buffer), _env(), _parser(parser) {}
  ParserEnv(const char* buffer,ScriptParser* parser) : ParserString(buffer), _env(), _parser(parser) {}

  virtual ~ParserEnv() {}

          Metadata&               env() { return _env; }
          ScriptParser*           parser() const { return _parser; }

private:
  Metadata                        _env;
  ScriptParser*                   _parser;
};

/*
 * \\class ScriptParser
 *
 * created on: Dec 10, 2019
 *
 */
class ScriptParser
{
public:
  ScriptParser();
  virtual ~ScriptParser();

          Script                  parse_script(const char* text,CreatorContainer = CreatorContainer());
          ScriptAction*           read_line(ParserEnv&,CreatorContainer = CreatorContainer());
};

} /* namespace script */
} /* namespace jupiter */
} /* namespace brt */

#endif /* SCRIPT_SCRIPT_PARSER_HPP_ */
