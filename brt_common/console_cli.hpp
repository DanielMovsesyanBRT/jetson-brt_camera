/*
 * console_cli.hpp
 *
 *  Created on: Mar 17, 2020
 *      Author: daniel
 */

#ifndef BRT_COMMON_CONSOLE_CLI_HPP_
#define BRT_COMMON_CONSOLE_CLI_HPP_

#include <termios.h>
#include <stdio.h>

namespace brt
{
namespace jupiter
{

/*
 * \\class ConsoleCLI
 *
 * created on: Mar 17, 2020
 *
 */
class ConsoleCLI
{
public:
  ConsoleCLI(FILE* const = stdin);
  virtual ~ConsoleCLI();

          void                    move_to(int x,int y);

private:
          int                     activate();
          int                     release();

private:
  FILE* const                     _stream;
  termios                         _saved_state;
};

} /* namespace jupiter */
} /* namespace brt */

#endif /* BRT_COMMON_CONSOLE_CLI_HPP_ */
