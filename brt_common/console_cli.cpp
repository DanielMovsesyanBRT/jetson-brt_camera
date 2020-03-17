/*
 * console_cli.cpp
 *
 *  Created on: Mar 17, 2020
 *      Author: daniel
 */
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>

#include <iostream>

#include "console_cli.hpp"

namespace brt
{
namespace jupiter
{

/*
 * \\fn Constructor ConsoleCLI::ConsoleCLI
 *
 * created on: Mar 17, 2020
 * author: daniel
 *
 */
ConsoleCLI::ConsoleCLI(FILE* const stream /*= stdin*/)
: _stream(stream)
{
  activate();
}

/*
 * \\fn Destructor ConsoleCLI::~ConsoleCLI
 *
 * created on: Mar 17, 2020
 * author: daniel
 *
 */
ConsoleCLI::~ConsoleCLI()
{
  release();
}

/*
 * \\fn int ConsoleCLI::activate
 *
 * created on: Mar 17, 2020
 * author: daniel
 *
 *  Call this to change the terminal related to the stream to "raw" state.
 * (Usually you call this with stdin).
 * This means you get all keystrokes, and special keypresses like CTRL-C
 * no longer generate interrupts.
 *
 * You must restore the state before your program exits, or your user will
 * frantically have to figure out how to type 'reset' blind, to get their terminal
 * back to a sane state.
 *
 * The function returns 0 if success, errno error code otherwise.
 */
int ConsoleCLI::activate()
{
  struct termios raw, actual;
  int fd;

  if (_stream == nullptr)
    return errno = EINVAL;

  /* Tell the C library not to buffer any data from/to the stream. */
  if (setvbuf(_stream, nullptr, _IONBF, 0))
    return errno = EIO;

  /* Write/discard already buffered data in the stream. */
  fflush (_stream);

  /* Termios functions use the file descriptor. */
  fd = fileno(_stream);
  if (fd == -1)
    return errno = EINVAL;

  /* Discard all unread input and untransmitted output. */
  tcflush(fd, TCIOFLUSH);

  /* Get current terminal settings. */
  if (tcgetattr(fd, &_saved_state))
    return errno;

  /* New terminal settings are based on current ones. */
  raw = _saved_state;

  /* Because the terminal needs to be restored to the original state,
   * you want to ignore CTRL-C (break). */
  raw.c_iflag |= IGNBRK; /* do ignore break, */
  raw.c_iflag &= ~BRKINT; /* do not generate INT signal at break. */

  /* Make sure we are enabled to receive data. */
  raw.c_cflag |= CREAD;

  /* Do not generate signals from special keypresses. */
  raw.c_lflag &= ~ISIG;

  /* Do not echo characters. */
  raw.c_lflag &= ~ECHO;

  /* Most importantly, disable "canonical" mode. */
  raw.c_lflag &= ~ICANON;

  /* In non-canonical mode, we can set whether getc() returns immediately
   * when there is no data, or whether it waits until there is data.
   * You can even set the wait timeout in tenths of a second.
   * This sets indefinite wait mode. */
  raw.c_cc[VMIN] = 1; /* Wait until at least one character available, */
  raw.c_cc[VTIME] = 0; /* Wait indefinitely. */

  /* Set the new terminal settings. */
  if (tcsetattr(fd, TCSAFLUSH, &raw))
    return errno;

  /* tcsetattr() is happy even if it did not set *all* settings.
   * We need to verify. */
  if (tcgetattr(fd, &actual))
  {
    const int saved_errno = errno;
    /* Try restoring the old settings! */
    tcsetattr(fd, TCSANOW, &_saved_state);
    return errno = saved_errno;
  }

  if (actual.c_iflag != raw.c_iflag || actual.c_oflag != raw.c_oflag
      || actual.c_cflag != raw.c_cflag || actual.c_lflag != raw.c_lflag)
  {
    /* Try restoring the old settings! */
    tcsetattr(fd, TCSANOW, &_saved_state);
    return errno = EIO;
  }

  /* Success! */
  return 0;
}

/*
 * \\fn int ConsoleCLI::release
 *
 * created on: Mar 17, 2020
 * author: daniel
 *
 */
int ConsoleCLI::release()
{
  int fd, result;

  if (_stream == nullptr)
    return errno = EINVAL;

  /* Write/discard all buffered data in the stream. Ignores errors. */
  fflush (_stream);

  /* Termios functions use the file descriptor. */
  fd = fileno(_stream);
  if (fd == -1)
    return errno = EINVAL;

  /* Discard all unread input and untransmitted output. */
  do
  {
    result = tcflush(fd, TCIOFLUSH);
  }
  while (result == -1 && errno == EINTR);

  /* Restore terminal state. */
  do
  {
    result = tcsetattr(fd, TCSAFLUSH, &_saved_state);
  }
  while (result == -1 && errno == EINTR);
  if (result == -1)
    return errno;

  /* Success. */
  return 0;
}

/*
 * \\class ConsoleCLI::move_to
 *
 * created on: Mar 17, 2020
 *
 */
void ConsoleCLI::move_to(int x,int y)
{
  if (x < 0)
    std::cout << "\033[" << -x << "D";
  else if (x > 0)
    std::cout << "\033[" << x << "C";

  if (y < 0)
    std::cout << "\033[" << -y << "B";
  else if (y > 0)
    std::cout << "\033[" << y << "A";
}

} /* namespace jupiter */
} /* namespace brt */
