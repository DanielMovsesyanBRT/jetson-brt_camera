/*
 * session.cpp
 *
 *  Created on: Dec 11, 2019
 *      Author: daniel
 */

#include "session.hpp"

namespace brt {
namespace jupiter {
namespace script {


/*
 * \\fn Value Session::var
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
Value Session::var(std::string name)
{
  // check if local variable exist
  if (exist(name.c_str()))
    return value(name.c_str());

  // Now check all parent sessions
  Session* parent = _parent;
  while(parent != nullptr)
  {
    if (parent->exist(name.c_str()))
      return parent->value(name.c_str());

    parent = parent->_parent;
  }

  // Create new value and return
  return value(name.c_str());
}


/*
 * \\fn SessionObject*& Session::object
 *
 * created on: Dec 11, 2019
 * author: daniel
 *
 */
SessionObject*& Session::object(std::string name)
{
  // Check if local object exist
  if (object_exist(name))
    return _objects[name];

  // Now check all parent sessions
  Session* parent = _parent;
  while(parent != nullptr)
  {
    if (parent->object_exist(name))
      return parent->_objects[name];

    parent = parent->_parent;
  }

  // Create new object and return
  return _objects[name];
}


} // script
} // jupiter
} // brt
