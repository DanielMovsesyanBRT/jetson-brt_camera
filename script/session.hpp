/*
 * session.hpp
 *
 *  Created on: Dec 11, 2019
 *      Author: daniel
 */

#ifndef SCRIPT_SESSION_HPP_
#define SCRIPT_SESSION_HPP_


#include <metadata.hpp>
#include <map>
#include <string>

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
  Session(Session* parent = nullptr): _parent(parent) {}

  virtual ~Session()
  {
    while (!_objects.empty())
    {
      delete (*_objects.begin()).second;
      _objects.erase(_objects.begin());
    }
  }

  virtual Value                   var(std::string name);// { return value(name.c_str()); }
  virtual SessionObject*&         object(std::string name);// { return _objects[name]; }

  virtual bool                    object_exist(std::string name) const { return _objects.find(name) != _objects.end(); }

private:
  std::map<std::string,SessionObject*>
                                  _objects;
  Session*                        _parent;
};


} // script
} // jupiter
} // brt

#endif /* SCRIPT_SESSION_HPP_ */
