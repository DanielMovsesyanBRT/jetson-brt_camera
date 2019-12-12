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
#include <memory>

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
  virtual ~Session() {}

  virtual Value                   var(std::string name);
  virtual std::shared_ptr<SessionObject>&
                                  object(std::string name);

  virtual bool                    object_exist(std::string name) const { return _objects.find(name) != _objects.end(); }
          Session&                operator+=(const Metadata& meta) { Metadata::operator+=(meta); return *this; }
          Session&                operator+=(const Session&);

          Session*                global();

private:
  std::map<std::string,std::shared_ptr<SessionObject>>
                                  _objects;
  Session*                        _parent;
};


} // script
} // jupiter
} // brt

#endif /* SCRIPT_SESSION_HPP_ */
