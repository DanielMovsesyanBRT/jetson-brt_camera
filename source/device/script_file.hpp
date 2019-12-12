//
// Created by daniel on 3/28/19.
//

#ifndef I2C_00_SCRIPTFILE_HPP
#define I2C_00_SCRIPTFILE_HPP

#include <stdio.h>

#include <vector>
#include <memory>
#include <string>
#include <atomic>

#include "device_action.hpp"
#include <script_parser.hpp>

namespace brt {

namespace jupiter {


class ScriptPtr;
/**
 * class ScriptFile
 */
class ScriptFile
{
friend ScriptPtr;
  ScriptFile(const char *file_name);
public:
  virtual ~ScriptFile();

        bool                      load();
        bool                      is_loaded() const { return !_script.empty(); }

        bool                      run(Metadata mt = Metadata());
        bool                      run(const char *text, Metadata mt = Metadata());

        Value                     run_macro(const char *macro_name,std::vector<Value> arguments, Metadata mt = Metadata());
private:

  /*
   * \\class ActionCreator
   *
   * created on: Dec 11, 2019
   *
   */
  class ActionCreator : public script::iActionSource
  {
  public:
    ActionCreator() {}
    virtual ~ActionCreator() {}

    virtual script::ScriptAction*   get_action(const char* action);
  };

  script::Script                  _script;

  //std::vector<DeviceAction*>      _action_list;
  std::string                     _file_path;
  std::atomic_bool                _busy;
  ActionCreator                   _ac;
};


/*
 * \\class ScriptPtr
 *
 * created on: Jul 2, 2019
 *
 */
class ScriptPtr : public std::shared_ptr<ScriptFile>
{
public:
  ScriptPtr() : std::shared_ptr<ScriptFile>() {}
  ScriptPtr(const char *file_name) : std::shared_ptr<ScriptFile>(new ScriptFile(file_name)) {}
};

} // jupiter
} // brt


#endif //I2C_00_SCRIPTFILE_HPP
