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

#include "script_action.hpp"

namespace brt {

namespace jupiter {


class ScriptPtr;
/**
 * class ScriptFile
 */
class ScriptFile : public Session
{
friend ScriptPtr;

  ScriptFile(const char *file_name);
public:
  virtual ~ScriptFile();

        bool                      load();
        bool                      is_loaded() const { return !_action_list.empty(); }

        bool                      run();
        bool                      run(const char *text);

        bool                      run_macro(const char *macro_name,script::Value& val, std::vector<script::Value> arguments = std::vector<script::Value>());
private:
  std::vector<ScriptAction*>      _action_list;
  std::string                     _file_path;
  std::atomic_bool                _busy;
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
