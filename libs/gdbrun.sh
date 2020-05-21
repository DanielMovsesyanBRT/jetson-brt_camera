#!/bin/bash

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
_cur_dir="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

#export LD_LIBRARY_PATH=$_cur_dir:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
if [ ! -z ${LD_LIBRARY_PATH} ]; then
  export LD_LIBRARY_PATH=$_cur_dir:$LD_LIBRARY_PATH
else
  export LD_LIBRARY_PATH=$_cur_dir
fi

echo "Executing gdbserver with LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

/usr/bin/gdbserver :2345 $@