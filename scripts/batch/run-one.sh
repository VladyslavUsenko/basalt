#!/usr/bin/env bash
##
## BSD 3-Clause License
##
## This file is part of the Basalt project.
## https://gitlab.com/VladyslavUsenko/basalt.git
##
## Copyright (c) 2019-2021, Vladyslav Usenko and Nikolaus Demmel.
## All rights reserved.
##

#
# This script runs on the slurm nodes to run rootba for one config.

set -e
set -o pipefail
set -x

error() {
  local parent_lineno="$1"
  local message="$2"
  local code="${3:-1}"
  if [[ -n "$message" ]] ; then
    echo "Error on or near line ${parent_lineno}: ${message}; exiting with status ${code}"
  else
    echo "Error on or near line ${parent_lineno}; exiting with status ${code}"
  fi
  echo "Failed" >> status.log
  exit "${code}"
}
trap 'error ${LINENO}' ERR

# number of logical cores on linux and macos
NUM_CORES=`(which nproc > /dev/null && nproc) || sysctl -n hw.logicalcpu || echo 1`

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

BASALT_BIN_DIR="${BASALT_BIN_DIR:-$SCRIPT_DIR/../../build}"

FOLDER="${1}"

cd "$FOLDER"

if ! which time 2> /dev/null; then
    echo "Did not find 'time' executable. Not installed?"
    exit 1
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    TIMECMD="`which time` -lp"
else
    TIMECMD="`which time` -v"
fi

echo "Started" >> status.log

# set environment variables according to config
while read l; do
    if [ -n "$l" ]; then
        eval "export $l"
    fi
done <<< `"$SCRIPT_DIR"/query-config.py basalt_config.json batch_run.env --format-env`

# lookup executable to run
EXECUTABLE=`"$SCRIPT_DIR"/query-config.py basalt_config.json batch_run.executable basalt_vio`

# lookup args
ARGS=`"$SCRIPT_DIR"/query-config.py basalt_config.json batch_run.args --format-cli`

CMD="$BASALT_BIN_DIR/$EXECUTABLE"

echo "Running on '`hostname`', nproc: $NUM_CORES, bin: $CMD"

# run as many times as specified (for timing tests to make sure filecache is hot); default is once
rm -f output.log
NUM_RUNS=`"$SCRIPT_DIR"/query-config.py basalt_config.json batch_run.num_runs 1`
echo "Will run $NUM_RUNS times."
for i in $(seq $NUM_RUNS); do
    echo ">>> Run $i" |& tee -a output.log
    { $TIMECMD "$CMD" $ARGS; } |& tee -a output.log
done

echo "Completed" >> status.log
