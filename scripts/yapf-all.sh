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


# Format all python source files in the project.
# Optionally take folder as argument; default are `python` and `script` dirs.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# default folders if not passed
if [ $# -lt 1 ]; then
    set -- "$SCRIPT_DIR"/../python "$SCRIPT_DIR/batch"
fi

echo "Formatting: $@"

yapf -i -r "$@"
