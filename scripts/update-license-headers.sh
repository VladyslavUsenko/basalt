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

# Update license headers in source files.

# Dependency: licenseheaders python package (install with pip)

# TODO: Make it also update C++ files automatically. (Consider files with multiple headers, e.g. track.h and union_find.h)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

DIRS=(
    "$SCRIPT_DIR/../python/"
    "$SCRIPT_DIR/../scripts"
)

YEAR="2019-2021"
OWNER="Vladyslav Usenko and Nikolaus Demmel"
TEMPLATE="$SCRIPT_DIR/templates/license-py-sh.tmpl"

for d in "${DIRS[@]}"
do
    licenseheaders -d "$d" -y $YEAR -o "$OWNER" -t "$TEMPLATE" -vv
done
