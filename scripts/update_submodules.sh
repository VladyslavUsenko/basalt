#!/usr/bin/env bash

set -x

git submodule sync --recursive
git submodule update --init --recursive

