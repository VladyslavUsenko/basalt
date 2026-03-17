#!/bin/sh
set -eu

echo "Basalt now uses vcpkg manifest mode."
echo "Install build tools (cmake>=3.24, ninja, c++ compiler), bootstrap vcpkg, then use:"
echo "  cmake --preset relwithdebinfo"
