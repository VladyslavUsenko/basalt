#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ "$OSTYPE" == "darwin"* ]]; then
${DIR}/install_mac_os_deps.sh
else
${DIR}/install_ubuntu_deps.sh
fi
