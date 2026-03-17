#!/bin/bash
set -euo pipefail

find_release_artifact() {
  shopt -s nullglob
  local artifacts=(artifacts/basalt-*.tar.gz)
  shopt -u nullglob

  if [ "${#artifacts[@]}" -eq 0 ]; then
    echo "No release artifact found under artifacts/" >&2
    exit 1
  fi

  if [ "${#artifacts[@]}" -gt 1 ]; then
    echo "Expected exactly one release artifact, found ${#artifacts[@]}:" >&2
    printf '  %s\n' "${artifacts[@]}" >&2
    exit 1
  fi

  printf '%s\n' "${artifacts[0]}"
}

# Install runtime dependencies required by basalt_vio (GUI applications)
apt-get update -qq
apt-get install -y -qq libegl1 libgl1 libglu1 libx11-6 libxcursor1 libxinerama1 libxrandr2 libxi6 libxtst6

# Extract release tarball
artifact="$(find_release_artifact)"
tar -xzf "${artifact}"

if [ ! -d release/bin ] || [ ! -d release/data ]; then
  echo "Invalid release archive layout in ${artifact}" >&2
  exit 1
fi

# Install executables to standard location
mkdir -p /usr/local/bin /usr/local/lib /usr/etc/basalt
install -m 0755 release/bin/* /usr/local/bin/

# Install shared library to standard location
install -m 0755 release/lib/libbasalt.so* /usr/local/lib/
ldconfig

# Install data files
cp -r release/data/* /usr/etc/basalt/
