#!/bin/sh
set -eu

# Basalt Platform Detection Script
# Detects the current platform and returns it in standard format: {arch}-{os}
# Used by both packaging and installer scripts for consistency

# Detect operating system
detect_os() {
    _ostype="$(uname -s)"

    case "${_ostype}" in
        Linux)
            echo "unknown-linux-gnu"
            ;;
        Darwin)
            echo "apple-darwin"
            ;;
        MINGW*|MSYS*|CYGWIN*|Windows_NT)
            echo "pc-windows-gnu"
            ;;
        *)
            echo "Unsupported OS: ${_ostype}" >&2
            exit 1
            ;;
    esac
}

# Detect CPU architecture
detect_arch() {
    _cputype="$(uname -m)"
    _ostype="${1:-}"

    # On macOS, uname -m can lie due to Rosetta
    if [ "${_ostype}" = "Darwin" ]; then
        if [ "${_cputype}" = "x86_64" ] || [ "${_cputype}" = "i386" ]; then
            # Check if actually running on ARM64 with Rosetta
            if sysctl hw.optional.arm64 2> /dev/null | grep -q ': 1'; then
                _cputype="aarch64"
            fi
        fi
    fi

    case "${_cputype}" in
        x86_64|amd64|x64)
            echo "x86_64"
            ;;
        aarch64|arm64|armv8*)
            echo "aarch64"
            ;;
        armv7*|armv6*)
            echo "armv7"
            ;;
        i686|i386|x86)
            echo "i686"
            ;;
        *)
            echo "Unsupported architecture: ${_cputype}" >&2
            exit 1
            ;;
    esac
}

# Main detection
main() {
    _ostype_raw="$(uname -s)"
    _os="$(detect_os)"
    _arch="$(detect_arch "${_ostype_raw}")"

    echo "${_arch}-${_os}"
}

main "$@"
