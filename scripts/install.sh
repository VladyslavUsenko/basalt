#!/bin/sh
set -eu

# Basalt Installer Script
# Usage: curl -LsSf https://gitlab.com/VladyslavUsenko/basalt/-/raw/master/scripts/install.sh | sh
# Or with a specific version: curl -LsSf .../install.sh | sh -s -- <version>

APP_NAME="basalt"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# GitLab instance configuration
if [ -n "${CI_SERVER_URL:-}" ]; then
    GITLAB_URL="${CI_SERVER_URL}"
    PROJECT="${CI_PROJECT_PATH:-${CI_PROJECT_NAME:-basalt}}"
elif [ -n "${BASALT_GITLAB_URL:-}" ]; then
    GITLAB_URL="${BASALT_GITLAB_URL}"
    PROJECT="${BASALT_PROJECT:-VladyslavUsenko/basalt}"
else
    GITLAB_URL="https://gitlab.com"
    PROJECT="VladyslavUsenko/basalt"
fi

DOWNLOAD_URL="${BASALT_DOWNLOAD_URL:-}"
ARTIFACTS_DIR="${BASALT_ARTIFACTS_DIR:-${SCRIPT_DIR%/scripts}/artifacts}"
PROJECT_API_PATH="$(printf '%s' "${PROJECT}" | sed 's/\//%2F/g')"
BASE_URL="${GITLAB_URL}/api/v4/projects/${PROJECT_API_PATH}/releases"

INSTALL_FROM_ARTIFACTS=0
VERSION_ARG=""

log() {
    echo "$*"
}

warn() {
    echo "Warning: $*" >&2
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

print_help() {
    cat <<'EOF'
Basalt installer

Usage:
  scripts/install.sh [<version>]
  scripts/install.sh --from-artifacts
  scripts/install.sh --print-build-deps

Modes:
  Default               Download a release archive from GitLab and install it to ~/.local
  --from-artifacts      Install from a local artifacts/basalt-*.tar.gz archive
  --print-build-deps    Print build dependency guidance and exit
EOF
}

print_build_deps() {
    cat <<'EOF'
Basalt now uses vcpkg manifest mode.
Install build tools (cmake>=3.24, ninja, c++ compiler), bootstrap vcpkg, then use:
  cmake --preset relwithdebinfo
EOF
}

parse_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --from-artifacts)
                INSTALL_FROM_ARTIFACTS=1
                ;;
            --print-build-deps|--build-help)
                print_build_deps
                exit 0
                ;;
            -h|--help)
                print_help
                exit 0
                ;;
            --)
                shift
                break
                ;;
            -*)
                die "Unknown option: $1"
                ;;
            *)
                if [ -n "${VERSION_ARG}" ]; then
                    die "Unexpected extra argument: $1"
                fi
                VERSION_ARG="$1"
                ;;
        esac
        shift
    done

    if [ "$#" -gt 0 ]; then
        if [ -n "${VERSION_ARG}" ]; then
            die "Unexpected extra argument: $1"
        fi
        VERSION_ARG="$1"
        shift
    fi

    if [ "$#" -gt 0 ]; then
        die "Unexpected arguments: $*"
    fi

    if [ "${INSTALL_FROM_ARTIFACTS}" -eq 1 ] && [ -n "${VERSION_ARG}" ]; then
        die "--from-artifacts cannot be combined with a release version argument"
    fi
}

detect_platform() {
    if [ -f "${SCRIPT_DIR}/detect_platform.sh" ]; then
        sh "${SCRIPT_DIR}/detect_platform.sh"
        return
    fi

    _ostype="$(uname -s)"
    _cputype="$(uname -m)"

    case "${_ostype}" in
        Linux)
            _ostype="unknown-linux-gnu"
            ;;
        Darwin)
            _ostype="apple-darwin"
            if [ "${_cputype}" = "x86_64" ] || [ "${_cputype}" = "i386" ]; then
                if sysctl hw.optional.arm64 2>/dev/null | grep -q ': 1'; then
                    _cputype="aarch64"
                fi
            fi
            ;;
        *)
            die "Unsupported OS: ${_ostype}"
            ;;
    esac

    case "${_cputype}" in
        x86_64|amd64|x64)
            _cputype="x86_64"
            ;;
        aarch64|arm64|armv8*)
            _cputype="aarch64"
            ;;
        *)
            die "Unsupported architecture: ${_cputype}"
            ;;
    esac

    echo "${_cputype}-${_ostype}"
}

get_latest_version() {
    _latest="$(curl -s "${BASE_URL}/per/latest" 2>/dev/null | grep -o '"tag_name":"[^"]*"' | cut -d'"' -f4)"

    if [ -n "${_latest}" ]; then
        echo "${_latest}"
        return 0
    fi

    curl -s "${BASE_URL}" 2>/dev/null | grep -o '"tag_name":"[^"]*"' | head -n1 | cut -d'"' -f4
}

artifact_url() {
    _version="$1"
    _artifact="$2"

    echo "${GITLAB_URL}/${PROJECT}/-/releases/${_version}/downloads/${_artifact}"
}

download_checksum() {
    _version="$1"
    _artifact="$2"
    _checksum_url="$(artifact_url "${_version}" "${_artifact}.sha256")"

    log "Downloading checksum from ${_checksum_url}..."
    if ! curl -sSLf "${_checksum_url}" -o "${_artifact}.sha256" 2>/dev/null; then
        warn "Could not download checksum file"
        return 1
    fi

    return 0
}

checksum_cmd() {
    if command -v sha256sum >/dev/null 2>&1; then
        echo "sha256sum"
    elif command -v shasum >/dev/null 2>&1; then
        echo "shasum -a 256"
    else
        return 1
    fi
}

verify_checksum() {
    _file="$1"
    _expected="$2"

    if [ -z "${_expected}" ]; then
        warn "No checksum available, skipping verification"
        return 0
    fi

    if ! _checksum_cmd="$(checksum_cmd)"; then
        warn "No SHA256 tool found, skipping checksum verification"
        return 0
    fi

    _actual="$(${_checksum_cmd} "${_file}" | awk '{print $1}')"

    if [ "${_actual}" != "${_expected}" ]; then
        die "Checksum mismatch for ${_file}
Expected: ${_expected}
Got:      ${_actual}"
    fi

    log "Checksum verified: ${_file}"
}

find_release_artifact() {
    _arch="$(detect_platform)"
    _matches=""

    for _artifact in "${ARTIFACTS_DIR}"/basalt-*-"${_arch}".tar.gz; do
        if [ -f "${_artifact}" ]; then
            if [ -n "${_matches}" ]; then
                die "Expected exactly one release artifact for ${_arch} under ${ARTIFACTS_DIR}/"
            fi
            _matches="${_artifact}"
        fi
    done

    [ -n "${_matches}" ] || die "No release artifact found for ${_arch} under ${ARTIFACTS_DIR}/"
    printf '%s\n' "${_matches}"
}

run_privileged() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        return 1
    fi
}

check_linux_runtime_deps() {
    _missing=""

    if command -v dpkg >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
        for _pkg in libegl1 libgl1 libglu1-mesa libx11-6 libxcursor1 libxinerama1 libxrandr2 libxi6 libxtst6; do
            if ! dpkg -s "${_pkg}" >/dev/null 2>&1; then
                _missing="${_missing} ${_pkg}"
            fi
        done
        printf '%s\n' "${_missing}"
        return 0
    fi

    if command -v ldd >/dev/null 2>&1; then
        printf '%s\n' ""
        return 0
    fi

    warn "Unable to verify Linux runtime dependencies automatically on this system"
    printf '%s\n' ""
}

ensure_runtime_deps() {
    case "$(uname -s)" in
        Linux)
            _missing="$(check_linux_runtime_deps)"
            if [ -z "${_missing}" ]; then
                log "Runtime dependencies already installed."
                return 0
            fi

            log "Installing missing runtime dependencies:${_missing}"
            run_privileged apt-get update -qq || die "Failed to update package index for runtime dependencies"
            # shellcheck disable=SC2086
            run_privileged apt-get install -y -qq $_missing || die "Failed to install runtime dependencies:${_missing}"
            ;;
        Darwin)
            log "Runtime dependency auto-install check is not required on macOS."
            ;;
        *)
            warn "Runtime dependency auto-install check is not implemented for $(uname -s)"
            ;;
    esac
}

resolve_archive_source() {
    _tmp_dir="$1"
    _version="$2"

    cd "${_tmp_dir}"

    if [ "${INSTALL_FROM_ARTIFACTS}" -eq 1 ]; then
        _artifact_path="$(find_release_artifact)"
        log "Using local artifact ${_artifact_path}"
        cp "${_artifact_path}" basalt.tar.gz
        return 0
    fi

    if [ -n "${DOWNLOAD_URL}" ]; then
        _url="${DOWNLOAD_URL}"
        _artifact="basalt.tar.gz"
    else
        _arch="$(detect_platform)"
        _artifact="basalt-${_version}-${_arch}.tar.gz"
        _url="$(artifact_url "${_version}" "${_artifact}")"
    fi

    log "Downloading from ${_url} ..."
    if ! curl -sSLf "${_url}" -o basalt.tar.gz; then
        case "$(detect_platform)" in
            *apple-darwin)
                die "No prebuilt ${APP_NAME} release is available for $(detect_platform).
Build from source on macOS or provide BASALT_DOWNLOAD_URL with a compatible archive."
                ;;
        esac
        die "Failed to download ${_artifact}
Please verify that the version exists for your platform at:
${GITLAB_URL}/${PROJECT}/-/releases"
    fi

    if [ -z "${DOWNLOAD_URL}" ]; then
        download_checksum "${_version}" "${_artifact}" || true
        if [ -f "${_artifact}.sha256" ]; then
            _expected_checksum="$(awk '{print $1}' < "${_artifact}.sha256")"
            verify_checksum "basalt.tar.gz" "${_expected_checksum}"
        fi
    fi
}

extract_release_archive() {
    _tmp_dir="$1"
    cd "${_tmp_dir}"

    log "Extracting..."
    tar xzf basalt.tar.gz

    [ -d "release" ] || die "Invalid archive structure (expected 'release' directory)"
    [ -d "release/bin" ] || die "Invalid archive structure (missing 'release/bin')"
    [ -d "release/data" ] || die "Invalid archive structure (missing 'release/data')"
    [ -d "release/lib" ] || die "Invalid archive structure (missing 'release/lib')"
}

get_rc_file() {
    _shell="$(basename "${SHELL:-/bin/sh}")"

    case "$_shell" in
        bash)
            if [ -f "${HOME}/.bashrc" ]; then
                echo "${HOME}/.bashrc"
            else
                echo "${HOME}/.profile"
            fi
            ;;
        zsh)
            echo "${HOME}/.zshrc"
            ;;
        fish)
            echo "${HOME}/.config/fish/config.fish"
            ;;
        *)
            echo "${HOME}/.profile"
            ;;
    esac
}

get_env_script_path() {
    case "$(basename "${SHELL:-/bin/sh}")" in
        fish)
            echo "${HOME}/.basalt/env.fish"
            ;;
        *)
            echo "${HOME}/.basalt/env"
            ;;
    esac
}

get_env_source_line() {
    _env_script_path="$1"

    case "$(basename "${SHELL:-/bin/sh}")" in
        fish)
            echo "source \"${_env_script_path}\""
            ;;
        *)
            echo ". \"${_env_script_path}\""
            ;;
    esac
}

has_env_source_line() {
    _rc_file="$1"
    _source_line="$2"

    if [ -f "$_rc_file" ]; then
        grep -F "$_source_line" "$_rc_file" >/dev/null 2>/dev/null
    else
        return 1
    fi
}

write_env_script() {
    _env_script_path="$1"

    mkdir -p "$(dirname "$_env_script_path")"

    case "$(basename "${SHELL:-/bin/sh}")" in
        fish)
            cat > "$_env_script_path" <<'EOF'
# Basalt installer
if not contains $HOME/.local/bin $fish_user_paths
    set -a fish_user_paths $HOME/.local/bin
end
if test (uname) = Darwin
    if not set -q DYLD_LIBRARY_PATH
        set -gx DYLD_LIBRARY_PATH $HOME/.local/lib
    else if not contains $HOME/.local/lib $DYLD_LIBRARY_PATH
        set -gx DYLD_LIBRARY_PATH $HOME/.local/lib $DYLD_LIBRARY_PATH
    end
else
    if not set -q LD_LIBRARY_PATH
        set -gx LD_LIBRARY_PATH $HOME/.local/lib
    else if not contains $HOME/.local/lib $LD_LIBRARY_PATH
        set -gx LD_LIBRARY_PATH $HOME/.local/lib $LD_LIBRARY_PATH
    end
end
EOF
            ;;
        *)
            cat > "$_env_script_path" <<'EOF'
# Basalt installer
case ":${PATH}:" in
    *:"$HOME/.local/bin":*) ;;
    *) PATH="$HOME/.local/bin:$PATH" ;;
esac
export PATH

if [ "$(uname)" = "Darwin" ]; then
    case ":${DYLD_LIBRARY_PATH:-}:" in
        *:"$HOME/.local/lib":*) ;;
        *)
            if [ -n "${DYLD_LIBRARY_PATH:-}" ]; then
                DYLD_LIBRARY_PATH="$HOME/.local/lib:$DYLD_LIBRARY_PATH"
            else
                DYLD_LIBRARY_PATH="$HOME/.local/lib"
            fi
            ;;
    esac
    export DYLD_LIBRARY_PATH
else
    case ":${LD_LIBRARY_PATH:-}:" in
        *:"$HOME/.local/lib":*) ;;
        *)
            if [ -n "${LD_LIBRARY_PATH:-}" ]; then
                LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
            else
                LD_LIBRARY_PATH="$HOME/.local/lib"
            fi
            ;;
    esac
    export LD_LIBRARY_PATH
fi
EOF
            ;;
    esac

    chmod +x "$_env_script_path" 2>/dev/null || true
}

add_env_source_line() {
    _rc_file="$1"
    _source_line="$2"

    mkdir -p "$(dirname "$_rc_file")"
    echo "" >> "$_rc_file"
    echo "# Basalt installer" >> "$_rc_file"
    echo "$_source_line" >> "$_rc_file"
}

install_release_tree() {
    _tmp_dir="$1"
    _install_dir="${HOME}/.local/bin"
    _lib_dir="${HOME}/.local/lib"
    _data_dir="${HOME}/.local/etc/basalt"

    mkdir -p "${_install_dir}" "${_lib_dir}" "${_data_dir}"

    log "Installing binaries to ${_install_dir} ..."
    cp -f "${_tmp_dir}"/release/bin/* "${_install_dir}/"
    chmod +x "${_install_dir}"/basalt_* 2>/dev/null || true

    log "Installing library to ${_lib_dir} ..."
    cp -f "${_tmp_dir}"/release/lib/* "${_lib_dir}/"

    if [ "$(uname)" = "Darwin" ]; then
        for _lib in "${_tmp_dir}"/release/lib/*.dylib; do
            if [ -f "${_lib}" ]; then
                _libname="$(basename "${_lib}")"
                install_name_tool -id "${_lib_dir}/${_libname}" "${_lib_dir}/${_libname}" 2>/dev/null || true
            fi
        done
    fi

    log "Installing data files to ${_data_dir} ..."
    cp -rf "${_tmp_dir}"/release/data/* "${_data_dir}/"
}

write_receipt() {
    _version="$1"
    _source="$2"
    _receipt_dir="${HOME}/.basalt"
    _receipt_file="${_receipt_dir}/install.json"

    mkdir -p "${_receipt_dir}"
    cat > "${_receipt_file}" <<EOF
{
  "version": "${_version}",
  "arch": "$(detect_platform)",
  "install_dir": "${HOME}/.local",
  "source": "${_source}",
  "installed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

    printf '%s\n' "${_receipt_file}"
}

update_shell_environment() {
    _install_dir="${HOME}/.local/bin"
    _lib_dir="${HOME}/.local/lib"
    _rc_file="$(get_rc_file)"
    _env_script_path="$(get_env_script_path)"
    _source_line="$(get_env_source_line "${_env_script_path}")"
    _library_env_var="LD_LIBRARY_PATH"

    if [ "$(uname)" = "Darwin" ]; then
        _library_env_var="DYLD_LIBRARY_PATH"
    fi

    write_env_script "${_env_script_path}"

    _rc_updated="false"
    if ! has_env_source_line "${_rc_file}" "${_source_line}"; then
        echo ""
        log "Adding ${_install_dir} to PATH and ${_lib_dir} to ${_library_env_var} in ${_rc_file} ..."
        add_env_source_line "${_rc_file}" "${_source_line}"
        _rc_updated="true"
    fi

    _sourced_rc="false"
    case "$(basename "${SHELL:-/bin/sh}")" in
        bash|zsh)
            # shellcheck disable=SC1090
            . "${_env_script_path}"
            _sourced_rc="true"
            ;;
    esac

    _manual_source_cmd="source ${_env_script_path}"
    if [ "$(basename "${SHELL:-/bin/sh}")" != "fish" ]; then
        _manual_source_cmd=". ${_env_script_path}"
    fi

    echo ""
    if [ "${_sourced_rc}" = "true" ]; then
        echo "Updated environment loaded from ${_env_script_path} for this shell session."
        echo ""
        echo "If needed later, either:"
        echo "  1. Run: ${_manual_source_cmd}"
        echo "  2. Or restart your shell"
    elif [ "${_rc_updated}" = "true" ]; then
        echo "To complete the installation, either:"
        echo "  1. Run: ${_manual_source_cmd}"
        echo "  2. Or restart your shell"
    else
        echo "You can now run basalt binaries from your shell."
    fi
}

main() {
    parse_args "$@"

    _version="${VERSION_ARG}"
    _source="release"
    _tmp_dir="$(mktemp -d)"

    trap 'if [ -n "${_tmp_dir:-}" ] && [ -d "${_tmp_dir}" ]; then rm -rf "${_tmp_dir}"; fi' EXIT INT TERM

    if [ "${INSTALL_FROM_ARTIFACTS}" -eq 0 ]; then
        if [ -z "${_version}" ]; then
            _version="$(get_latest_version)"
            [ -n "${_version}" ] || die "Failed to resolve the latest release version"
            log "No version specified, using latest: ${_version}"
        else
            log "Installing ${APP_NAME} ${_version}..."
        fi
        log "Installing ${APP_NAME} ${_version} for $(detect_platform)..."
    else
        _source="artifacts"
        _version="local-artifact"
        log "Installing ${APP_NAME} from local artifact for $(detect_platform)..."
    fi

    resolve_archive_source "${_tmp_dir}" "${_version}"
    extract_release_archive "${_tmp_dir}"
    ensure_runtime_deps
    install_release_tree "${_tmp_dir}"
    _receipt_file="$(write_receipt "${_version}" "${_source}")"

    cd "${HOME}"
    rm -rf "${_tmp_dir}"
    _tmp_dir=""

    update_shell_environment

    echo ""
    echo "Basalt ${_version} installed successfully!"
    echo ""
    echo "Binaries:  ${HOME}/.local/bin"
    echo "Library:   ${HOME}/.local/lib"
    echo "Data:      ${HOME}/.local/etc/basalt"
    echo "Receipt:   ${_receipt_file}"
}

main "$@"
