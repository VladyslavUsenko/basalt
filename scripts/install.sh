#!/bin/sh
set -eu

# Basalt Installer Script
# Usage: curl -LsSf https://gitlab.com/VladyslavUsenko/basalt/-/raw/master/scripts/install.sh | sh
# Or with a specific version: curl -LsSf .../install.sh | sh -s -- <version>

APP_NAME="basalt"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# GitLab instance configuration
# Use CI predefined variables when available, otherwise allow user override
if [ -n "${CI_SERVER_URL:-}" ]; then
    # Running in GitLab CI - use predefined variables
    GITLAB_URL="${CI_SERVER_URL}"
    PROJECT="${CI_PROJECT_PATH:-${CI_PROJECT_NAME:-basalt}}"
elif [ -n "${BASALT_GITLAB_URL:-}" ]; then
    # User override
    GITLAB_URL="${BASALT_GITLAB_URL}"
    PROJECT="${BASALT_PROJECT:-VladyslavUsenko/basalt}"
else
    # Default for manual installation
    GITLAB_URL="https://gitlab.com"
    PROJECT="VladyslavUsenko/basalt"
fi

DOWNLOAD_URL="${BASALT_DOWNLOAD_URL:-}"
ARTIFACT_JOB_NAME="${BASALT_ARTIFACT_JOB_NAME:-ubuntu22-build}"
BASE_URL="${GITLAB_URL}/api/v4/projects/${PROJECT}/releases"

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

# Get latest version from GitLab API
get_latest_version() {
    _latest="$(curl -s "${BASE_URL}/per/latest" 2>/dev/null | grep -o '"tag_name":"[^"]*"' | cut -d'"' -f4)"

    if [ -n "${_latest}" ]; then
        echo "${_latest}"
        return 0
    fi

    # GitLab's /per/latest endpoint can return 404 even when releases exist.
    curl -s "${BASE_URL}" 2>/dev/null | grep -o '"tag_name":"[^"]*"' | head -n1 | cut -d'"' -f4
}

artifact_url() {
    _version="$1"
    _artifact="$2"

    echo "${GITLAB_URL}/${PROJECT}/-/jobs/artifacts/${_version}/raw/artifacts/${_artifact}?job=${ARTIFACT_JOB_NAME}"
}

# Download checksum file
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

# Verify checksum
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
    return 0
}

# Get shell rc file
get_rc_file() {
    _shell="$(basename "${SHELL:-/bin/sh}")"

    case "$_shell" in
        bash)
            # Prefer .bashrc for interactive shells, fallback to .profile
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

# Check if PATH update is needed
needs_path_update() {
    _rc_file="$1"

    case "$(basename "${SHELL:-/bin/sh}")" in
        fish)
            if [ -f "$_rc_file" ]; then
                ! grep -q "fish_user_paths" "$_rc_file" 2>/dev/null || ! grep -q "\$HOME/.local/bin" "$_rc_file" 2>/dev/null
            else
                true
            fi
            ;;
        *)
            if [ -f "$_rc_file" ]; then
                ! grep -q "\$HOME/.local/bin" "$_rc_file" 2>/dev/null
            else
                true
            fi
            ;;
    esac
}

# Update PATH in shell rc
update_path() {
    _rc_file="$1"

    mkdir -p "$(dirname "$_rc_file")"

    case "$(basename "${SHELL:-/bin/sh}")" in
        fish)
            echo "" >> "$_rc_file"
            echo "# Basalt installer" >> "$_rc_file"
            echo "if not contains \$HOME/.local/bin \$fish_user_paths" >> "$_rc_file"
            echo "    set -a fish_user_paths \$HOME/.local/bin" >> "$_rc_file"
            echo "end" >> "$_rc_file"
            ;;
        *)
            echo "" >> "$_rc_file"
            echo "# Basalt installer" >> "$_rc_file"
            echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$_rc_file"
            ;;
    esac
}

# Main installation
main() {
    _version="${1:-}"
    _arch="$(detect_platform)"
    _tmp_dir=""

    if [ -z "$_version" ]; then
        _version="$(get_latest_version)"
        [ -n "${_version}" ] || die "Failed to resolve the latest release version"
        log "No version specified, using latest: $_version"
    else
        log "Installing basalt ${_version}..."
    fi

    _install_dir="${HOME}/.local/bin"
    _lib_dir="${HOME}/.local/lib"
    _data_dir="${HOME}/.local/etc/basalt"
    _receipt_dir="${HOME}/.basalt"

    log "Installing ${APP_NAME} ${_version} for ${_arch}..."

    # Create directories
    mkdir -p "$_install_dir"
    mkdir -p "$_lib_dir"
    mkdir -p "$_data_dir"
    mkdir -p "$_receipt_dir"

    # Download to temporary directory
    _tmp_dir="$(mktemp -d)"
    trap 'if [ -n "${_tmp_dir}" ] && [ -d "${_tmp_dir}" ]; then rm -rf "${_tmp_dir}"; fi' EXIT INT TERM
    cd "$_tmp_dir"

    # Construct artifact name based on platform
    if [ -n "${DOWNLOAD_URL}" ]; then
        _url="${DOWNLOAD_URL}"
        _artifact="basalt.tar.gz"
    else
        _artifact="basalt-${_version}-${_arch}.tar.gz"
        _url="$(artifact_url "${_version}" "${_artifact}")"
    fi

    log "Downloading from $_url ..."
    if ! curl -sSLf "$_url" -o basalt.tar.gz; then
        case "${_arch}" in
            *apple-darwin)
                die "No prebuilt ${APP_NAME} release is available for ${_arch}.
Current GitLab releases publish Linux artifacts only.
Build from source on macOS or provide BASALT_DOWNLOAD_URL with a compatible archive."
                ;;
        esac
        die "Failed to download ${_artifact}
Please verify that the version exists for your platform at:
${GITLAB_URL}/${PROJECT}/-/releases"
    fi

    # Download and verify checksum
    if [ -z "${DOWNLOAD_URL}" ]; then
        download_checksum "${_version}" "${_artifact}" || true
        if [ -f "basalt.tar.gz.sha256" ]; then
            _expected_checksum="$(awk '{print $1}' < basalt.tar.gz.sha256)"
            verify_checksum "basalt.tar.gz" "${_expected_checksum}"
        fi
    fi

    log "Extracting..."
    tar xzf basalt.tar.gz

    # Check for expected directory structure
    if [ ! -d "release" ]; then
        die "Invalid archive structure (expected 'release' directory)"
    fi

    # Install binaries
    log "Installing binaries to $_install_dir ..."
    if [ -d "release/bin" ]; then
        cp -f release/bin/* "$_install_dir/"
        chmod +x "$_install_dir"/basalt_* 2>/dev/null || true
    fi

    # Install library
    log "Installing library to $_lib_dir ..."
    if [ -d "release/lib" ]; then
        cp -f release/lib/* "$_lib_dir/"
        # On macOS, update install names for dylib
        if [ "$(uname)" = "Darwin" ]; then
            for lib in release/lib/*.dylib; do
                if [ -f "$lib" ]; then
                    libname="$(basename "$lib")"
                    install_name_tool -id "$_lib_dir/$libname" "$_lib_dir/$libname" 2>/dev/null || true
                fi
            done
        fi
    fi

    # Install data files
    log "Installing data files to $_data_dir ..."
    if [ -d "release/data" ]; then
        cp -rf release/data/* "$_data_dir/"
    fi

    # Create installation receipt
    _receipt_file="${_receipt_dir}/install.json"
    cat > "${_receipt_file}" <<EOF
{
  "version": "${_version}",
  "arch": "${_arch}",
  "install_dir": "${HOME}/.local",
  "installed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

    # Clean up
    cd "$HOME"
    rm -rf "$_tmp_dir"
    _tmp_dir=""

    # Update PATH if needed
    _rc_file="$(get_rc_file)"
    if needs_path_update "$_rc_file"; then
        echo ""
        log "Adding $_install_dir to PATH in $_rc_file ..."
        update_path "$_rc_file"
    fi

    echo ""
    echo "Basalt $_version installed successfully!"
    echo ""
    echo "Binaries:  $_install_dir"
    echo "Library:   $_lib_dir"
    echo "Data:      $_data_dir"
    echo "Receipt:   $_receipt_file"
    echo ""
    if needs_path_update "$_rc_file"; then
        echo "To complete the installation, either:"
        echo "  1. Run: source $_rc_file"
        echo "  2. Or restart your shell"
    else
        echo "You can now run basalt binaries from your shell."
    fi
}

main "$@"
