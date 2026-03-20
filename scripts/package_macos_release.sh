#!/bin/bash
set -euo pipefail

# Basalt macOS Release Packaging Script
# Usage: ./scripts/package_macos_release.sh [version] [--upload <remote>]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BASALT_RELEASE_BUILD_DIR:-build/release}"
RELEASE_DIR="${REPO_ROOT}/artifacts/release"
ARTIFACTS_DIR="${REPO_ROOT}/artifacts"
GITLAB_BASE_URL="${GITLAB_BASE_URL:-${CI_SERVER_URL:-}}"
GITLAB_URL="${CI_API_V4_URL:-}"
PROJECT_ID="${CI_PROJECT_ID:-${GITLAB_PROJECT_ID:-}}"
PROJECT_PATH="${CI_PROJECT_PATH:-${GITLAB_PROJECT_PATH:-}}"
JOB_TOKEN="${CI_JOB_TOKEN:-}"
PRIVATE_TOKEN="${GITLAB_TOKEN:-}"
VERSION="dev"
UPLOAD_REMOTE=""

usage() {
  cat <<EOF
Usage: ./scripts/package_macos_release.sh [version] [--upload <remote>]

Options:
  --upload <remote>  Upload the generated artifacts to the GitLab release for <version>
                     using the specified git remote to resolve the GitLab host/project.
  -h, --help         Show this help message.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --upload)
      if [ "$#" -lt 2 ]; then
        echo "Missing remote name for --upload." >&2
        usage >&2
        exit 1
      fi
      UPLOAD_REMOTE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [ "${VERSION}" != "dev" ]; then
        echo "Unexpected extra argument: $1" >&2
        usage >&2
        exit 1
      fi
      VERSION="$1"
      shift
      ;;
  esac
done

ARCH_TRIPLET="$(sh "${SCRIPT_DIR}/detect_platform.sh")"

echo "Packaging basalt ${VERSION} for macOS ${ARCH_TRIPLET}..."

cd "${REPO_ROOT}"

STRIP_BIN=""
if command -v llvm-strip >/dev/null 2>&1; then
  STRIP_BIN="llvm-strip"
elif command -v strip >/dev/null 2>&1; then
  STRIP_BIN="strip"
else
  echo "Warning: no strip tool found; packaged files will not be stripped" >&2
fi

strip_artifact() {
  local artifact_path="$1"

  if [ -z "${STRIP_BIN}" ]; then
    return 0
  fi

  if ! "${STRIP_BIN}" "${artifact_path}" >/dev/null 2>&1; then
    echo "Warning: failed to strip ${artifact_path}" >&2
  fi
}

require_cmd() {
  local cmd="$1"

  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Required command '${cmd}' not found." >&2
    exit 1
  fi
}

resolve_gitlab_project() {
  local remote_name="$1"
  local remote_url

  if [ -n "${GITLAB_BASE_URL}" ] && [ -n "${PROJECT_PATH}" ]; then
    return 0
  fi

  remote_url="$(git config --get "remote.${remote_name}.url" || true)"
  if [ -z "${remote_url}" ] && [ "${remote_name}" != "gitlab" ]; then
    remote_name="gitlab"
    remote_url="$(git config --get "remote.${remote_name}.url" || true)"
  fi

  if [ -z "${remote_url}" ]; then
    echo "Could not determine GitLab project from git remotes. Set GITLAB_BASE_URL and GITLAB_PROJECT_PATH." >&2
    exit 1
  fi

  if [ -z "${GITLAB_BASE_URL}" ]; then
    case "${remote_url}" in
      git@*:* )
        GITLAB_BASE_URL="https://$(printf '%s\n' "${remote_url}" | sed -E 's#^git@([^:]+):.*#\1#')"
        ;;
      ssh://git@*/* )
        GITLAB_BASE_URL="https://$(printf '%s\n' "${remote_url}" | sed -E 's#^ssh://git@([^/]+)/.*#\1#')"
        ;;
      https://*|http://* )
        GITLAB_BASE_URL="$(printf '%s\n' "${remote_url}" | sed -E 's#^((https?)://[^/]+)/.*#\1#')"
        ;;
      * )
        echo "Unsupported git remote URL format: ${remote_url}" >&2
        exit 1
        ;;
    esac
  fi

  if [ -z "${PROJECT_PATH}" ]; then
    case "${remote_url}" in
      git@*:* )
        PROJECT_PATH="$(printf '%s\n' "${remote_url}" | sed -E 's#^git@[^:]+:(.*)\.git$#\1#; s#^git@[^:]+:(.*)$#\1#')"
        ;;
      ssh://git@*/* )
        PROJECT_PATH="$(printf '%s\n' "${remote_url}" | sed -E 's#^ssh://git@[^/]+/(.*)\.git$#\1#; s#^ssh://git@[^/]+/(.*)$#\1#')"
        ;;
      https://*|http://* )
        PROJECT_PATH="$(printf '%s\n' "${remote_url}" | sed -E 's#^(https?://[^/]+)/(.*)\.git$#\2#; s#^(https?://[^/]+)/(.*)$#\2#')"
        ;;
    esac
  fi

  if [ -z "${GITLAB_BASE_URL}" ] || [ -z "${PROJECT_PATH}" ]; then
    echo "Failed to derive GitLab host and project path from remote '${remote_name}'." >&2
    exit 1
  fi

  if [ -z "${GITLAB_URL}" ]; then
    GITLAB_URL="${GITLAB_BASE_URL}/api/v4"
  fi
}

get_auth_header() {
  if [ -n "${JOB_TOKEN}" ]; then
    echo "JOB-TOKEN: ${JOB_TOKEN}"
  elif [ -n "${PRIVATE_TOKEN}" ]; then
    echo "PRIVATE-TOKEN: ${PRIVATE_TOKEN}"
  else
    echo "GitLab token not set. Expected CI_JOB_TOKEN or GITLAB_TOKEN for release upload." >&2
    exit 1
  fi
}

resolve_project_id() {
  local auth_header="$1"

  if [ -n "${PROJECT_ID}" ]; then
    return 0
  fi

  if [ -z "${PROJECT_PATH}" ]; then
    echo "CI_PROJECT_ID or CI_PROJECT_PATH must be set for release upload." >&2
    exit 1
  fi

  PROJECT_ID="$(
    curl -sS --header "${auth_header}" \
      "${GITLAB_URL}/projects/$(jq -rn --arg v "${PROJECT_PATH}" '$v|@uri')" |
      jq -r '.id // empty'
  )"

  if [ -z "${PROJECT_ID}" ]; then
    echo "Failed to resolve GitLab project ID for ${PROJECT_PATH}." >&2
    exit 1
  fi
}

release_exists() {
  local auth_header="$1"
  local status

  status="$(
    curl -sS -o /dev/null -w "%{http_code}" \
      --header "${auth_header}" \
      "${GITLAB_URL}/projects/${PROJECT_ID}/releases/${VERSION}"
  )"

  [ "${status}" = "200" ]
}

upload_file_to_gitlab() {
  local auth_header="$1"
  local artifact_path="$2"
  local response
  local upload_url
  local relative_url

  response="$(
    curl -sS --request POST \
      --header "${auth_header}" \
      --form "file=@${artifact_path}" \
      "${GITLAB_URL}/projects/${PROJECT_ID}/uploads"
  )"

  upload_url="$(echo "${response}" | jq -r '.full_path // empty')"
  if [ -n "${upload_url}" ]; then
    case "${upload_url}" in
      http://*|https://*|ftp://*) ;;
      /*) upload_url="${GITLAB_BASE_URL}${upload_url}" ;;
      *) upload_url="${GITLAB_BASE_URL}/${upload_url}" ;;
    esac
  fi

  if [ -z "${upload_url}" ]; then
    relative_url="$(echo "${response}" | jq -r '.url // empty')"
    if [ -n "${relative_url}" ]; then
      case "${relative_url}" in
        http://*|https://*|ftp://*) upload_url="${relative_url}" ;;
        /*) upload_url="${GITLAB_BASE_URL}${relative_url}" ;;
        *) upload_url="${GITLAB_BASE_URL}/${relative_url}" ;;
      esac
    fi
  fi

  if [ -z "${upload_url}" ]; then
    echo "Failed to upload ${artifact_path} to GitLab." >&2
    echo "${response}" | jq -r '.message // .error // .' >&2
    exit 1
  fi

  printf '%s\n' "${upload_url}"
}

upsert_release_link() {
  local auth_header="$1"
  local name="$2"
  local url="$3"
  local filepath="$4"
  local link_type="$5"
  local links_response
  local existing_id
  local payload
  local response

  payload="$(jq -n \
    --arg name "${name}" \
    --arg url "${url}" \
    --arg filepath "${filepath}" \
    --arg link_type "${link_type}" \
    '{name: $name, url: $url, filepath: $filepath, link_type: $link_type}')"

  links_response="$(
    curl -sS \
      --header "${auth_header}" \
      "${GITLAB_URL}/projects/${PROJECT_ID}/releases/${VERSION}/assets/links"
  )"
  existing_id="$(echo "${links_response}" | jq -r --arg name "${name}" '.[] | select(.name == $name) | .id' | head -n1)"

  if [ -n "${existing_id}" ]; then
    response="$(
      curl -sS --request PUT \
        --header "${auth_header}" \
        --header "Content-Type: application/json" \
        --data "${payload}" \
        "${GITLAB_URL}/projects/${PROJECT_ID}/releases/${VERSION}/assets/links/${existing_id}"
    )"
  else
    response="$(
      curl -sS --request POST \
        --header "${auth_header}" \
        --header "Content-Type: application/json" \
        --data "${payload}" \
        "${GITLAB_URL}/projects/${PROJECT_ID}/releases/${VERSION}/assets/links"
    )"
  fi

  if ! echo "${response}" | jq -e '.url' >/dev/null 2>&1; then
    echo "Failed to register release asset link for ${name}." >&2
    echo "${response}" | jq -r '.message // .error // .' >&2
    exit 1
  fi
}

upload_release_asset() {
  local auth_header="$1"
  local filename="$2"
  local link_type="$3"
  local artifact_path="${ARTIFACTS_DIR}/${filename}"
  local uploaded_url

  if [ ! -f "${artifact_path}" ]; then
    echo "Expected artifact not found: ${artifact_path}" >&2
    exit 1
  fi

  echo "Uploading ${filename} to GitLab..."
  uploaded_url="$(upload_file_to_gitlab "${auth_header}" "${artifact_path}")"
  upsert_release_link "${auth_header}" "${filename}" "${uploaded_url}" "/${filename}" "${link_type}"
}

# Check if build exists
if [ ! -d "${BUILD_DIR}" ]; then
  echo "Build directory not found. Run cmake --preset release and cmake --build --preset release first." >&2
  exit 1
fi

# Set up release directory
rm -rf "${RELEASE_DIR}"
mkdir -p "${RELEASE_DIR}"/{bin,lib,data}

# Copy binaries
echo "Copying binaries..."
RELEASE_BINS=(
  basalt_calibrate
  basalt_calibrate_imu
  basalt_kitti_eval
  basalt_mapper
  basalt_mapper_sim
  basalt_mapper_sim_naive
  basalt_opt_flow
  basalt_rs_t265_record
  basalt_rs_t265_vio
  basalt_time_alignment
  basalt_vio
  basalt_vio_sim
)

for bin in "${RELEASE_BINS[@]}"; do
  bin_path="${BUILD_DIR}/${bin}"
  if [ ! -x "${bin_path}" ]; then
    echo "Required executable not found: ${bin_path}" >&2
    exit 1
  fi
  install -m 0755 "${bin_path}" "${RELEASE_DIR}/bin/${bin}"
  strip_artifact "${RELEASE_DIR}/bin/${bin}"
  echo "  ${bin}"
done

for script in scripts/basalt_*.py; do
  if [ ! -f "${script}" ]; then
    continue
  fi
  install -m 0755 "${script}" "${RELEASE_DIR}/bin/$(basename "${script}")"
  echo "  $(basename "${script}")"
done

# Copy shared library
echo "Copying shared library..."
if [ -f "${BUILD_DIR}/libbasalt.dylib" ]; then
  install -m 0755 "${BUILD_DIR}/libbasalt.dylib" "${RELEASE_DIR}/lib/libbasalt.dylib"
  strip_artifact "${RELEASE_DIR}/lib/libbasalt.dylib"
  echo "  libbasalt.dylib"
else
  echo "Required shared library not found: ${BUILD_DIR}/libbasalt.dylib" >&2
  exit 1
fi

# Copy data files
echo "Copying data files..."
cp -a data/. "${RELEASE_DIR}/data/"

# Create tarball with standardized naming
cd "${ARTIFACTS_DIR}"
TARBALL="basalt-${VERSION}-${ARCH_TRIPLET}.tar.gz"
echo "Creating ${TARBALL}..."
tar -czf "${TARBALL}" release/

cd "${REPO_ROOT}"
bash "${SCRIPT_DIR}/generate_checksums.sh"

if [ -n "${UPLOAD_REMOTE}" ]; then
  require_cmd curl
  require_cmd jq

  resolve_gitlab_project "${UPLOAD_REMOTE}"
  AUTH_HEADER="$(get_auth_header)"
  resolve_project_id "${AUTH_HEADER}"

  if ! release_exists "${AUTH_HEADER}"; then
    echo "GitLab release '${VERSION}' does not exist for project ${PROJECT_ID}." >&2
    exit 1
  fi

  upload_release_asset "${AUTH_HEADER}" "${TARBALL}" "package"
  upload_release_asset "${AUTH_HEADER}" "${TARBALL}.sha256" "other"
  upload_release_asset "${AUTH_HEADER}" "checksums.txt" "other"
fi

echo ""
echo "=========================================="
echo "Release package created!"
echo "=========================================="
echo "Location: artifacts/${TARBALL}"
if [ -n "${UPLOAD_REMOTE}" ]; then
  echo "Release assets updated for tag: ${VERSION}"
fi
echo ""
echo "Contents:"
echo "  - Binaries: $(ls -1 "${RELEASE_DIR}/bin" | wc -l | tr -d ' ')"
echo "  - Libraries: $(ls -1 "${RELEASE_DIR}/lib" 2>/dev/null | wc -l | tr -d ' ' || echo 0)"
echo "  - Data files: $(ls -1 "${RELEASE_DIR}/data" | wc -l | tr -d ' ')"
echo ""
