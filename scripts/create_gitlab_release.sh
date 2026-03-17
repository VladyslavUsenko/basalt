#!/bin/bash
set -euo pipefail

# Basalt GitLab Release Creation Script
# Creates a GitLab release with artifacts for a given tag

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GITLAB_BASE_URL="${CI_SERVER_URL:-https://gitlab.com}"
# Configuration from GitLab CI environment variables
# Uses predefined GitLab CI variables: https://docs.gitlab.com/ci/variables/predefined_variables/
GITLAB_URL="${CI_API_V4_URL:-${GITLAB_BASE_URL}/api/v4}"
PROJECT_ID="${CI_PROJECT_ID:-}"
PROJECT_PATH="${CI_PROJECT_PATH:-}"
TAG_NAME="${CI_COMMIT_TAG:-}"
JOB_TOKEN="${CI_JOB_TOKEN:-}"
PRIVATE_TOKEN="${GITLAB_TOKEN:-}"
ARTIFACT_JOB_NAME="${BASALT_ARTIFACT_JOB_NAME:-ubuntu22-build}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

add_release_link() {
    local auth_header="$1"
    local name="$2"
    local url="$3"
    local filepath="$4"
    local link_type="$5"

    local link_data
    local link_response

    link_data=$(jq -n \
        --arg name "${name}" \
        --arg url "${url}" \
        --arg filepath "${filepath}" \
        --arg link_type "${link_type}" \
        '{
            name: $name,
            url: $url,
            filepath: $filepath,
            link_type: $link_type
        }')

    link_response=$(curl -s --request POST \
        --header "${auth_header}" \
        --header "Content-Type: application/json" \
        --data "${link_data}" \
        "${GITLAB_URL}/projects/${PROJECT_ID}/releases/${TAG_NAME}/assets/links")

    if echo "${link_response}" | jq -e '.url' > /dev/null 2>&1; then
        log_info "    ✓ Link added"
    else
        log_warn "    ✗ Failed to add link for ${name}"
        echo "${link_response}" | jq -r '.message // .error // .' >&2
    fi
}

artifact_url() {
    local artifact_path="$1"
    local project_url="${CI_PROJECT_URL:-${GITLAB_BASE_URL}/${PROJECT_PATH}}"

    printf '%s/-/jobs/artifacts/%s/raw/artifacts/%s?job=%s\n' \
        "${project_url}" "${TAG_NAME}" "${artifact_path}" "${ARTIFACT_JOB_NAME}"
}

# Check required variables
check_requirements() {
    if [ -z "${TAG_NAME}" ]; then
        log_error "CI_COMMIT_TAG is not set. This script should only run on git tags."
        exit 1
    fi

    if [ -z "${PROJECT_ID}" ] && [ -z "${PRIVATE_TOKEN}" ]; then
        log_error "Either CI_PROJECT_ID or GITLAB_TOKEN must be set."
        exit 1
    fi

    if [ -z "${PROJECT_ID}" ]; then
        # Try to get project ID from API using project path
        log_info "CI_PROJECT_ID not set, attempting to resolve project path..."
        if [ -n "${PRIVATE_TOKEN}" ]; then
            # Use CI_PROJECT_PATH if available, otherwise require it to be set
            PROJECT_SEARCH="${PROJECT_PATH:-}"
            if [ -z "${PROJECT_SEARCH}" ]; then
                log_error "CI_PROJECT_PATH must be set when CI_PROJECT_ID is not available"
                exit 1
            fi

            PROJECT_RESPONSE=$(curl -s --header "PRIVATE-TOKEN: ${PRIVATE_TOKEN}" \
                "${GITLAB_URL}/projects?search=${PROJECT_SEARCH}")

            PROJECT_ID=$(echo "${PROJECT_RESPONSE}" | jq -r '.[0].id // empty')

            if [ -z "${PROJECT_ID}" ]; then
                log_error "Could not resolve project ID for '${PROJECT_SEARCH}'"
                exit 1
            fi
            log_info "Resolved project ID: ${PROJECT_ID}"
        else
            log_error "Cannot resolve project ID without GITLAB_TOKEN"
            exit 1
        fi
    fi

    # Check for required commands
    for cmd in curl jq; do
        if ! command -v "${cmd}" &> /dev/null; then
            log_error "Required command '${cmd}' not found. Please install it."
            exit 1
        fi
    done
}

# Get authentication token
get_auth_header() {
    if [ -n "${JOB_TOKEN}" ]; then
        echo "JOB-TOKEN: ${JOB_TOKEN}"
    elif [ -n "${PRIVATE_TOKEN}" ]; then
        echo "PRIVATE-TOKEN: ${PRIVATE_TOKEN}"
    else
        log_error "No authentication token available (CI_JOB_TOKEN or GITLAB_TOKEN required)"
        exit 1
    fi
}

# Check if release already exists
check_existing_release() {
    local auth_header="$1"

    log_info "Checking if release for ${TAG_NAME} already exists..."

    RELEASE_RESPONSE=$(curl -s --header "${auth_header}" \
        "${GITLAB_URL}/projects/${PROJECT_ID}/releases/${TAG_NAME}" || echo "")

    if echo "${RELEASE_RESPONSE}" | jq -e '.tag_name' > /dev/null 2>&1; then
        log_warn "Release ${TAG_NAME} already exists. Skipping creation."
        return 1
    fi

    return 0
}

# Generate release notes
generate_release_notes() {
    local artifacts_dir="${REPO_ROOT}/artifacts"
    local checksums_file="${artifacts_dir}/checksums.txt"
    local project_url="${CI_PROJECT_URL:-${GITLAB_BASE_URL}/${PROJECT_PATH}}"

    cat <<EOF
# Basalt ${TAG_NAME}

## Installation

\`\`\`bash
curl -LsSf ${project_url}/-/raw/master/scripts/install.sh | sh
\`\`\`

Or with a specific version:
\`\`\`bash
curl -LsSf ${project_url}/-/raw/master/scripts/install.sh | sh -s -- ${TAG_NAME}
\`\`\`

## Artifacts

EOF

    if [ -f "${checksums_file}" ]; then
        cat <<EOF
### Checksums

\`\`\`
$(cat "${checksums_file}")
\`\`\`

EOF
    fi
}

# Create the release
create_release() {
    local auth_header="$1"
    local release_notes="$2"

    log_info "Creating release ${TAG_NAME}..."

    # Create release with assets as links
    RELEASE_DATA=$(jq -n \
        --arg name "Basalt ${TAG_NAME}" \
        --arg tag_name "${TAG_NAME}" \
        --arg description "${release_notes}" \
        '{
            name: $name,
            tag_name: $tag_name,
            description: $description
        }')

    RELEASE_RESPONSE=$(curl -s --request POST \
        --header "${auth_header}" \
        --header "Content-Type: application/json" \
        --data "${RELEASE_DATA}" \
        "${GITLAB_URL}/projects/${PROJECT_ID}/releases")

    if echo "${RELEASE_RESPONSE}" | jq -e '.tag_name' > /dev/null 2>&1; then
        log_info "Release ${TAG_NAME} created successfully!"
        return 0
    else
        log_error "Failed to create release."
        echo "${RELEASE_RESPONSE}" | jq -r '.message // .error // .' >&2
        return 1
    fi
}

# Upload artifact links to the release
upload_artifact_links() {
    local auth_header="$1"
    local artifacts_dir="${REPO_ROOT}/artifacts"
    local project_url="${CI_PROJECT_URL:-${GITLAB_BASE_URL}/${PROJECT_PATH}}"

    log_info "Uploading artifact links..."

    cd "${artifacts_dir}"

    # Find all tar.gz files and their checksums
    for artifact in basalt-*.tar.gz; do
        if [ -f "${artifact}" ]; then
            local artifact_name="${artifact}"
            local artifact_url
            artifact_url="$(artifact_url "${artifact_name}")"
            local artifact_filepath="/${artifact_name}"

            log_info "  Adding link: ${artifact_name}"
            add_release_link "${auth_header}" "${artifact_name}" "${artifact_url}" "${artifact_filepath}" "package"

            # Also add checksum file
            if [ -f "${artifact}.sha256" ]; then
                local checksum_name="${artifact}.sha256"
                local checksum_url
                checksum_url="$(artifact_url "${checksum_name}")"
                local checksum_filepath="/${checksum_name}"

                log_info "  Adding link: ${checksum_name}"
                add_release_link "${auth_header}" "${checksum_name}" "${checksum_url}" "${checksum_filepath}" "other"
            fi
        fi
    done

    # Add checksums.txt
    if [ -f "checksums.txt" ]; then
        local checksums_name="checksums.txt"
        local checksums_url
        checksums_url="$(artifact_url "${checksums_name}")"
        local checksums_filepath="/${checksums_name}"

        log_info "  Adding link: ${checksums_name}"
        add_release_link "${auth_header}" "${checksums_name}" "${checksums_url}" "${checksums_filepath}" "other"
    fi
}

# Main execution
main() {
    log_info "Starting GitLab release creation for ${TAG_NAME}..."

    check_requirements

    AUTH_HEADER="$(get_auth_header)"

    # Check if release already exists
    if ! check_existing_release "${AUTH_HEADER}"; then
        log_warn "Release already exists. Skipping."
        exit 0
    fi

    # Generate release notes
    RELEASE_NOTES="$(generate_release_notes)"

    # Create release
    if create_release "${AUTH_HEADER}" "${RELEASE_NOTES}"; then
        # Upload artifact links
        upload_artifact_links "${AUTH_HEADER}"

        local project_url="${CI_PROJECT_URL:-${GITLAB_BASE_URL}/${PROJECT_PATH}}"
        log_info "Release ${TAG_NAME} completed successfully!"
        log_info "View at: ${project_url}/-/releases/${TAG_NAME}"
    else
        log_error "Failed to create release ${TAG_NAME}"
        exit 1
    fi
}

main "$@"
