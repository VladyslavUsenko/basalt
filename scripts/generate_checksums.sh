#!/bin/bash
set -euo pipefail

# Basalt Checksum Generation Script
# Generates SHA256 checksums for release artifacts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}/artifacts"

echo "Generating SHA256 checksums..."

checksum_cmd() {
    if command -v sha256sum >/dev/null 2>&1; then
        echo "sha256sum"
    elif command -v shasum >/dev/null 2>&1; then
        echo "shasum -a 256"
    else
        echo "No SHA256 tool found (expected sha256sum or shasum)" >&2
        exit 1
    fi
}

CHECKSUM_CMD="$(checksum_cmd)"

# Generate checksums for all tar.gz files
if ls basalt-*.tar.gz 1> /dev/null 2>&1; then
    ${CHECKSUM_CMD} basalt-*.tar.gz > checksums.txt
    echo "Created checksums.txt"

    # Create individual .sha256 files for each artifact
    for artifact in basalt-*.tar.gz; do
        if [ -f "${artifact}" ]; then
            ${CHECKSUM_CMD} "${artifact}" | awk '{print $1}' > "${artifact}.sha256"
            echo "Created ${artifact}.sha256"
        fi
    done

    echo ""
    echo "Checksums generated:"
    cat checksums.txt
else
    echo "No basalt-*.tar.gz files found in artifacts/"
    exit 1
fi
