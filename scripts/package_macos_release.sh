#!/bin/bash
set -euo pipefail

# Basalt macOS Release Packaging Script
# Usage: ./scripts/package_macos_release.sh [version]

VERSION="${1:-dev}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BASALT_RELEASE_BUILD_DIR:-build/release}"

ARCH_TRIPLET="$(sh "${SCRIPT_DIR}/detect_platform.sh")"

echo "Packaging basalt ${VERSION} for macOS ${ARCH_TRIPLET}..."

cd "${REPO_ROOT}"

# Check if build exists
if [ ! -d "${BUILD_DIR}" ]; then
  echo "Build directory not found. Run ./scripts/build_macos.sh first." >&2
  exit 1
fi

# Set up release directory
RELEASE_DIR="${REPO_ROOT}/artifacts/release"
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
  basalt_vio
  basalt_vio_sim
)

for bin in "${RELEASE_BINS[@]}"; do
  bin_path="${BUILD_DIR}/${bin}"
  if [ ! -f "${bin_path}" ]; then
    echo "Warning: ${bin_path} not found, skipping" >&2
    continue
  fi
  cp "${bin_path}" "${RELEASE_DIR}/bin/"
  chmod +x "${RELEASE_DIR}/bin/${bin}"
  echo "  ${bin}"
done

# Copy shared library
echo "Copying shared library..."
if [ -f "${BUILD_DIR}/libbasalt.dylib" ]; then
  cp "${BUILD_DIR}/libbasalt.dylib" "${RELEASE_DIR}/lib/"
  echo "  libbasalt.dylib"
else
  echo "Warning: libbasalt.dylib not found" >&2
fi

# Copy data files
echo "Copying data files..."
cp -r data/* "${RELEASE_DIR}/data/"

# Create tarball with standardized naming
cd artifacts
TARBALL="basalt-${VERSION}-${ARCH_TRIPLET}.tar.gz"
echo "Creating ${TARBALL}..."
tar -czf "${TARBALL}" release/

cd "${REPO_ROOT}"
echo ""
echo "=========================================="
echo "Release package created!"
echo "=========================================="
echo "Location: artifacts/${TARBALL}"
echo ""
echo "Contents:"
echo "  - Binaries: $(ls -1 "${RELEASE_DIR}/bin" | wc -l | tr -d ' ')"
echo "  - Libraries: $(ls -1 "${RELEASE_DIR}/lib" 2>/dev/null | wc -l | tr -d ' ' || echo 0)"
echo "  - Data files: $(ls -1 "${RELEASE_DIR}/data" | wc -l | tr -d ' ')"
echo ""
