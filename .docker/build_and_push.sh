#!/usr/bin/env bash
set -euo pipefail

# Determine the repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default values
IMAGE_NAME="${IMAGE_NAME:-vladyslavusenko/basalt-ci}"
IMAGE_TAG="${IMAGE_TAG:-ubuntu22}"
IMAGE_REF="${IMAGE_NAME}:${IMAGE_TAG}"
DOCKERFILE="${DOCKERFILE:-${SCRIPT_DIR}/Dockerfile}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
BUILDER_NAME="${BUILDER_NAME:-basalt-ci-builder}"

# Change to repository root
cd "${REPO_ROOT}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but not found in PATH." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "docker daemon is not available. Start Docker first." >&2
  exit 1
fi

if ! docker buildx version >/dev/null 2>&1; then
  echo "docker buildx is required but not available." >&2
  exit 1
fi

if [ ! -f "${DOCKERFILE}" ]; then
  echo "Dockerfile not found: ${DOCKERFILE}" >&2
  exit 1
fi

# Set up buildx builder
if ! docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
  docker buildx create --name "${BUILDER_NAME}" --driver docker-container --use
else
  docker buildx use "${BUILDER_NAME}"
fi

docker buildx inspect --bootstrap >/dev/null

echo "Building and pushing ${IMAGE_REF}"
echo "Platforms: ${PLATFORMS}"
echo "Dockerfile: ${DOCKERFILE}"

docker buildx build \
  --platform "${PLATFORMS}" \
  --file "${DOCKERFILE}" \
  --tag "${IMAGE_REF}" \
  --push \
  .

echo "Done: ${IMAGE_REF}"
