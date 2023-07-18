#!/bin/bash
#
# Build the base Docker image for the backend service and push it to the Docker registry

# Set Docker registry user
DOCKER_REGISTRY_USER=arjeo

# Check if user is logged in to Docker registry
if ! docker info | grep -q "${DOCKER_REGISTRY_USER}"; then
  echo "Error: Not logged in to Docker registry as ${DOCKER_REGISTRY_USER}"
  exit 1
fi

# Build Docker image
if ! docker build -t ${DOCKER_REGISTRY_USER}/talk-to-repo-backend-base -f ../backend/Dockerfile.base ../backend; then
  echo "Error: Docker build failed"
  exit 1
fi

# Push Docker image to registry
if ! docker push ${DOCKER_REGISTRY_USER}/talk-to-repo-backend-base; then
  echo "Error: Docker push failed"
  exit 1
fi

echo "Docker image built and pushed successfully"