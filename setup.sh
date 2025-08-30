#!/bin/bash
set -e

# Define the target directory
DIR="semantic-release-git-actions"

# Create directory and navigate into it
mkdir -p "$DIR" &&
    cd "$DIR" &&
    touch file.txt .releaserc.prerelease.yaml .releaserc.yaml &&
    mkdir -p .github/workflows &&
    touch .github/workflows/semantic-release.yaml
