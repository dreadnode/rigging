# Testing Renovate Locally

This guide explains how to use our Taskfile to test Renovate locally using Docker.

## Prerequisites

- Docker installed and running
- Task v3 or later installed
- [GitHub CLI](https://cli.github.com/) installed and authenticated (`gh auth token`)

## Configuration

The Renovate configuration is stored in `.github/renovate.json5`. This file defines:

- Package managers (poetry, pip)
- Auto-merge rules
- Dependencies grouping
- Update patterns

## Environment Setup

1. Set your GitHub token (two options):

   a. Using a personal access token:

   ```bash
   export GITHUB_TOKEN=your_github_token
   ```

   b. Using GitHub CLI (recommended):

   ```bash
   export GITHUB_TOKEN=$(gh auth token)
   ```

1. Configure the repository you want to test:

   ```bash
   export REPOSITORY=org/repo
   ```

## Basic Usage

Run Renovate with default settings:

```bash
task renovate-docker-debug
```

This command will:

1. Verify Docker is running
1. Validate repository format
1. Execute Renovate in debug mode

## Advanced Configuration

### Modifying Log Level

```bash
export LOG_LEVEL=debug  # Options: debug, info, warn, error
task renovate-docker-debug
```

### Changing Platform

```bash
export PLATFORM=gitlab  # Default: github
task renovate-docker-debug
```

### Custom Parameters

You can pass additional parameters to Renovate:

```bash
task renovate-docker-debug -- --dry-run=true
```

## Environment Variables Reference

| Variable     | Description          | Required | Default |
| ------------ | -------------------- | -------- | ------- |
| GITHUB_TOKEN | Authentication token | Yes      | -       |
| REPOSITORY   | Target repository    | Yes      | -       |
| LOG_LEVEL    | Logging verbosity    | No       | debug   |
| PLATFORM     | Git platform         | No       | github  |

## Example Usage

Using GitHub CLI (recommended):

```bash
# One-line command with automatic token generation
TASK_X_REMOTE_TASKFILES=1 REPOSITORY="org/repo" GITHUB_TOKEN=$(gh auth token) task renovate:renovate-docker-debug
```

Manual token setup:

```bash
export GITHUB_TOKEN=your_github_token
export REPOSITORY="org/repo"
TASK_X_REMOTE_TASKFILES=1 task renovate:renovate-docker-debug
```

## Troubleshooting

### Common Issues

1. **Docker not running:**

   ```bash
   Error: Docker daemon is not running
   ```

   Solution: Start Docker desktop or daemon

1. **Invalid repository format:**

   ```bash
   Error: Repository must be in org/repo format
   ```

Solution: Ensure REPOSITORY is set correctly (e.g., "microsoft/vscode")

1. **Authentication failed:**

   ```bash
   Error: Authentication error
   ```

Solution: Verify GITHUB_TOKEN is set and has correct permissions
