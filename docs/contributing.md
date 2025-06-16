# Contributing Guide

Thank you for your interest in contributing to this project! This guide will
help you get started with contributing effectively.

## Getting Started

1. Fork the repository and clone your fork:

   ```bash
   gh repo clone dreadnode/python-template
   cd python-template
   ```

1. Set up your development environment with your preferred package manager:

   ```bash
   # Using pip
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt

   # Using uv
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt

   # Using poetry
   poetry install
   ```

1. Install and configure pre-commit hooks:

   ```bash
   pre-commit install
   ```

1. Create a new branch for your contribution:

   ```bash
   git checkout -b your-feature-name
   ```

## Development Process

### Code Quality Checks

Before submitting your changes, ensure all quality checks pass:

```bash
# Run all checks at once
task check

# Or run individual checks:
task format  # Code formatting with Black
task lint    # Linting with Ruff
task types   # Type checking with mypy
task test    # Run tests with pytest
```

The pre-commit hooks will automatically run most checks when you commit changes.

### Documentation

- Add documentation for new features in the `docs/` directory
- Update existing documentation to reflect your changes
- Run `mkdocs serve` to preview documentation locally

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/) for your
commit messages:

```bash
# Format: <type>: <description>
feat: add new template feature
fix: resolve issue with pytest config
docs: improve setup instructions
ci: update GitHub Actions workflow
```

Common types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `ci`, `chore`

## Pull Request Process

1. Update your branch with the latest changes:

   ```bash
   git fetch origin
   git rebase origin/main
   ```

1. Push your changes:

   ```bash
   git push origin your-feature-name
   ```

1. Create a pull request with:

   1. Clear, descriptive title following conventional commits
   1. Detailed description of changes
   1. Links to related issues
   1. Screenshots for UI changes (if applicable)
   1. Documentation updates
   1. Template updates (if required)

1. Address any review feedback and update your PR

## Code Standards

- Follow the existing code style (enforced by Black and Ruff)
- Maintain type hints and add new ones for your code
- Write clear docstrings for new functions and classes
- Add tests for new features or bug fixes
- Keep changes focused and atomic
- Update example code if needed

## Project Structure

When adding new features, follow the project structure:

```bash
.
├── docs/              # Documentation
├── examples/          # Usage examples
├── tests/             # Test suite
└── pyproject.toml     # Project configuration
```

## Getting Help

- Check existing issues and pull requests
- Open a new issue for:
  - Bug reports
  - Feature requests
  - Questions about the template
  - Contributing questions

We aim to respond to all issues promptly.
