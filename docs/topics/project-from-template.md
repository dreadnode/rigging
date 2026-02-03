# Setting Up a New Project from python-template

This guide walks through creating a new project using our Python template
repository.

## Initial Repository Setup

1. Create new repository from template:

   - Go to the python-template repository
   - Click "Use this template"
   - Repository name: Your project name (e.g., `siqcalc`)
   - Add description (e.g., "THE BEST Calculator example")
   - Click "Create repository"

1. Clone your new repository:

   ```bash
   gh repo clone dreadnode/your-project-name
   cd your-project-name
   ```

## Project Configuration

1. Remove template files:

   ```bash
   # Remove example files
   rm -rf src/* tests/*
   ```

1. Update `pyproject.toml`:

   - Set your project name, version, and description
   - Update author information
   - Configure dependencies as needed

1. Create your project structure:

   ```bash
   # Create source files
   mkdir -p src/your_project_name
   touch src/your_project_name/__init__.py

   # Create test directory
   mkdir -p tests
   ```

## Development Environment Setup

1. Create and activate virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

1. Install dependencies:

   ```bash
   # Upgrade pip
   python -m pip install --upgrade pip

   # Install core development tools
   pip install poetry pytest pytest-cov ruff

   # Install project in editable mode
   pip install -e .
   ```

1. Update lock and initialize Poetry:

   ```bash
   poetry lock
   poetry install
   ```

## Github Actions Configuration

1. Add your `OPENAI_API_KEY` to the repo's Action Secrets for
   [AI PR decorating][ai-pr-decorating]

1. Ensure the `@dreadnode/team` team have `write` access to the repository, as
   per the `CODEOWNERS` [definition][codeowners]

[ai-pr-decorating]: https://github.com/dreadnode/python-template/blob/b2e90f5905ae8c4793ffe5e646577754dc6b4fe6/.github/workflows/rigging_pr_description.yaml#L33-L34
[codeowners]: https://github.com/dreadnode/python-template/blob/b2e90f5905ae8c4793ffe5e646577754dc6b4fe6/CODEOWNERS#L1

1. Update repository general settings (optional), recommended:

   - [Suggestions to update pull request branches][pr-branches]
   - Enable [auto-merge capabilities][auto-merge]
   - [Automatic branch deletion][branch-deletion]
   - Enable [signed commits][signed-commits]
   - Remove features such as [Wikis][wikis] and [Projects][projects] that
     aren't necessary (rule of least requirements)

1. Setup [branch protection rules][branch-protection] (optional, but strongly
   recommended for public repositories)

[pr-branches]: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-suggestions-to-update-pull-request-branches
[auto-merge]: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-auto-merge-for-pull-requests-in-your-repository
[branch-deletion]: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-the-automatic-deletion-of-branches
[signed-commits]: https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits
[wikis]: https://docs.github.com/en/communities/documenting-your-project-with-wikis
[projects]: https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects
[branch-protection]: https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/managing-a-branch-protection-rule

1. Require a pull request before merging

   - Require at least 1 approval
   - Dismiss stale pull request approvals when new commits are pushed

1. Require status checks to pass before merging

   - Require branches to be up to date before merging
   - Required checks: `test`, `lint`, `format`

1. Include administrators in these restrictions
1. Allow force pushes: Disable (protect history)

## Quality Checks

Run these commands to verify your setup:

```bash
# Format code
ruff format .

# Run linting
ruff check .

# Run tests with coverage
pytest --cov=src --cov-report=term-missing
```

## Documentation

1. Create basic documentation:

   ```bash
   mkdir -p docs
   touch docs/index.md
   ```

1. Build documentation locally:

   ```bash
   poetry run mkdocs serve
   ```

   View at http://127.0.0.1:8000

## Common Issues

1. **Poetry Installation Conflicts**: If you encounter dependency conflicts
   while installing Poetry:

   ```bash
   # Alternative installation method
   curl -sSL https://install.python-poetry.org | python3 -
   ```

   Add to PATH if needed:

   ```bash
   export PATH="$HOME/.local/bin:$PATH"
   ```

1. **Import Errors**: If you see import errors in tests:

   - Verify your project structure matches the expected layout
   - Ensure you've installed the project in editable mode
     (`pip install -e .`)

1. **Linting Errors**: For Ruff formatting issues:

   ```bash
   # Auto-fix formatting
   ruff format .

   # Auto-fix linting issues where possible
   ruff check . --fix
   ```

## Next Steps

1. Start adding your project code to `src/`
1. Write tests in `tests/`
1. Update documentation in `docs/`
1. Configure CI/CD in `.github/workflows/`

## Reference

- [Python Template Documentation](link-to-template-docs)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [Ruff Documentation](https://beta.ruff.rs/docs/)
- [pytest Documentation](https://docs.pytest.org/)
