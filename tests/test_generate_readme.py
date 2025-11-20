import os
import tempfile
from pathlib import Path

import pytest

from scripts.generate_readme import get_section_content, main

SAMPLE_README = """# Test Project 🐍

<!-- BEGIN_AUTO_BADGES -->
<div align="center">
Old badge content
</div>
<!-- END_AUTO_BADGES -->

## Manual Content
This should not be modified.
"""

SAMPLE_TEMPLATE = """# {{ project_name }}

<!-- BEGIN_AUTO_BADGES -->
<div align="center">
New badge content
</div>
<!-- END_AUTO_BADGES -->
"""

SAMPLE_PYPROJECT = """
[project]
name = "test-project"
version = "0.1.0"
description = "Test Description"

[tool.readme]
github_org = "test-org"
emoji = "🐍"
"""


def test_get_section_content():
    content = SAMPLE_README
    start_marker = "<!-- BEGIN_AUTO_BADGES -->"
    end_marker = "<!-- END_AUTO_BADGES -->"

    section, start, end = get_section_content(content, start_marker, end_marker)

    assert "Old badge content" in section
    assert start >= 0
    assert end > start


@pytest.fixture
def temp_project():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create temporary project structure
        project_dir = Path(tmpdir)

        # Create templates directory
        template_dir = project_dir / "templates"
        template_dir.mkdir()

        # Create template file
        with open(template_dir / "README.md.j2", "w") as f:
            f.write(SAMPLE_TEMPLATE)

        # Create pyproject.toml
        with open(project_dir / "pyproject.toml", "w") as f:
            f.write(SAMPLE_PYPROJECT)

        # Create initial README
        with open(project_dir / "README.md", "w") as f:
            f.write(SAMPLE_README)

        # Change to temp directory for test
        original_dir = os.getcwd()
        os.chdir(project_dir)

        yield project_dir

        # Restore original directory
        os.chdir(original_dir)


def test_readme_generation(temp_project):
    # Run the generator
    main()

    # Read the generated README
    with open(temp_project / "README.md") as f:
        content = f.read()

    # Check that the auto-generated section was updated
    assert "New badge content" in content

    # Check that manual content was preserved
    assert "Manual Content" in content
    assert "This should not be modified" in content


def test_readme_generation_dry_run(temp_project, capsys):
    # Run the generator with --dry-run
    import sys

    sys.argv.append("--dry-run")
    main()
    sys.argv.remove("--dry-run")

    # Check stdout contains the changes
    captured = capsys.readouterr()
    assert "New badge content" in captured.out

    # Check file wasn't modified
    with open(temp_project / "README.md") as f:
        content = f.read()
    assert "Old badge content" in content
