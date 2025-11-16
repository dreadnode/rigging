import argparse
import re
import subprocess  # nosec B404 - This is not called by user input
import sys
from collections import defaultdict


def get_commits_between_tags(previous_tag, current_tag):
    """Get all commits between two tags."""
    try:
        # If current_tag is "HEAD", use the latest commit
        if current_tag == "HEAD":
            cmd = f"git log {previous_tag}..HEAD --pretty=format:'%h|%an|%s'"
        else:
            cmd = f"git log {previous_tag}..{current_tag} --pretty=format:'%h|%an|%s'"

        # nosec B602 - This is not called by user input
        result = subprocess.run(  # nosec B602 - This is not called by user input
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        return result.stdout.strip().split("\n")
    except subprocess.CalledProcessError as e:
        print(f"Error getting commits between tags: {e}")
        sys.exit(1)


def get_latest_tag():
    """Get the latest tag in the repository."""
    try:
        cmd = "git describe --tags --abbrev=0"
        result = subprocess.run(  # nosec B602 - This is not called by user input
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting latest tag: {e}")
        sys.exit(1)


def get_previous_tag(current_tag):
    """Get the tag before the current tag."""
    try:
        cmd = f"git describe --tags --abbrev=0 {current_tag}^"
        result = subprocess.run(  # nosec B602 - This is not called by user input
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting previous tag: {e}")
        sys.exit(1)


def categorize_commits(commits):
    """Categorize commits based on conventional commit prefixes or patterns."""
    categories = {
        "feat": "New Features",
        "fix": "Bug Fixes",
        "docs": "Documentation",
        "style": "Styling",
        "refactor": "Code Refactoring",
        "perf": "Performance Improvements",
        "test": "Tests",
        "build": "Build System",
        "ci": "CI",
        "chore": "Chores",
        "revert": "Reverts",
    }

    categorized_commits = defaultdict(list)

    for commit in commits:
        if not commit:
            continue

        hash_id, author, message = commit.split("|", 2)

        # Check for conventional commit format (type: message)
        match = re.match(r"^(\w+)(\(.+\))?!?: (.+)$", message)
        if match:
            commit_type = match.group(1).lower()
            commit_scope = match.group(2) if match.group(2) else ""
            commit_message = match.group(3)

            category = categories.get(commit_type, "Other Changes")
            categorized_commits[category].append(
                {
                    "hash": hash_id,
                    "author": author,
                    "message": commit_message,
                    "scope": commit_scope,
                }
            )
        else:
            categorized_commits["Other Changes"].append(
                {"hash": hash_id, "author": author, "message": message, "scope": ""}
            )

    return categorized_commits


def format_for_llm(current_tag, previous_tag, categorized_commits):
    """Format the commits in a way that's useful for an LLM."""
    output = []

    # Add header
    output.append(f"# Release Notes: {current_tag}")
    output.append(f"Changes from {previous_tag} to {current_tag}\n")

    # Add instructions for the LLM
    output.append("## Instructions for the LLM")
    output.append(
        "Please generate comprehensive release notes based on the following commit information."
    )
    output.append(
        "Organize the notes by category, highlight major features and fixes, and provide a brief summary of the release."  # noqa: E501
    )
    output.append("The notes should be clear, concise, and suitable for users of the software.\n")

    # Add categorized commits
    output.append("## Commit Information by Category")

    for category, commits in categorized_commits.items():
        if commits:
            output.append(f"### {category}")
            for commit in commits:
                scope = f" {commit['scope']}" if commit["scope"] else ""
                output.append(
                    f"- [{commit['hash']}] {commit['message']}{scope} (by {commit['author']})"
                )
            output.append("")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Generate release notes content for LLM from git commits."
    )
    parser.add_argument(
        "--tag",
        help="The current tag to generate notes for. Default is the latest tag.",
        default=None,
    )
    parser.add_argument(
        "--previous-tag",
        help="The previous tag to compare against. Default is automatic detection.",
        default=None,
    )
    parser.add_argument(
        "--output",
        help="Output file. If not specified, prints to stdout.",
        default=None,
    )

    args = parser.parse_args()

    # Determine current tag
    current_tag = args.tag if args.tag else get_latest_tag()

    # Determine previous tag
    previous_tag = args.previous_tag if args.previous_tag else get_previous_tag(current_tag)

    print(
        f"Generating release notes from {previous_tag} to {current_tag}...",
        file=sys.stderr,
    )

    # Get and categorize commits
    commits = get_commits_between_tags(previous_tag, current_tag)
    if not commits or (len(commits) == 1 and not commits[0]):
        print(
            f"No commits found between {previous_tag} and {current_tag}",
            file=sys.stderr,
        )
        sys.exit(0)

    categorized_commits = categorize_commits(commits)

    # Format for LLM
    content = format_for_llm(current_tag, previous_tag, categorized_commits)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(content)
        print(f"Release notes content written to {args.output}", file=sys.stderr)
    else:
        print(content)


if __name__ == "__main__":
    main()
