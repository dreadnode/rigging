import argparse  # noqa: INP001
import re
import typing as t
from pathlib import Path

from markdown import Markdown  # type: ignore [import-untyped]
from markdownify import MarkdownConverter  # type: ignore [import-untyped]
from markupsafe import Markup
from mkdocstrings_handlers.python._internal.config import PythonConfig
from mkdocstrings_handlers.python._internal.handler import (
    PythonHandler,
)

# ruff: noqa: T201


class CustomMarkdownConverter(MarkdownConverter):  # type: ignore [misc]
    # Strip extra whitespace from code blocks
    def convert_pre(self, el: t.Any, text: str, parent_tags: t.Any) -> t.Any:
        return super().convert_pre(el, text.strip(), parent_tags)

    # bold items with doc-section-title in a span class
    def convert_span(self, el: t.Any, text: str, parent_tags: t.Any) -> t.Any:  # noqa: ARG002
        if "doc-section-title" in el.get("class", []):
            return f"**{text.strip()}**"
        return text

    # Remove the div wrapper for inline descriptions
    def convert_div(self, el: t.Any, text: str, parent_tags: t.Any) -> t.Any:
        if "doc-md-description" in el.get("class", []):
            return text.strip()
        return super().convert_div(el, text, parent_tags)

    # Map mkdocstrings details classes to Mintlify callouts
    def convert_details(self, el: t.Any, text: str, parent_tags: t.Any) -> t.Any:  # noqa: ARG002
        classes = el.get("class", [])

        # Handle source code details specially
        if "quote" in classes:
            summary = el.find("summary")
            if summary:
                file_path = summary.get_text().replace("Source code in ", "").strip()
                content = text[text.find("```") :]
                return f'\n<Accordion title="Source code in {file_path}" icon="code">\n{content}\n</Accordion>\n'

        callout_map = {
            "note": "Note",
            "warning": "Warning",
            "info": "Info",
            "tip": "Tip",
        }

        callout_type = None
        for cls in classes:
            if cls in callout_map:
                callout_type = callout_map[cls]
                break

        if not callout_type:
            return text

        content = text.strip()
        if content.startswith(callout_type):
            content = content[len(callout_type) :].strip()

        return f"\n<{callout_type}>\n{content}\n</{callout_type}>\n"

    def convert_table(self, el: t.Any, text: str, parent_tags: t.Any) -> t.Any:
        # Check if this is a highlighttable (source code with line numbers)
        if "highlighttable" in el.get("class", []):
            code_cells = el.find_all("td", class_="code")
            if code_cells:
                code = code_cells[0].get_text()
                code = code.strip()
                code = code.replace("```", "~~~")
                return f"\n```python\n{code}\n```\n"

        return super().convert_table(el, text, parent_tags)


class AutoDocGenerator:
    def __init__(self, source_paths: list[str], theme: str = "material", **options: t.Any) -> None:
        self.source_paths = source_paths
        self.theme = theme
        self.handler = PythonHandler(PythonConfig.from_data(), base_dir=Path.cwd())
        self.options = options

        self.handler._update_env(  # noqa: SLF001
            Markdown(),
            config={"mdx": ["toc"]},
        )

        md = Markdown(extensions=["fenced_code"])

        def simple_convert_markdown(
            text: str,
            heading_level: int,
            html_id: str = "",
            **kwargs: t.Any,
        ) -> t.Any:
            return Markup(md.convert(text) if text else "")  # noqa: S704 # nosec

        self.handler.env.filters["convert_markdown"] = simple_convert_markdown

    def generate_docs_for_module(
        self,
        module_path: str,
    ) -> str:
        options = self.handler.get_options(
            {
                "docstring_section_style": "list",
                "merge_init_into_class": True,
                "show_signature_annotations": True,
                "separate_signature": True,
                "show_source": True,
                "show_labels": False,
                "show_bases": False,
                **self.options,
            },
        )

        module_data = self.handler.collect(module_path, options)
        html = self.handler.render(module_data, options)

        return str(
            CustomMarkdownConverter(
                code_language="python",
            ).convert(html),
        )

    def process_mdx_file(self, file_path: Path) -> bool:
        content = file_path.read_text(encoding="utf-8")
        original_content = content

        # Find the header comment block
        header_match = re.search(
            r"\{\s*/\*\s*((?:::.*?\n?)*)\s*\*/\s*\}",
            content,
            re.MULTILINE | re.DOTALL,
        )

        if not header_match:
            return False

        header = header_match.group(0)
        module_lines = header_match.group(1).strip().split("\n")

        # Generate content for each module
        markdown_blocks = []
        for line in module_lines:
            if line.startswith(":::"):
                module_path = line.strip()[3:].strip()
                if module_path:
                    markdown = self.generate_docs_for_module(module_path)
                    markdown_blocks.append(markdown)

        keep_end = content.find(header) + len(header)
        new_content = content[:keep_end] + "\n\n" + "\n".join(markdown_blocks)

        # Write back if changed
        if new_content != original_content:
            file_path.write_text(new_content, encoding="utf-8")
            print(f"[+] Updated: {file_path}")
            return True

        return False

    def process_directory(self, directory: Path, pattern: str = "**/*.mdx") -> int:
        if not directory.exists():
            print(f"[!] Directory does not exist: {directory}")
            return 0

        files_processed = 0
        files_modified = 0

        for mdx_file in directory.glob(pattern):
            if mdx_file.is_file():
                files_processed += 1
                if self.process_mdx_file(mdx_file):
                    files_modified += 1

        return files_modified


def main() -> None:
    """Main entry point for the script."""

    parser = argparse.ArgumentParser(description="Generate auto-docs for MDX files")
    parser.add_argument("--directory", help="Directory containing MDX files", default="docs")
    parser.add_argument("--pattern", default="**/*.mdx", help="File pattern to match")
    parser.add_argument(
        "--source-paths",
        nargs="+",
        default=["dreadnode"],
        help="Python source paths for module discovery",
    )
    parser.add_argument(
        "--show-if-no-docstring",
        type=bool,
        default=False,
        help="Show module/class/function even if no docstring is present",
    )
    parser.add_argument("--theme", default="material", help="Theme to use for rendering")

    args = parser.parse_args()

    # Create generator
    generator = AutoDocGenerator(
        source_paths=args.source_paths,
        theme=args.theme,
        show_if_no_docstring=args.show_if_no_docstring,
    )

    # Process directory
    directory = Path(args.directory)
    modified_count = generator.process_directory(directory, args.pattern)

    print(f"\n[+] Auto-doc generation complete. {modified_count} files were updated.")


if __name__ == "__main__":
    main()
