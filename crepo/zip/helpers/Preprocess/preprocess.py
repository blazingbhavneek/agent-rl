import subprocess
import tempfile
from pathlib import Path
from typing import Dict

# import tree_sitter_cpp as tsc
import tree_sitter_custom as tsc
from tree_sitter import Language, Parser, Tree

from helpers.time_it import time_it
from state.state import State


# region extract macros and includes from files and save in the STATE...
def extract_all_macros(filepath):
    if isinstance(filepath, str):
        filepath = Path(filepath)
    with open(filepath, "rb") as f:
        source_code = f.read()
    parser = Parser(Language(tsc.language()))
    tree = parser.parse(source_code)
    results = {}

    def walk_tree(node):
        # Check if the current node is a macro definition
        if node.type in ["preproc_def", "preproc_function_def"]:
            name_node = node.child_by_field_name("name")
            value_node = node.child_by_field_name("value")

            if name_node and value_node:
                name = source_code[name_node.start_byte : name_node.end_byte].decode(
                    "latin-1"
                )
                value = source_code[value_node.start_byte : value_node.end_byte].decode(
                    "latin-1"
                )

                # Clean up multi-line backslashes
                clean_value = value.replace("\\\n", "").replace("\\\r\n", "").strip()
                results[name] = clean_value

        # Recursively visit all children (this dives into #if, #ifdef, etc.)
        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)
    return results


def extract_includes(filepath):
    if isinstance(filepath, str):
        filepath = Path(filepath)
    STATE = State()
    C_LANGUAGE = Language(tsc.language())
    parser = Parser(C_LANGUAGE)
    trees = STATE.get("TREES")
    if filepath.name not in trees:
        return []
    tree = trees[filepath.name][0]
    include_paths = []
    source_code = trees[filepath.name][1]

    def walk(node):
        # Handle #include directives
        if node.type == "preproc_include":
            # The 'path' field points to the <header.h> or "header.h" node
            path_node = node.child_by_field_name("path")
            if path_node:
                raw_path = source_code[
                    path_node.start_byte : path_node.end_byte
                ].decode("latin-1")
                # Remove the C-style delimiters: < > or " "
                clean_path = raw_path.strip('<">')
                include_paths.append(clean_path)

        # Recurse through all children to catch includes inside #if/#else blocks
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return include_paths


# endregion


class Preprocess:
    def __init__(self):
        self.language = Language(tsc.language())
        self.parser = Parser(self.language)

    def remove_comments(self, code: bytes) -> bytes:
        tree = self.parser.parse(code)
        comment_ranges: list[tuple[int, int]] = []

        def collect_comments(node):
            if node.type == "comment":
                comment_ranges.append((node.start_byte, node.end_byte))
            for child in node.children:
                collect_comments(child)

        collect_comments(tree.root_node)
        if not comment_ranges:
            return code

        comment_ranges.sort(key=lambda r: r[0])
        result = bytearray()
        pos = 0

        for start, end in comment_ranges:
            result.extend(code[pos:start])
            # Replace each byte with space, but keep newlines intact
            for byte in code[start:end]:
                if byte == ord(b"\n"):
                    result.append(byte)
                else:
                    result.append(ord(b" "))
            pos = end

        result.extend(code[pos:])
        return bytes(result)

    def _run_unifdef(self, code: bytes) -> bytes:
        """
        Runs the unifdef command to clean #if #else directives.
        Uses a temporary file to interface with the CLI tool.
        """
        # -t: ignore non-C lines, -k: keep blank lines, -b: replace deleted lines with blank lines
        command = ["unifdef", "-t", "-k", "-b"]

        try:
            process = subprocess.run(
                command,
                input=code,
                capture_output=True,
                check=False,  # unifdef returns 1 if it made changes, which is fine
            )
            # unifdef exit codes: 0=no changes, 1=changes made, 2=error
            if process.returncode == 2:
                print(
                    f"Warning: unifdef encountered an error: {process.stderr.decode()}"
                )
                return code
            return process.stdout
        except FileNotFoundError:
            print(
                "Error: 'unifdef' command not found. Please install it on your system."
            )
            return code

    # @time_it
    def preprocess(
        self,
        project_structure: Dict[str, Path] | None = None,
        file_path: Path | None = None,
    ) -> Dict[str, tuple[Tree, bytes]] | tuple[Tree, bytes] | None:
        """
        Iterates through the project, cleans .h and .c files,
        and returns a mapping of filenames to Tree-sitter Trees.
        """
        # ONLY USED WHEN A SINGLE FILE NEEDS TO BE PARSED AND CLEANED>
        if file_path:  # then its externally called
            content = file_path.read_bytes()
            if file_path.suffix == ".h":
                # Header files: Only remove comments
                clean_content = self.remove_comments(content)

            elif file_path.suffix == ".c":
                # C files: 1. Remove comments -> 2. Run unifdef
                content_no_comments = self.remove_comments(content)
                clean_content = self._run_unifdef(content_no_comments)

            # Parse the cleaned code into a Tree-sitter Tree

            tree = self.parser.parse(clean_content)

            return (tree, clean_content)

        processed_trees: Dict[str, tuple[Tree, str]] = {}
        # import sys

        for filename, path in project_structure.items():
            if not path.exists():
                continue

            content = path.read_bytes()
            if path.suffix == ".h":
                # Header files: Only remove comments
                clean_content = self.remove_comments(content)

            elif path.suffix == ".c":
                # C files: 1. Remove comments -> 2. Run unifdef
                content_no_comments = self.remove_comments(content)
                clean_content = self._run_unifdef(content_no_comments)

            # Parse the cleaned code into a Tree-sitter Tree

            tree = self.parser.parse(clean_content)

            content = path.read_bytes()
            if path.suffix == ".h":
                # Header files: Only remove comments
                clean_content = self.remove_comments(content)

            elif path.suffix == ".c":
                # C files: 1. Remove comments -> 2. Run unifdef
                content_no_comments = self.remove_comments(content)
                clean_content = self._run_unifdef(content_no_comments)

            else:
                continue

            # Parse the cleaned code into a Tree-sitter Tree

            tree = self.parser.parse(clean_content)

            processed_trees[filename] = (tree, clean_content)

        return processed_trees
