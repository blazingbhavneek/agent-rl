import os
from pathlib import Path
from pprint import pprint

import clang.cindex


def get_local_function_definitions(
    code_bytes: bytes, file_name: str = "input.c"
) -> dict:
    """
    Extracts function definitions using libclang.

    :param code_bytes: Raw bytes of the C source file
    :param file_name: Virtual filename for the unsaved file (helps clang determine language)
    :returns: { name: {name, return_type, line, start_line, end_line} }
    """
    index = clang.cindex.Index.create()

    # Parse from unsaved buffer
    tu = index.parse(
        file_name,
        args=["-x", "c"],
        unsaved_files=[(file_name, code_bytes)],
        options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
    )

    definitions = {}

    def traverse(cursor):
        # Only consider cursors from our file, not included headers
        if cursor.location.file and cursor.location.file.name != file_name:
            return

        if (
            cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL
            and cursor.is_definition()
        ):
            name = cursor.spelling
            return_type = cursor.result_type.spelling
            start = cursor.extent.start.line
            end = cursor.extent.end.line

            definitions[name] = {
                "name": name,
                "return_type": return_type,
                "line": start,
                "start_line": start,
                "end_line": end,
            }

        for child in cursor.get_children():
            traverse(child)

    traverse(tu.cursor)
    return definitions


if __name__ == "__main__":
    file_name = (
        Path(__file__).resolve().parent.parent / "src/src_rbt/src/rbt001/rbt001Main.c"
    )
    pprint(get_local_function_definitions(file_name.read_bytes(), file_name.name))
