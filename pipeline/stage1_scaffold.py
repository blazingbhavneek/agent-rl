from __future__ import annotations

import sys
from pathlib import Path

from .config import PipelineConfig
from .common import (
    TEST_FILE_MARKERS,
    _project_source_files,
    _source_includes_for_test_file,
    read_text,
    write_text,
)


def ensure_test_file(cfg: PipelineConfig, paths: dict) -> None:
    """
    Ensure the CUnit test file exists and includes actual production .c files.

    Important:
    - Uses cfg.source_dir as the source of truth.
    - Does not guess src/<process>.c.
    - Includes every .c found under cfg.source_dir.
    - Defines main before production includes so production main is renamed.
    """
    test_file: Path = paths["test_file"]
    process_name: str = paths["process_name"]
    test_file.parent.mkdir(parents=True, exist_ok=True)

    # Make includes for all the files present in the source
    production_include_lines = _source_includes_for_test_file(cfg, test_file)

    # Replace the main function with <process_name>_entry_main so we can write tests for it without it executing and interfering with test main.
    define_main = f"#define main {process_name}_entry_main"
    production_include_block = ""
    if production_include_lines:
        production_include_block = (
            f"\n/* Production sources — main renamed so CUnit owns int main(void). */\n"
            f"{define_main}\n"
            + "\n".join(production_include_lines)
            + "\n#undef main\n"
        )
    
    # This is the main test code for the process # TODO: Tell every prompt that is going to add stuff to it, to respect the boundaries defined by this
    if not test_file.exists():
        skeleton = f"""/* CUnit tests for {process_name} */
{TEST_FILE_MARKERS[0]}
#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
{production_include_block}

{TEST_FILE_MARKERS[1]}
/* Compatibility definitions go here if needed. */

{TEST_FILE_MARKERS[2]}
/* Static globals shared by stubs/tests go here. */

{TEST_FILE_MARKERS[3]}
/* Reusable helper functions go here. */

{TEST_FILE_MARKERS[4]}
/* Linker wrapper stubs go here. */

{TEST_FILE_MARKERS[5]}
/* Test cases go here. */

{TEST_FILE_MARKERS[6]}
int main(void)
{{
    CU_pSuite suite = NULL;
    if (CU_initialize_registry() != CUE_SUCCESS) {{
        return CU_get_error();
    }}
    suite = CU_add_suite("{process_name}_suite", NULL, NULL);
    if (suite == NULL) {{
        CU_cleanup_registry();
        return CU_get_error();
    }}
    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    {{
        unsigned int failures = CU_get_number_of_failures();
        CU_cleanup_registry();
        return failures == 0 ? 0 : 1;
    }}
}}
"""
        write_text(test_file, skeleton)
        print(f"[pipeline] created test file: {test_file}", file=sys.stderr)
        print("[pipeline] actual source files included:", file=sys.stderr)

        # Logging number of C files
        for src in _project_source_files(cfg):
            print(f" - {src}", file=sys.stderr)
        return
    
    # Following is for continuation logic, if the test file already exists but an agent downstream made some bad edits it needs to detected and fixed
    text = read_text(test_file)
    changed = False

    # Ensure the includes section marker exists (may be missing in very old scaffolds).
    if TEST_FILE_MARKERS[0] not in text:
        text = TEST_FILE_MARKERS[0] + "\n" + text
        changed = True

    missing_includes = [
        line for line in production_include_lines
        if line not in text
    ]
    if missing_includes:
        has_define = define_main in text
        if has_define:
            # Case A: #define main already present — insert new #includes right after it
            # so they stay inside the define/undef rename block.
            new_inc_block = "\n".join(missing_includes) + "\n"
            text = text.replace(
                define_main + "\n",
                define_main + "\n" + new_inc_block,
                1,
            )
        else:
            # Case B: no #define main yet — wrap the new includes in a full define/undef block.
            insertion = (
                f"\n/* Production sources — main renamed so CUnit owns int main(void). */\n"
                f"{define_main}\n"
                + "\n".join(missing_includes)
                + "\n#undef main\n"
            )
            text = text.replace(TEST_FILE_MARKERS[0], TEST_FILE_MARKERS[0] + insertion, 1)
        changed = True

    # Ensure all section markers exist (append any that are missing at end of file).
    # Instead of end of file, append just before main defintion, keep the main runner below always
    # TODO: Add to prompt to not remove these header and anytime adding something to file it needs to organize based on this header
    for marker in TEST_FILE_MARKERS:
        if marker not in text:
            text += f"\n\n{marker}\n"
            changed = True

    if changed:
        write_text(test_file, text)
        print(f"[pipeline] updated test file with actual source includes: {test_file}", file=sys.stderr)
        print("[pipeline] actual source files included:", file=sys.stderr)
        for src in _project_source_files(cfg):
            print(f" - {src}", file=sys.stderr)
