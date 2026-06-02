## C Code Test Stub & Coverage Workflow

Step 1: Identify the functions to be stubbed.

Parse output: Extract external_defined_funcs (not defined in that file but others in the same project) and external_undefined_funcs (undefined functions need to be stubbed).

- For the external_defined_funcs the function doesn't need to be stubbed its defined in the another c file and that file is linked with this file. HALT if: Script fails or output cannot be parsed.

Finding stub function for the files in the project source

/home/seigyo/test_env/.venv/bin/activate;uv_run

/home/seigyo/test_env/stub_identifier/stub_identifier.py {{PROJECT_FOLDER}}

PROJECT_FOLDER = absolute path for project's root folder.

## Step 2: Initialize Isolated Test Directory & Template Makefile

1. Extract HOME path from {{PROJECT_FOLDER}}/Makefile. If not explicitly defined, fallback to {HOME = ../..}.

2. Create test directory: mkdir -p $HOME/unit_tests/$(basename {{PROJECT_FOLDER}}). We can't modify and add any file in the original project's folder.

3. Generate template Makefile: <execute_command> cd $HOME/unit_tests/$(basename {{PROJECT_FOLDER}}) && do_mkmf {{PROJECT_FOLDER}} <requires_approval>true</requires_approval>

</execute_command> HALT if: do_mkmf fails or template Makefile is not generated.

## Step 3: Port Original Makefile Configuration to Test Makefile

<read_file> {{PROJECT_FOLDER}}/Makefile </read_file>

## Extract & Inject:

- Parse CFLAGS, CXXFLAGS, INCLUDE (or equivalent include paths), LDFLAGS, and system/library flags.

- Append them to the new Makefile located at $HOME/unit_tests/$(basename {{PROJECT_FOLDER}})/Makefile.

- DO NOT redefine PROJECT. It is already set to VERSION_MNG in the system environment and will be inherited automatically.

- Add necessary header search paths (-I) pointing back to {{PROJECT_FOLDER}}/include and system directories.

```xml

<write_to_file> $HOME/unit_tests/$(basename {{PROJECT_FOLDER}})/Makefile {{injected_config_block}}

append </write_to_file>

```

- The variable $HOME is same as the one in the source's makefile not the env variable of the system.

## Step 4: Generate Integrated Test & Stub File (test_{{{BASENAME}}}.c)

- This test file contains test for all the files in the PROJECT_FOLDER (Mentioned in the SRCS in the original Makefile...).

For each .c file in the project.

4a: Process external_undefined_funcs

Middleware (pmf_* / mpf_*)

1. Query MCP server "mpf-pmf_func_info" for prototype and docs.

2. HALT if: Query returns empty/error.

3. Generate stub using the same mandatory docstring format (noting MCP as source).

## Other External Functions

1. Search headers: rg -n --type-add 'header:*.h' -t header '{{func}}' ~

2. HALT if: Prototype not found.

3. Extract signature + docstring from header or the file in which it is found

4. Generate linker-compatible stub with docstring.

- Don't make any dependency headers in the folder other than the test file if anything needs to be defined please do that in the same file.

## 4b: Append Test Skeleton

Below all stubs, append the CUnit test suite initialization, teardown, and a placeholder for test cases.

<write_to_file> $HOME/unit_tests/$(basename {{PROJECT_FOLDER}})/test_{{BASENAME}}.c

{{generated_test_and_stubs}} overwrite </write_to_file>

## Step 5: Configure Test Target in New Makefile

```html

<read_file> $HOME/unit_tests/$(basename {{PROJECT_FOLDER}}))/Makefile </read_file>

```

Append the following block (ensure tabs are used for recipe lines, not spaces):

```bash

# === TEST TARGET FOR {{TARGET_FILE}} ===

TEST_PROGRAM = test_{{BASENAME}}

TEST_SRCS = test_{{BASENAME}}.c

TEST_LIBS = -lcunit

TEST_REPORT_FILE = test_{{BASENAME}}_report.txt # for the cunit test cases report and the coverage data.

TEST_LOG_FILE = test_{{BASENAME}}_log.txt # for the generated logs (passed to stderr)

COVERAGE_FLAGS = --coverage -ffunction-sections -fdata-sections

WRAP_FLAGS = $(WRAP_FUNCS) # Populate dynamically based on detected external_undefined_funcs

.PHONY: test clean-test

```

print.md

```bash

test_{TEST_PROGRAM}: $(TEST_SRCS)

$(CC) $(CFLAGS_LINUX) $(CFLAGS) $(INCLUDE) $(COVERAGE_FLAGS) $(TEST_SRCS) -o

$(TEST_PROGRAM) $(TEST_LIBS) \

-Wl,--gc-sections \

$(WRAP_FLAGS)

./$(TEST_PROGRAM) > $(TEST_REPORT_FILE) 2>$(TEST_LOG_FILE)

gcov {{TARGET_FILE}} >> $(TEST_REPORT_FILE)

clean-test:

rm -f $(TEST_PROGRAM) $(TEST_REPORT_FILE) $(TEST_LOG_FILE) *.gcda *.gcno *.o

```

```bash

<write_to_file> $HOME/unit_tests/$(basename {{PROJECT_FOLDER}})/Makefile {{append_block}} append

</write_to_file>

```

## Step 6: Verify Compilation, Coverage

```bash

<execute_command> cd $HOME/unit_tests/$(basename {{PROJECT_FOLDER}}) && make test

<requires_approval>false</requires_approval> </execute_command>

```

HALT if: Compilation fails. Output exact GCC error lines, request signature or include-path fix.

Success Output: Confirm generation of:

- $HOME/unit_tests/$(basename {{PROJECT_FOLDER}})/test_{{BASENAME}}

- test_{{BASENAME}}_report.txt (CUnit output + gcov coverage)

- test_{{BASENAME}}_log.txt (stderr/debug logs)

- *.gcda / *.gcno coverage artifacts
