# Stage 4 / Unit Test Generation — Known Failure Modes

## Root Causes for Wrong Makefile / Missing Source / Coverage = 0

### 1. Agent wraps the target function itself
- Agent writes `__wrap_<target_func>()` in unit test file
- `sync_wrap_flags` picks it up → appends `-Wl,--wrap=target_func` to unit Makefile
- Linker redirects all calls to wrapper → real implementation in PROD_OBJ never runs → coverage = 0
- **Fix**: check after agent run — if `__wrap_{func_id}` is in unit test file, trigger fix

### 2. `include $(MASTER_MAKEFILE)` rule conflicts
- Master defines `TEST_PROGRAM`, `TEST_SRCS`, `test:`, `clean-test:`, `PRODUCTION_SRCS`, `WRAP_FUNCS`
- Unit overrides these after include, but master's `$(TEST_PROGRAM): $(TEST_SRCS)` rule may compile master `.c` into unit binary depending on GNU Make rule resolution
- Master's `PRODUCTION_SRCS` inherited — agent sometimes adds those to unit link command
- **Fix**: remove `include $(MASTER_MAKEFILE)`, embed only needed vars explicitly

### 3. Agent includes production `.c` directly AND PROD_OBJ exists
- Duplicate symbol at link, or linker picks inline version which has no `.gcno` → coverage = 0
- `_sync_stub_srcs` doesn't detect this
- **Fix**: prompt must explicitly forbid including production source in test file

### 4. `CC` is commented out
- `#CC = gcc` — if master Makefile doesn't set CC, falls back to `cc`
- `--coverage` behavior may differ
- **Fix**: uncomment, hardcode `CC = gcc`

### 5. Static target function
- Agent writes direct call from test file to a `static` function → linker error (different TU)
- Prompt says "execute through real caller" but agent ignores it
- **Fix**: analysis must flag static functions; prompt must be explicit

### 6. Agent tests adjacent functions, not target
- Same PROD_OBJ covers entire source file
- Agent tests unrelated functions → those lines covered, target lines `start_line..end_line` never touched
- `check_function_coverage` returns 0 for target
- **Fix**: prompt must explicitly state which line range must be hit

### 7. Binary crashes before `__gcov_flush` / `.gcda` not written
- `run_make_test` → binary runs → crashes mid-run → `.gcda` never flushed → gcov sees no hits → coverage = 0
- Reported as normal compile success, coverage mysteriously 0
- **Fix**: check for `.gcda` existence after run; diagnose crashes before coverage check

### 8. Agent edits master test file
- Agent given `folder=repo_root`, nothing prevents editing master test file
- Corrupts master for all subsequent functions
- **Fix**: `protect_source` in `run_agent` should also protect master test file

### 9. Agent changes `PROD_SRC` in unit Makefile
- Points to wrong file → correct source never compiled with coverage → gcov match fails
- **Fix**: validate PROD_SRC in unit Makefile after agent run

### 10. Agent adds production `.c` to `TEST_SRCS` in unit Makefile
- Source compiled twice, `--wrap` flags from one TU don't apply to the other
- **Fix**: same as #3 — prompt + post-agent Makefile validation

### 11. `_validate_stub_locally` infinite loop (Stage 2, blocks Stage 4)
- `while True:` with no attempt cap
- If agent never fixes stub, Stage 2 hangs forever
- **Fix**: add `max_stub_validate_attempts` cap, fail stub gracefully

### 12. `_sync_stub_srcs` exclusion regex misses multi-line signatures
- `r'__wrap_(\w+)\s*\([^)]*\)\s*\{'` — fails on multi-line function signatures
- Stub stays in `STUB_SRCS` even though test file defines local override → duplicate symbol
- **Fix**: use multiline regex or parse block-level
