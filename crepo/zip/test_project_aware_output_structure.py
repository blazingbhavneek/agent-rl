from __future__ import annotations

# =============================================================================
# REGION 01 - TEST CONTEXT AND IMPORT STUBS
# =============================================================================
#
# This test characterizes the original project_aware.py output structure without
# requiring a real C project, Tree-sitter, call graph analysis, or an LLM server.
#
# It then runs the same fake project through project_aware_new.py. The expected
# difference is:
#
#   project_aware.py:
#       writes only the legacy CSV/stats output tree.
#
#   project_aware_new.py:
#       preserves the legacy CSV/stats output tree and additionally writes
#       dpo_llm_data/... attempt/selection artifacts.
#

import csv
import importlib
import json
import sys
import types
from pathlib import Path

import pytest


ZIP_DIR = Path(__file__).resolve().parent
RESULTS_PREFIX = "/home/seigyo/c_repo/c_repo/results/csv_results"


def _install_import_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    parser_pkg = types.ModuleType("parser")
    parser_files = types.ModuleType("parser.parser_files")

    class ImportOnlyParseFiles:
        def __init__(self, *args, **kwargs):
            self.paths = kwargs.get("paths", [])

        def get_parsed_results(self, get_upper=True):
            return [(path, "int main(void) { return target_func(42); }") for path in self.paths]

    parser_files.parseFiles = ImportOnlyParseFiles
    monkeypatch.setitem(sys.modules, "parser", parser_pkg)
    monkeypatch.setitem(sys.modules, "parser.parser_files", parser_files)

    resolver_pkg = types.ModuleType("makefile_resolver")
    resolver_mod = types.ModuleType("makefile_resolver.makefile_resolver")
    resolver_mod.return_project_mapping = lambda show=False, project_path=None: ({}, [])
    monkeypatch.setitem(sys.modules, "makefile_resolver", resolver_pkg)
    monkeypatch.setitem(sys.modules, "makefile_resolver.makefile_resolver", resolver_mod)

    call_graph_pkg = types.ModuleType("call_graph")
    call_graph_mod = types.ModuleType("call_graph.call_graph")
    call_graph_data = types.ModuleType("call_graph.data_classes")
    call_graph_gen = types.ModuleType("call_graph.gen_graph")

    class DummyCallTreeNode:
        def __init__(self, name="node", children=None):
            self.name = name
            self.get_name = name
            self.children = children or []

    class DummyCustomTree:
        def __init__(self, name="node"):
            self.name = name
            self.get_name = name
            self.children = []

        def add_child(self, child):
            self.children.append(child)

    call_graph_mod.orchestrate = lambda *args, **kwargs: None
    call_graph_data.CallTreeNode = DummyCallTreeNode
    call_graph_data.custom_tree = DummyCustomTree
    call_graph_gen.make_graph = lambda paths: None
    monkeypatch.setitem(sys.modules, "call_graph", call_graph_pkg)
    monkeypatch.setitem(sys.modules, "call_graph.call_graph", call_graph_mod)
    monkeypatch.setitem(sys.modules, "call_graph.data_classes", call_graph_data)
    monkeypatch.setitem(sys.modules, "call_graph.gen_graph", call_graph_gen)

    clang_pkg = types.ModuleType("clang")
    clang_cindex = types.ModuleType("clang.cindex")
    clang_pkg.cindex = clang_cindex
    monkeypatch.setitem(sys.modules, "clang", clang_pkg)
    monkeypatch.setitem(sys.modules, "clang.cindex", clang_cindex)

    tree_sitter = types.ModuleType("tree_sitter")

    class DummyLanguage:
        def __init__(self, *args, **kwargs):
            pass

    class DummyParser:
        def __init__(self, *args, **kwargs):
            pass

        def parse(self, code):
            return types.SimpleNamespace(root_node=types.SimpleNamespace(children=[]))

    class DummyTree:
        pass

    tree_sitter.Language = DummyLanguage
    tree_sitter.Parser = DummyParser
    tree_sitter.Tree = DummyTree
    monkeypatch.setitem(sys.modules, "tree_sitter", tree_sitter)

    tree_sitter_custom = types.ModuleType("tree_sitter_custom")
    tree_sitter_custom.language = lambda: object()
    monkeypatch.setitem(sys.modules, "tree_sitter_custom", tree_sitter_custom)

    ollama = types.ModuleType("ollama")
    ollama.Client = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "ollama", ollama)

    pandas = types.ModuleType("pandas")
    pandas.json_normalize = lambda data, sep="->": data
    pandas.read_csv = lambda *args, **kwargs: []
    pandas.concat = lambda frames, axis=0, ignore_index=True: frames[0]
    monkeypatch.setitem(sys.modules, "pandas", pandas)

    pick = types.ModuleType("pick")
    pick.pick = lambda *args, **kwargs: (None, 0)
    monkeypatch.setitem(sys.modules, "pick", pick)


def _fresh_import(monkeypatch: pytest.MonkeyPatch, module_name: str):
    _install_import_stubs(monkeypatch)
    monkeypatch.syspath_prepend(str(ZIP_DIR))

    for name in ["project_aware_new", "project_aware"]:
        sys.modules.pop(name, None)

    if module_name == "project_aware_new":
        importlib.import_module("project_aware")

    return importlib.import_module(module_name)


# =============================================================================
# REGION 02 - FAKE PROJECT RUNTIME
# =============================================================================
#
# The fake runtime patches only the expensive or external parts:
#
#   - project mapping
#   - preprocessing
#   - function detection
#   - call graph generation
#   - context parsing
#   - LLM calls
#   - CSV output path
#
# The original make_llm_calls_for_function and trace_variable logic still run.
#


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    return str(value)


def _relative_tree(root: Path) -> list[str]:
    if not root.exists():
        return []

    entries = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        entries.append(f"{rel}/" if path.is_dir() else rel)
    return entries


def _mapped_path_factory(outputs_root: Path):
    def mapped_path(*parts):
        path = Path(*parts)
        path_text = str(path)
        if path_text.startswith(RESULTS_PREFIX):
            rel = path_text[len(RESULTS_PREFIX) :].lstrip("/")
            return outputs_root / "csv_results" / rel
        return path

    return mapped_path


def _configure_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    module,
    run_root: Path,
    outputs_root: Path,
) -> None:
    source_root = run_root / "source" / "TEST_PROJECT"
    source_root.mkdir(parents=True)
    source_file = source_root / "main.c"
    source_file.write_text("int target_func(int value) { return value; }\n", encoding="utf-8")

    module_cache = run_root / "module_cache"
    module_cache.mkdir()
    monkeypatch.setattr(module, "__file__", str(module_cache / f"{module.__name__}.py"))
    if hasattr(module, "_base"):
        monkeypatch.setattr(
            module._base,
            "__file__",
            str(module_cache / "project_aware.py"),
        )

    monkeypatch.setattr(module, "Path", _mapped_path_factory(outputs_root))
    if hasattr(module, "DPO_DATA_ROOT"):
        monkeypatch.setattr(module, "DPO_DATA_ROOT", outputs_root / "dpo_llm_data")
        module._DPO_PATH_INDEX_BY_IDENTITY.clear()
        module._DPO_NEXT_PATH_INDEX_BY_FUNCTION.clear()

    state = module.State()
    state.reset()
    state.set("PROJECT_NAME", "TEST_PROJECT")
    state.set("FUNCTION_POINTER_ARGS", {})
    state.set("FUNCTION_TYPES", {
        "target_func": {
            "indices": [1],
            "get_upper": True,
            "dependent_functions": ["target_func"],
            "type": "OPENF",
            "launch": "NO DATA",
        }
    })
    state.set("TOOL_DEFINITION", [])
    state.set("TOOLS", {})
    state.set("FUNCTION_MAP", {})

    def fake_return_project_mapping(show=False, project_path=None):
        return {"main.c": str(source_file)}, ["main.c"]

    class FakePreprocess:
        def preprocess(self, project_structure):
            return {
                "main.c": (
                    object(),
                    b"int main(void) { return target_func(42); }",
                )
            }

    def fake_get_local_function_definitions(code_bytes):
        return {
            "main": {},
            "target_func": {},
        }

    def fake_identify_funs_to_trace(project_structure, trees, name_of_json="unused"):
        return {
            "target_func": {
                "indices": [1],
                "get_upper": True,
                "dependent_functions": ["target_func"],
            }
        }

    raw_path = [
        "[main.c]main[1:4]",
        "[main.c:2]target_func[2:2]",
    ]

    call_graph_data = {
        "launch_via": "NO DATA",
        "call_function": "main",
        "function_name": "target_func",
        "function_name_src": {
            "path": str(source_file),
            "line_number": "1",
        },
        "target_name_src": {
            "path": str(source_file),
            "line_number": "2",
        },
    }

    def fake_orchestrate(**kwargs):
        return ({}, [((raw_path, None), call_graph_data)])

    class FakeParseFiles:
        def __init__(self, project_structure, paths, macro_data, file_name_bytes):
            self.paths = paths

        def get_parsed_results(self, get_upper=True):
            context = "\n".join(
                [
                    "int main(void) {",
                    "    return target_func(42); /*CONSIDER THIS CALL*/",
                    "}",
                ]
            )
            return [(path, context) for path in self.paths]

    def fake_llm_calls(
        project_structure,
        function_name_to_traced,
        argument_numbers,
        intial_context,
        path,
        get_upper=True,
        collect_history=False,
    ):
        answer = module.outputModel(output="1:42", call_number="7")
        stats = module.Stats.model_validate(
            {
                "Tokens": {
                    "Input_tokens": 11,
                    "Output_tokens": 5,
                    "Total_tokens": 16,
                },
                "Iterations": 1,
                "Random_tool_calls": 0,
                "Other_tool_errors": 0,
                "Incorrect_details": [],
            }
        )
        if collect_history:
            return answer, stats, [
                {"role": "user", "content": f"trace {function_name_to_traced}"},
                {"role": "assistant", "content": "1:42"},
            ]
        return answer, stats

    def fake_save_dict_csv(data_dict, *args, **kwargs):
        csv_path = outputs_root / "csv_results" / "TEST_PROJECT.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "row_json": json.dumps(_json_safe(data_dict), sort_keys=True),
        }
        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["row_json"])
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    monkeypatch.setattr(module, "return_project_mapping", fake_return_project_mapping)
    monkeypatch.setattr(module, "Preprocess", FakePreprocess)
    monkeypatch.setattr(module, "get_local_function_definitions", fake_get_local_function_definitions)
    monkeypatch.setattr(module, "extract_all_macros", lambda path: {})
    monkeypatch.setattr(module, "extract_includes", lambda path: [])
    monkeypatch.setattr(module, "identify_funs_to_trace", fake_identify_funs_to_trace)
    monkeypatch.setattr(module, "orchestrate", fake_orchestrate)
    monkeypatch.setattr(module, "make_graph", lambda paths: None)
    monkeypatch.setattr(module, "parseFiles", FakeParseFiles)
    monkeypatch.setattr(module, "llm_calls", fake_llm_calls)
    monkeypatch.setattr(module, "save_dict_csv", fake_save_dict_csv)

    if module.__name__ == "project_aware":
        monkeypatch.setattr(
            module,
            "run_with_retry",
            lambda func, args=(), timeout=180, retries=2: func(*tuple(args)),
        )


# =============================================================================
# REGION 03 - CHARACTERIZATION TESTS
# =============================================================================


@pytest.mark.parametrize(
    ("module_name", "expect_dpo"),
    [
        ("project_aware", False),
        ("project_aware_new", True),
    ],
)
def test_project_aware_output_tree(monkeypatch, tmp_path, module_name, expect_dpo):
    module = _fresh_import(monkeypatch, module_name)

    run_root = tmp_path / module_name
    outputs_root = run_root / "outputs"
    _configure_fake_runtime(monkeypatch, module, run_root, outputs_root)

    summary = module.trace_variable(run_root / "source" / "TEST_PROJECT")

    assert "target_func" in summary
    legacy_tree = [
        entry
        for entry in _relative_tree(outputs_root)
        if entry.startswith("csv_results/")
    ]
    assert legacy_tree == [
        "csv_results/",
        "csv_results/TEST_PROJECT.csv",
        "csv_results/stats/",
        "csv_results/stats/TEST_PROJECT_STATS.json",
        "csv_results/stats/stats/",
    ]

    dpo_tree = [
        entry
        for entry in _relative_tree(outputs_root)
        if entry.startswith("dpo_llm_data/")
    ]

    if not expect_dpo:
        assert dpo_tree == []
        return

    assert any(entry.endswith("selected.json") for entry in dpo_tree)
    assert sum(entry.endswith("/history.json") for entry in dpo_tree) == module.DPO_ATTEMPTS_PER_PATH
    assert sum(entry.endswith("/answer.json") for entry in dpo_tree) == module.DPO_ATTEMPTS_PER_PATH
    assert sum(entry.endswith("/score.json") for entry in dpo_tree) == module.DPO_ATTEMPTS_PER_PATH
