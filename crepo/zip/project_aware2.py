from __future__ import annotations


# region imports

# in built
import gc
import json
import multiprocessing
import os
import pickle
import re
import subprocess
import sys
import time
from collections import defaultdict
from parser.parser_files import parseFiles
from pathlib import Path
from pprint import pprint
from typing import Literal
import random
from collections import Counter

# 3rd party
import clang.cindex
import ollama
import pandas as pd
from pick import pick
from pydantic import BaseModel

# for highlighting the context which is c code..
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import CLexer

# from rich import print as rprint
from rich.console import Console
from rich.markup import escape
from rich.syntax import Syntax
from rich.tree import Tree
from tree_sitter import Language, Parser
from tree_sitter_custom import language

# intra-project import
from call_graph.call_graph import orchestrate
from call_graph.data_classes import CallTreeNode, custom_tree
from call_graph.gen_graph import make_graph
from client.llm2 import OllamaClient
from helpers.dict_to_csv import (
    save_dict_csv,
)  # will save the generated dictionary (containing the model's output..)
from helpers.extract_functions_from_c import (
    get_local_function_definitions,
)  # (to get function_names from c_files.ss)
from helpers.Preprocess.preprocess import (
    Preprocess,
    extract_all_macros,
    extract_includes,
)
from helpers.time_it import time_it
from makefile_resolver.makefile_resolver import return_project_mapping
from models import (
    Combined,
    FunctionTokenCount,
    Stats,
    TokenCount,
    aiDetermined,
    outputModel,
    outputModelForReturn,
)
from state.load_data import load_files
from state.state import State
from tools.tools import (
    set_tool_def,
)  # will set the tools and their definition in the state.

# set_tool_def()

# endregion imports

# =============================================================================
# HARD-CODED DPO SETTINGS
# =============================================================================

DPO_ATTEMPTS_PER_PATH = 5
DPO_MAX_CONCURRENT_AGENTS = 100

DPO_DATA_ROOT = Path("./dpo_llm_data")

DPO_SUPPRESS_AGENT_STDOUT = True
DPO_PRINT_COMPLETED_ATTEMPTS = True
DPO_PRINT_SELECTED_ATTEMPTS = True

DPO_FAILED_SCORE = -1_000_000


# region HELPERS (Identifying funs to trace and printing trees, dfs.)

# STATE = load_files()

comment_regex = r"(^|\s)(\/\/.*|\/\*[\s\S]*?\*\/)"
PROJECT_STRUCTURE = {}

# FUNCTION_POINTER_ARGS =
# logger = logging.getLogger(__name__)
console = Console()

GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
ORANGE = "\033[38;5;208m"  # 38;5;208m is an 8-bit color code for orange
RESET = "\033[0m"

# Parses the given source code bytes using Tree-sitter and walks the syntax tree.
# Collects and returns the names of all function calls found in the code.
def extract_function_calls(code: bytes) -> list[str]:
    """
    RETURNS A LIST OF FUNCTION CALLED IN THIS CODE.
    """
    # import from tree_sitter_customustompp as tsc
    import tree_sitter_custom as tsc
    from tree_sitter import Language, Parser

    # init tree sitter
    lang = Language(tsc.language())
    parser = Parser(lang)
    tree = parser.parse(code)

    calls: list[str] = []

    # recursive function to extract function calls given a tree
    def traverse(node):
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node:
                name = code[func_node.start_byte : func_node.end_byte].decode(
                    "latin-1", errors="replace"
                )
                calls.append(name)

        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    return calls


def identify_funs_to_trace(
    project_structure: dict[str, str],
    trees: dict,
    name_of_json: str = "json_data/mpf_data.json",
) -> (
    dict[str, dict[str, any]] | None
):  # will return {function_name, [list of indice of arguments to trace...]}

    # file_path = Path(name_of_json)

    # This is runtime state, it can only be set once, setting it again would fail
    # Acts like a normal dict?
    STATE = State()
    functions_to_detect = STATE.get("FUNCTION_TYPES")

    ans = {}  # {function_name, [indices to trace for it....]}
    if not functions_to_detect:
        print(f"Data 'FUNCTION_TYPES' Not in state.")

    # Start going through each file in the project
    for file_name, file_path_str in project_structure.items():
        if file_name.endswith(".h"):
            continue  # as we don't look at the function declared in the header files...

        # read each file and extract the functions
        bytes_content = trees[file_name][1]  # content
        functions_called = extract_function_calls(bytes_content)
        # print('Functions called in ',file_name,functions_called)
        # sys.exit()

        # only care about the functions we have to track
        for func in functions_to_detect:

            if func in functions_called:
                list_of_indices = [
                    ind for ind in functions_to_detect[func].get("indices") 
                    # TODO: What are indices here?
                    # Probably, all of their locations in the given project
                ]
                get_upper = functions_to_detect[func].get("get_upper") 
                # TODO: Get upper? uppercase? 
                # FIXED: I know, upper means if we have to read the code above it or below it, if its an initiator/closer then we to trace above it
                # Otherwise if its a creator of something that will be used downstream, we need to go below it 

                # for argument in functions_to_detect[func]:
                #     if isinstance(argument.get('indices'),list):
                #         list_of_indices.append(argument[1]+1) # +1 as its and index to convert it to 1 based indexing.
                ans[func] = {
                    "indices": list_of_indices,
                    "get_upper": get_upper,
                    "dependent_functions": functions_to_detect[func].get(
                        "dependent_functions"
                    ),  # list of str
                }

    return ans


def dfs_on_path_trees(
    tree_node: custom_tree,
    dependent_function: str,
    str_path: list[str],
    curr_path: list[str],
) -> bool:
    if not tree_node:
        return False

    curr_path.append(tree_node.name)

    found = False

    if tree_node.get_name == dependent_function:
        str_path.clear()
        str_path.extend(curr_path)
        found = True

    for child in tree_node.children:
        if dfs_on_path_trees(child, dependent_function, str_path, curr_path):
            found = True

    curr_path.pop()
    return found


def make_tree_custom(node: CallTreeNode) -> custom_tree:
    t = custom_tree(name=node.get_display_label)
    for children in node.children:
        t.add(make_tree_custom(node=children))
    return t


def make_tree(
    path: list[str] | tuple[int, list[CallTreeNode]],
    dependent_function: str | None = None,
) -> Tree | custom_tree | tuple[list, int]:
    tree = None
    last_tree = None
    if isinstance(path, tuple):
        index, path_list = path
        for node in path_list:

            # we need to traverse...
            if tree is None:
                tree = make_tree_custom(node=node)
                last_tree = tree
            else:
                child = make_tree_custom(node=node)
                last_tree.add(child)
                last_tree = child
        print(f"PROCESSED TREE FOR PATH NUM_{index+1}")
        # return tree,index
        result_path: list[str] = []
        dfs_on_path_trees(
            tree_node=tree,
            dependent_function=dependent_function,
            str_path=result_path,
            curr_path=[],
        )
        return result_path, index
    else:
        for node in path:
            if isinstance(node, str):
                if tree is None:
                    tree = Tree(escape(node))
                    last_tree = tree
                else:
                    child = Tree(escape(node))
                    last_tree.add(child)
                    last_tree = child

        # console.print(tree)
        return tree


@time_it()
def print_or_return_possible_paths_trees(
    paths: list[list[str | CallTreeNode]],
    dependent_function: str | None = None,
    result_path_list: list[list[str]] | None = None,
) -> None | custom_tree:
    # will print rich trees made from the found paths...
    # console.print(paths)
    from functools import partial

    if isinstance(paths[0][0], CallTreeNode):
        with multiprocessing.Pool(
            processes=min(10, multiprocessing.cpu_count())
        ) as pool:
            print("Using multiprocessing for processing paths")
            result_iter = pool.imap_unordered(
                partial(make_tree, dependent_function=dependent_function),
                enumerate(paths),
                chunksize=2,
            )
            for result_list, index in result_iter:
                result_path_list.append(result_list)
                # if index%100==0:
                #     gc.collect()
        return None
        # for i,path in enumerate(paths,start = 1):
        #     tree,index = make_tree((i,path))
        #     dependent_path: list[str] = []
        #     dfs_on_path_trees(tree_node=tree,dependent_function=dependent_function,str_path=dependent_path,curr_path=[])
        #     result_path_list.append(dependent_path)
        # return None
    for i, path in enumerate(paths, start=1):
        console.print(f"[bold red]PATH_{i}[/bold red]")
        # console.print(make_tree(path = path))
        console.print(make_tree(path=path))


# endregion


# --- AGENT LOGIC ---
def llm_calls(
    project_structure: dict[str, str],
    function_name_to_traced,
    argument_numbers: list[int],
    intial_context: str,
    path: str,
    get_upper: bool = True,
    collect_history: bool = False,
):
    STATE = State()
    messages_with_path_without_return = [
        {
            "role": "system",
            "content": """
            You are a **Static Backward Tracer** — a virtual compiler agent whose sole job is to determine the concrete runtime values of specific arguments passed to a target function, by tracing data flow backward through a provided call chain.

            ---

            ## CORE OPERATING PRINCIPLE

            You think like a c-compiler executing in **reverse**: you start at the **target function call-site**, identify which variables feed into the requested argument positions, and chase each variable's value backward through assignments, parameters, and callers — until you hit a **concrete literal value** or exhaust all resolution avenues (in which case you report `UNRESOLVED`).

            ---

            ## STEP-BY-STEP PROCEDURE

            ### Phase 1 — Orient

            1. Read the **CALL_GRAPH** (an ordered sequence of functions from `main()` → … → `target_function`).  
            2. Read the **INITIAL_CONTEXT** (trimmed source bodies of those functions).  
            3. Read any **MACRO INFO** block (expansions / callbacks / constants listed in comments at the top of the context).   
            4. Note which **argument indices** (1-based) of the target function you must resolve.

            ### Phase 2 — Locate the CORRECT Call-Site at EVERY Edge (CRITICAL)

            > **THIS PHASE IS THE MOST COMMON SOURCE OF ERRORS. FOLLOW PRECISELY.**
            > **This applies at EVERY hop in the CALL_GRAPH, not just the final one.**

            Given a CALL_GRAPH: `F1 → F2 → F3 → … → Fn (target)`

            At **each edge** `Fi → Fi+1`, function `Fi`'s body may contain **multiple calls**
            to `Fi+1`. You MUST select the correct call-site at **every such edge** using
            these rules in priority order:

            **RULE A (HIGHEST PRIORITY):**  
            Scan `Fi`'s **ENTIRE** body, from first line to last line, for calls to `Fi+1`
            that bear the annotation:  
            /*CONSIDER THIS CALL*/
            If **ANY** call to `Fi+1` inside `Fi` has this annotation, **select THAT call**
            and **IGNORE every other call** to `Fi+1` within `Fi`. No exceptions.

            > **Example — annotations at MULTIPLE edges:**
            > ```c
            > // CALL_GRAPH: main → DioGetPtr → mpf_mfs_open
            >
            > int main() {
            >     DioDbknr = (SdbDbknr *)DioGetPtr( SDB_FILENO_DBKNR, 0 );
            >     DioDcdef = (SdbDcdef *)DioGetPtr( SDB_FILENO_DCDEF, 0 );
            >     DioHealth = (Health  *)DioGetPtr( FNO_HEALTH, 0 ); /*CONSIDER THIS CALL*/
            > }
            >
            > void *DioGetPtr( int filenum, int sbnum ) {
            >     ret = mpf_mfs_open( &fcb, NULL, filenum, sbnum, 0, MPF_MFS_READLOCK ); /*CONSIDER THIS CALL*/
            > }
            > ```
            >
            > **Edge `main → DioGetPtr`:**  
            >  Select `DioGetPtr( FNO_HEALTH, 0 )` — has annotation.  
            >  NOT `DioGetPtr( SDB_FILENO_DBKNR, 0 )` — first call, not annotated.
            >
            > **Edge `DioGetPtr → mpf_mfs_open`:**  
            >  Select `mpf_mfs_open( &fcb, NULL, filenum, sbnum, 0, MPF_MFS_READLOCK )` — has annotation.
            >
            > Now tracing argument 3 of `mpf_mfs_open`:
            > → `filenum` (arg 3) is a parameter of `DioGetPtr` at index 1
            > → Jump to caller `main`, at the selected call-site: `DioGetPtr( FNO_HEALTH, 0 )`
            > → Arg 1 = `FNO_HEALTH` → macro → resolve via `find_definition` if not in the comments (But finding a value is not guranteed).


            ### Phase 3 — Backward Trace (the core loop)

            For **each** argument index you must resolve, do the following starting at the target call-site:
            current_value ← the expression at that argument position in the call
            current_function ← the caller

            **Repeat:**

            a. **Literal / constant?**  
            → `current_value` is a numeric literal, string literal, enum literal, `NULL`, `true`/`false`, etc.  
            → **Stop. Record this concrete value. Do NOT trace further up.**

            b. **Variable?**  
            → Search **backward** (above the call-site, within `current_function`) for the **last assignment** to this variable before the call.  
            - If found and assigned a **literal** → Stop, record it.  
            - If found and assigned the **return value of a function not in context** → use `find_definition` (see Tool Use below) to inspect that function, then resolve.  
            - If found and assigned **another variable or expression** → set `current_value` to that expression and continue the loop in the same function.  
            - If **not found** (the variable is a **parameter** of `current_function`) → identify which parameter index it is, then jump to the **caller of `current_function`** (next function up the CALL_GRAPH). At the call-site in that caller, pick the expression at the corresponding argument position. Set `current_value` to that expression, set `current_function` to that caller, and continue.

            c. **Macro / preprocessor symbol?**  
            → Check the **MACRO INFO** section first.  
            → If not there, call `find_definition` for the macro name.  
            → Replace with the resolved literal and stop, or mark `UNRESOLVED`.

            d. **Function call / complex expression?**  
            → If the function is in context, trace into it to find the return value.  
            → If not, use `find_definition` to get its definition and inspect.

            e. **Modified by an intervening function call** (e.g., passed by pointer/reference to a function between assignment and use)?  
            → If that modifying function is in context, trace the modification.  
            → If not, use `find_definition` **once** to retrieve it, then trace.

            ### Phase 4 — Macro-Expanded Call Chains

            If the MACRO INFO section states:
            func_a(a,b,c,d) (macro expansion)-> func_b(FILE,a,b,c,d) 
            macro_name = constant value (If any)

            then when the CALL_GRAPH passes through `func_a` → `func_b`, understand that:
            - `func_b`'s 1st argument is `FILE` (injected by the macro).
            - `func_b`'s 2nd argument corresponds to `func_a`'s 1st argument, and so on (shifted by the number of injected args).

            Adjust your argument-index mapping accordingly when crossing this boundary.

            ### Phase 5 — Resolve `call_number`

            After (or during) your trace, scan the **entire provided context only** for any invocation of:
            - `pmf_addevent(...)`, or  
            - `pmd_addvarevt(...)`

            If **either** is present anywhere in the context:
            - Resolve the **1st argument**(Usually a macro or constant use find_definition for resolving) (1-based) of that function call using the exact same backward-tracing procedure above.
            - Report that resolved value as `call_number`.

            If **neither** function appears anywhere in the context:
            - Report `call_number` as `None`.

            ### Phase 6 — Report

            Produce a final structured answer containing:
            - For each requested argument index: the **concrete resolved value** (literal), or `UNRESOLVED`.
            - `call_number`: the resolved 1st-argument value of `pmf_addevent`/`pmd_addvarevt`, or `None`.

            ---

            ## EARLY TERMINATION RULE

            You are **guaranteed** by the call graph that the path from `main()` to the target function is valid. You do **not** need to verify reachability. Therefore:
            - **Stop tracing upward the moment you resolve a value to a concrete literal.**  
            - Do **not** trace all the way to `main()` unless the data dependency genuinely flows that far without being assigned a constant anywhere along the chain.
            - Don't report all the arguments of a given function only those that are asked.


            ---

            ## IMPORTANT CONSTRAINTS (DO's and DON'Ts)

            | # | Rule |
            |---|------|
            | 1 | **All argument indices are 1-based.** `func(a, b, c)` → index 1 = `a`, index 2 = `b`, index 3 = `c`. |
            | 2 | **Never report a macro name or variable name as a final value.** You must resolve to a literal or say `UNRESOLVED`. |
            | 3 | **One `find_definition` call per symbol.** No retries. |
            | 4 | **Follow the CALL_GRAPH path exactly.** Ignore other callers or other paths not in the specified sequence. |
            | 5 | **For multiple calls to the same function within one body**, use the one marked `/*CONSIDER THIS CALL*/`, or the **last** occurrence if unmarked. |
            | 6 | **Macro expansions** that inject/reorder arguments must be accounted for when mapping argument indices across the expansion boundary. |
            | 7 | **Do not guess or hallucinate values.** If resolution is impossible with available information, report `UNRESOLVED`. |
            | 8 | **Show your tracing work** step-by-step (function by function, assignment by assignment) before giving the final answer so the reasoning is auditable. |

            ---

            ### OUTPUT FORMAT
            Report only the final resolved value for each requested ARG_INDEX and the call_number.
            """,
        },
        {
            "role": "user",
            "content": """Backward trace argument numbers **{argument_numbers}**  of function **{function_name_to_traced}** and the call_number if present or else None.
                    (Argument number -1 represents the RETURN VALUE of the function.)
                    **INITIAL CONTEXT:{intial_context}**
                    **CALL_GRAPH**: {path}
                    """,
        },
    ]
    messages_with_path_with_return = [  # only for those function which require their return value's tracing.
        {
            "role": "system",
            "content": """You are a C code backward tracer. You trace the return value of a target function to determine what operation is performed on it at the call site (report only READ, WRITE) and the value of the `call_number`.
            ## WHAT YOU ARE GIVEN
                INITIAL_CONTEXT: Function bodies from main() down to the target function (INITIAL CONTEXT). Lines are trimmed. This is your primary source (BUT YOU CAN USE TOOLS TO RESOLVE MACROS AND SEE FUNCTION BODIES THAT CONSUME THE RETURN VALUE).
                Macro expansions are shown as comments if any.
                CALL_GRAPH: Function call graph — only follow this, always, even if another path exists.
                TRACING METHOD (FOLLOW THIS EXACTLY)

            STEP 1: FIND THE TARGET CALL
            Find the exact line where the target function is called. Identify how its return value is captured (or not) at the call site.

            STEP 2: CLASSIFY THE RETURN VALUE USAGE (CRITICAL)
            At the call site, examine what happens to the return value immediately:

            Classification rules (CHECK IN THIS ORDER):

            NOTHING: The return value is discarded entirely — the call is a standalone statement with no assignment or use. e.g., target_func(a, b);
            READ: The return value is consumed — assigned to a variable, used in a conditional, passed as an argument, compared, or used in any expression. e.g., x = target_func(...), if (target_func(...)), other_func(target_func(...)).
            WRITE: The return value (typically a pointer) is written into — something is stored through it. e.g., *target_func(...) = value;, target_func(...)->field = value;.
            STEP 3: IF THE RETURN VALUE IS ASSIGNED TO A VARIABLE, TRACE FORWARD
            If the return value is assigned to a variable (e.g., ret = target_func(...)), scan the function body from the assignment DOWNWARD to the function exit, looking for how ret is ultimately used. This determines whether the overall operation is READ or WRITE.

            Usage priority (CHECK IN THIS ORDER):

            Passed to another function: consumer(ret) → use tools to read consumer and determine if it reads or writes through ret.
            Dereferenced and written to: *ret = val; or ret->field = val; → WRITE.
            Used in expression/condition/return: if (ret), return ret;, x = ret + 1; → READ.
            **DONT'S**:
            - DON'T LOOK FOR THE FUNCTION BODIES THAT ARE ALREADY GIVEN IN THE COTEXT.
            Also determine the call_number trace which is the  `1st argument of the function **pmf_addevent** or **pmf_addvarevt**` (WHATEVER PRESENT)
            **EXAMPLE**:
            int RbtMfsOpenFunc(...) {
                char *buf;
                buf = target_func(a, b);   // <-- return value captured in buf
                memcpy(dest, buf, len);    // <-- buf is READ here
                // Overall: READ
            }
            int RbtMfsWriteFunc(...) {
                char *ptr;
                ptr = target_func(a, b);   // <-- return value captured in ptr
                *ptr = 0x00;               // <-- ptr is WRITTEN THROUGH here
                // Overall: WRITE
            }

        """,
        },
        {
            "role": "user",
            "content": """DETERMINE THE TYPE OF OPERATION PERFORMED ON THE RETURNED POINT OF FUNCTION {function_name_to_traced} and the call_number if present or else None..
                            Find ALL possible constant values that can reach this argument from main().
                            **INITIAL CONTEXT:{intial_context}**
                            **CALL_GRAPH**: {path}
                            """,
        },
    ]

    messages = (
        messages_with_path_without_return
        if get_upper
        else messages_with_path_with_return
    )

    # region MAKING CLIENT AND SENDING DATA.

    data = {
        "user_prompt": messages[1].get("content"),
        "system_prompt": messages[0].get("content"),
        "tools": STATE.get("TOOL_DEFINITION"),
        "tool_functions": STATE.get(
            "TOOLS"
        ),  # dict of {'function_name': function}  IF TOOLS ARE IN SEPERATE FILE THEN USE getattr(my_tools(MODULE), name) INSTEAD OF GLOBALS},
        "project_structure": project_structure,
        "function_map": STATE.get("FUNCTION_MAP"),
        "output_model": outputModel if get_upper else outputModelForReturn,
    }
    # print(get_upper,'get_upper', outputModel if get_upper else outputModelForReturn)
    client = OllamaClient(data=data)
    # endregion
    # region STARTING TOOL CHAIN AND SENDING PROMPT DATA
    prompt_data = {
        "user_prompt": {
            "argument_numbers": argument_numbers,
            "function_name_to_traced": function_name_to_traced,
            "intial_context": intial_context,
            "path": path,
        },
        "system_prompt": {},
    }

    ans, stats = client.start_tool_chain(prompt_data=prompt_data)
    # ans,stats = client.start_new_tool_chain(prompt_data=prompt_data)
    # endregion

    if collect_history:
        return ans, stats, client.messages

    return ans, stats


# =============================================================================
# DPO STYLE LLM COLLECTION + FLATTENED PATH/ATTEMPT CONCURRENCY
# Add this block after llm_calls(...)
# =============================================================================


import contextlib
import json
import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel


# =============================================================================
# JSON / FILE HELPERS
# =============================================================================

def dpo_json_safe(obj: Any) -> Any:
    """
    Convert Pydantic models, Path objects, OpenAI/LangChain-ish message objects,
    exceptions, tuples, dicts, etc. into JSON-safe data.
    """

    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, list):
        return [dpo_json_safe(x) for x in obj]

    if isinstance(obj, tuple):
        return [dpo_json_safe(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): dpo_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, BaseException):
        return {
            "exception_type": type(obj).__name__,
            "message": str(obj),
        }

    if hasattr(obj, "model_dump"):
        return dpo_json_safe(obj.model_dump())

    if hasattr(obj, "dict"):
        return dpo_json_safe(obj.dict())

    if hasattr(obj, "content"):
        return {
            "type": type(obj).__name__,
            "content": dpo_json_safe(getattr(obj, "content", None)),
            "additional_kwargs": dpo_json_safe(getattr(obj, "additional_kwargs", None)),
            "response_metadata": dpo_json_safe(getattr(obj, "response_metadata", None)),
        }

    return str(obj)


def dpo_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dpo_json_safe(data), f, ensure_ascii=False, indent=4)


def dpo_safe_name(value: Any, max_len: int = 120) -> str:
    value = str(value)
    value = re.sub(r"[^a-zA-Z0-9_.=-]+", "_", value)
    value = value.strip("_")
    return value[:max_len] or "unknown"

import hashlib


def dpo_hash_text(value: Any, length: int = 12) -> str:
    raw = json.dumps(dpo_json_safe(value), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def dpo_job_fingerprint(job: "DPOLLMJob") -> str:
    return dpo_hash_text(
        {
            "process_name": job.process_name,
            "function_name_to_traced": job.function_name_to_traced,
            "source_file": job.source_file,
            "line_number": job.line_number,
            "argument_numbers": job.argument_numbers,
            "path": job.path,
            "intial_context": job.intial_context,
            "get_upper": job.get_upper,
        },
        length=16,
    )


def dpo_job_dir(job: DPOLLMJob) -> Path:
    process_dir = dpo_safe_name(job.process_name)

    src_stem = "unknown_src"
    if job.source_file:
        src_stem = dpo_safe_name(Path(job.source_file).stem)

    try:
        line_number = int(job.line_number) if job.line_number is not None else None
    except Exception:
        line_number = None

    line = line_number if line_number is not None else "unknown_line"

    path_hash = dpo_hash_text(job.path, length=8)
    context_hash = dpo_hash_text(job.intial_context, length=8)
    fingerprint = dpo_job_fingerprint(job)

    folder_name = (
        f"{dpo_safe_name(job.function_name_to_traced)}"
        f"__{src_stem}"
        f"__line_{line}"
        f"__path_{job.path_index:04d}"
        f"__p_{path_hash}"
        f"__c_{context_hash}"
        f"__fp_{fingerprint}"
    )

    return DPO_DATA_ROOT / process_dir / folder_name

def dpo_attempt_dir(job: DPOLLMJob, attempt_no: int) -> Path:
    return dpo_job_dir(job) / f"attempt_{attempt_no:02d}"


def dpo_selected_path(job: DPOLLMJob) -> Path:
    return dpo_job_dir(job) / "selected.json"

def dpo_answer_canonical_key(answer: Any) -> str:
    """
    Stable comparable key for majority vote.
    """
    safe = dpo_json_safe(answer)

    if isinstance(safe, dict):
        # Usually enough for your outputModel/outputModelForReturn
        comparable = {
            "output": safe.get("output"),
            "call_number": safe.get("call_number"),
        }
    else:
        comparable = safe

    return json.dumps(comparable, sort_keys=True, ensure_ascii=False)


def dpo_select_attempt_majority(attempt_results: list[DPOAttemptResult]) -> DPOAttemptResult:
    """
    Selection rule:
      1. Ignore failed attempts if any successful exist.
      2. Pick most common answer.
      3. If all successful answers are unique, randomly sample one successful.
      4. If all failed, randomly sample one failed.
    """
    successful = [
        r for r in attempt_results
        if r.answer is not None and r.error is None
    ]

    if not successful:
        return random.choice(attempt_results)

    counts = Counter(dpo_answer_canonical_key(r.answer) for r in successful)

    most_common_key, most_common_count = counts.most_common(1)[0]

    # all unique
    if most_common_count == 1:
        return random.choice(successful)

    matching = [
        r for r in successful
        if dpo_answer_canonical_key(r.answer) == most_common_key
    ]

    # if same answer appears multiple times, choose highest score version
    return max(matching, key=lambda r: r.score)

# =============================================================================
# DPO DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class DPOLLMJob:
    job_key: str
    process_name: str
    path_index: int

    project_structure: dict[str, str]
    function_name_to_traced: str
    argument_numbers: list[int]
    intial_context: str
    path: str
    get_upper: bool = True

    # optional metadata for folder naming
    source_file: str | None = None
    line_number: int | None = None

@dataclass
class DPOAttemptResult:
    job_key: str
    process_name: str
    path_index: int
    function_name_to_traced: str
    attempt_no: int

    score: int
    answer: Any
    stats: Any

    history_path: str | None
    score_path: str | None
    answer_path: str | None

    error_path: str | None = None
    error: str | None = None


# =============================================================================
# DPO FOLDER LAYOUT
# =============================================================================


# =============================================================================
# DPO SCORING
# =============================================================================

def dpo_score_attempt(answer: Any, stats: Any) -> dict[str, Any]:
    """
    Higher score is better.

    Uses your existing models:
      - outputModel / outputModelForReturn
      - Stats
      - TokenCount
    """

    score = 0
    reasons: list[str] = []

    answer_dict = dpo_json_safe(answer)
    stats_dict = dpo_json_safe(stats)

    if isinstance(answer_dict, dict):
        output = str(answer_dict.get("output", ""))
        call_number = answer_dict.get("call_number", None)
    else:
        output = ""
        call_number = None

    if isinstance(stats_dict, dict):
        iterations = int(stats_dict.get("Iterations", 0) or 0)
        random_tool_calls = int(stats_dict.get("Random_tool_calls", 0) or 0)
        other_tool_errors = int(stats_dict.get("Other_tool_errors", 0) or 0)

        tokens = stats_dict.get("Tokens", {}) or {}
        if isinstance(tokens, dict):
            total_tokens = int(tokens.get("Total_tokens", 0) or 0)
        else:
            total_tokens = 0
    else:
        iterations = 0
        random_tool_calls = 0
        other_tool_errors = 0
        total_tokens = 0

    if answer is not None:
        score += 100
        reasons.append("+100 valid_answer_model")
    else:
        score -= 500
        reasons.append("-500 no_answer_model")

    unresolved_count = output.upper().count("UNRESOLVED")

    if not output:
        score -= 200
        reasons.append("-200 empty_output")
    elif unresolved_count == 0:
        score += 50
        reasons.append("+50 no_unresolved")
    else:
        penalty = unresolved_count * 25
        score -= penalty
        reasons.append(f"-{penalty} unresolved_count={unresolved_count}")

    if output in {"READF", "WRITEF"}:
        score += 40
        reasons.append(f"+40 return_usage_resolved={output}")

    if call_number not in [None, "None", "NONE", "UNRESOLVED", ""]:
        score += 10
        reasons.append(f"+10 call_number_resolved={call_number}")

    if random_tool_calls:
        penalty = random_tool_calls * 30
        score -= penalty
        reasons.append(f"-{penalty} random_tool_calls={random_tool_calls}")

    if other_tool_errors:
        penalty = other_tool_errors * 25
        score -= penalty
        reasons.append(f"-{penalty} other_tool_errors={other_tool_errors}")

    if iterations:
        penalty = max(0, iterations - 1) * 2
        score -= penalty
        reasons.append(f"-{penalty} iterations={iterations}")

    if total_tokens:
        penalty = total_tokens // 20_000
        score -= penalty
        reasons.append(f"-{penalty} token_penalty_total_tokens={total_tokens}")

    return {
        "score": score,
        "reasons": reasons,
        "answer": answer_dict,
        "stats": stats_dict,
    }


# =============================================================================
# SINGLE ATTEMPT WORKER
# =============================================================================

def dpo_run_single_attempt(job: DPOLLMJob, attempt_no: int) -> DPOAttemptResult:
    """
    Runs exactly one independent LLM attempt for one path/job.

    Saves:
      - history.json
      - answer.json
      - score.json

    On failure:
      - error.json
    """

    attempt_dir = dpo_attempt_dir(job, attempt_no)

    history_path = attempt_dir / "history.json"
    score_path = attempt_dir / "score.json"
    answer_path = attempt_dir / "answer.json"
    error_path = attempt_dir / "error.json"

    try:
        if DPO_SUPPRESS_AGENT_STDOUT:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    answer, stats, history = llm_calls(
                        project_structure=job.project_structure,
                        function_name_to_traced=job.function_name_to_traced,
                        argument_numbers=job.argument_numbers,
                        intial_context=job.intial_context,
                        path=job.path,
                        get_upper=job.get_upper,
                        collect_history=True,
                    )
        else:
            answer, stats, history = llm_calls(
                project_structure=job.project_structure,
                function_name_to_traced=job.function_name_to_traced,
                argument_numbers=job.argument_numbers,
                intial_context=job.intial_context,
                path=job.path,
                get_upper=job.get_upper,
                collect_history=True,
            )

        score_data = dpo_score_attempt(answer=answer, stats=stats)

        dpo_write_json(
            history_path,
            {
                "job_key": job.job_key,
                "process_name": job.process_name,
                "path_index": job.path_index,
                "function_name_to_traced": job.function_name_to_traced,
                "argument_numbers": job.argument_numbers,

                "path": job.path,
                "path_hash": dpo_hash_text(job.path),
                "intial_context_hash": dpo_hash_text(job.intial_context),
                "job_fingerprint": dpo_job_fingerprint(job),
                "get_upper": job.get_upper,
                "attempt_no": attempt_no,

                "messages": history,
                "answer": answer,
                "stats": stats,
            },
        )

        dpo_write_json(
            answer_path,
            {
                "job_key": job.job_key,
                "process_name": job.process_name,
                "path_index": job.path_index,
                "function_name_to_traced": job.function_name_to_traced,
                "attempt_no": attempt_no,
                "answer": answer,
            },
        )

        dpo_write_json(
            score_path,
            {
                "job_key": job.job_key,
                "process_name": job.process_name,
                "path_index": job.path_index,
                "function_name_to_traced": job.function_name_to_traced,
                "attempt_no": attempt_no,
                **score_data,
                "history_path": str(history_path),
                "answer_path": str(answer_path),
            },
        )

        return DPOAttemptResult(
            job_key=job.job_key,
            process_name=job.process_name,
            path_index=job.path_index,
            function_name_to_traced=job.function_name_to_traced,
            attempt_no=attempt_no,
            score=int(score_data["score"]),
            answer=answer,
            stats=stats,
            history_path=str(history_path),
            score_path=str(score_path),
            answer_path=str(answer_path),
        )

    except Exception as e:
        dpo_write_json(
            error_path,
            {
                "job_key": job.job_key,
                "process_name": job.process_name,
                "path_index": job.path_index,
                "function_name_to_traced": job.function_name_to_traced,
                "attempt_no": attempt_no,
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
        )

        return DPOAttemptResult(
            job_key=job.job_key,
            process_name=job.process_name,
            path_index=job.path_index,
            function_name_to_traced=job.function_name_to_traced,
            attempt_no=attempt_no,
            score=DPO_FAILED_SCORE,
            answer=None,
            stats=None,
            history_path=None,
            score_path=None,
            answer_path=None,
            error_path=str(error_path),
            error=str(e),
        )


# =============================================================================
# FLATTENED EXECUTOR: ALL PATHS X ALL ATTEMPTS
# =============================================================================

def resolve_all_paths_with_dpo_flat(
    specs: list[dict[str, Any]],
) -> dict[str, tuple[Any, Any, DPOAttemptResult]]:
    """
    Real global/local flattening.

    specs item format:

    {
        "process_name": str,
        "project_structure": dict[str, str],
        "function_name_to_traced": str,
        "argument_numbers": list[int],
        "path_index": int,
        "path": str,
        "intial_context": str,
        "get_upper": bool,
        "source_file": str | None,
        "line_number": int | None,

        # optional but recommended:
        "raw_path": list[str],

        # optional:
        "job_key": str,
    }

    Returns:
        {
            job_key: (selected_answer, selected_stats, selected_attempt_metadata)
        }
    """

    def coerce_line_number(value):
        if value is None:
            return None

        try:
            return int(value)
        except Exception:
            return None

    def extract_meta_from_path_node(node: str):
        """
        Handles nodes like:

            [Dio860d.c]main[963:1035]
            [Dio860d.c:1024]Dio860dGsim1[657:958]
            [Dio860d.c:773]Dio860dTcsSend[514:573]
            [548]mpf_mfs_addque

        Returns:
            source_file, line_number, function_name
        """

        if not node:
            return None, None, None

        # [Dio860d.c:773]Dio860dTcsSend[514:573]
        m = re.match(
            r"^\[(?P<file>[^:\]\[]+\.[A-Za-z0-9_]+):(?P<line>\d+)\](?P<func>[A-Za-z_][A-Za-z0-9_]*)",
            node,
        )
        if m:
            return m.group("file"), int(m.group("line")), m.group("func")

        # [Dio860d.c]main[963:1035]
        m = re.match(
            r"^\[(?P<file>[^\]\[]+\.[A-Za-z0-9_]+)\](?P<func>[A-Za-z_][A-Za-z0-9_]*)",
            node,
        )
        if m:
            source_file = m.group("file")
            func_name = m.group("func")

            # recover start line from trailing [963:1035]
            range_match = re.search(r"\[(?P<start>\d+):(?P<end>\d+)\]\s*$", node)
            line_number = int(range_match.group("start")) if range_match else None

            return source_file, line_number, func_name

        # [548]mpf_mfs_addque
        m = re.match(
            r"^\[(?P<line>\d+)\](?P<func>[A-Za-z_][A-Za-z0-9_]*)",
            node,
        )
        if m:
            return None, int(m.group("line")), m.group("func")

        return None, None, None

    def infer_source_meta_from_raw_path(
        *,
        raw_path: list[str] | None,
        target_function: str,
        fallback_source_file: str | None,
        fallback_line_number: int | None,
    ):
        """
        Infers source file and line number from the raw call path.

        Important:
            clean_path_str() removes bracket metadata.
            So this only works if spec["raw_path"] is passed.
        """

        if not raw_path:
            return fallback_source_file, fallback_line_number

        last_seen_file = fallback_source_file
        found_file = fallback_source_file
        found_line = fallback_line_number

        for node in raw_path:
            source_file, line_number, func_name = extract_meta_from_path_node(node)

            if source_file:
                last_seen_file = source_file

            if func_name == target_function:
                found_file = source_file or last_seen_file or fallback_source_file

                if line_number is not None:
                    found_line = line_number

        return found_file, found_line

    jobs: list[DPOLLMJob] = []

    for spec in specs:
        process_name = spec["process_name"]
        function_name = spec["function_name_to_traced"]
        path_index = int(spec["path_index"])
        path_str = spec["path"]
        intial_context = spec["intial_context"]

        source_file = spec.get("source_file")
        line_number = coerce_line_number(spec.get("line_number"))

        source_file, line_number = infer_source_meta_from_raw_path(
            raw_path=spec.get("raw_path"),
            target_function=function_name,
            fallback_source_file=source_file,
            fallback_line_number=line_number,
        )

        line_number = coerce_line_number(line_number)

        path_hash = dpo_hash_text(path_str, length=8)
        context_hash = dpo_hash_text(intial_context, length=8)

        src_stem = Path(source_file).stem if source_file else "unknown_src"
        line = line_number if line_number is not None else "unknown_line"

        job_key = spec.get("job_key")

        if not job_key:
            job_key = (
                f"{process_name}::"
                f"{function_name}::"
                f"{src_stem}::"
                f"line_{line}::"
                f"path_{path_index:04d}::"
                f"p_{path_hash}::"
                f"c_{context_hash}"
            )

        jobs.append(
            DPOLLMJob(
                job_key=job_key,
                process_name=process_name,
                path_index=path_index,
                project_structure=spec["project_structure"],
                function_name_to_traced=function_name,
                argument_numbers=spec["argument_numbers"],
                intial_context=intial_context,
                path=path_str,
                get_upper=spec.get("get_upper", True),
                source_file=source_file,
                line_number=line_number,
            )
        )

    best_by_job = dpo_run_llm_jobs_flat(jobs)

    return {
        job_key: (best.answer, best.stats, best)
        for job_key, best in best_by_job.items()
    }

def dpo_run_llm_jobs_flat(
    jobs: list[DPOLLMJob],
) -> dict[str, DPOAttemptResult]:
    """
    Flattens all jobs and attempts into one global ProcessPoolExecutor.

    Example:
        10 paths x 5 attempts = 50 scheduled tasks

    But only DPO_MAX_CONCURRENT_AGENTS run at once.
    """

    if not jobs:
        return {}

    results_by_job: dict[str, list[DPOAttemptResult]] = {
        job.job_key: [] for job in jobs
    }

    job_by_key: dict[str, DPOLLMJob] = {
        job.job_key: job for job in jobs
    }

    total_attempts = len(jobs) * DPO_ATTEMPTS_PER_PATH

    print(
        f"[DPO START] jobs={len(jobs)} "
        f"attempts_per_job={DPO_ATTEMPTS_PER_PATH} "
        f"total_attempts={total_attempts} "
        f"max_concurrent_agents={DPO_MAX_CONCURRENT_AGENTS} "
        f"root={DPO_DATA_ROOT.resolve()}"
    )

    with ProcessPoolExecutor(max_workers=DPO_MAX_CONCURRENT_AGENTS) as executor:
        future_to_meta = {}

        for job in jobs:
            for attempt_no in range(1, DPO_ATTEMPTS_PER_PATH + 1):
                future = executor.submit(dpo_run_single_attempt, job, attempt_no)
                future_to_meta[future] = (job.job_key, attempt_no)

        completed = 0

        for future in as_completed(future_to_meta):
            completed += 1
            job_key, attempt_no = future_to_meta[future]

            try:
                result = future.result()
            except Exception as e:
                job = job_by_key[job_key]
                error_dir = dpo_attempt_dir(job, attempt_no)
                error_path = error_dir / "executor_error.json"

                dpo_write_json(
                    error_path,
                    {
                        "job_key": job_key,
                        "attempt_no": attempt_no,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )

                result = DPOAttemptResult(
                    job_key=job_key,
                    process_name=job.process_name,
                    path_index=job.path_index,
                    function_name_to_traced=job.function_name_to_traced,
                    attempt_no=attempt_no,
                    score=DPO_FAILED_SCORE,
                    answer=None,
                    stats=None,
                    history_path=None,
                    score_path=None,
                    answer_path=None,
                    error_path=str(error_path),
                    error=str(e),
                )

            results_by_job[result.job_key].append(result)

            if DPO_PRINT_COMPLETED_ATTEMPTS:
                artifact = result.score_path or result.error_path
                print(
                    f"[DPO ATTEMPT DONE] "
                    f"{completed}/{total_attempts} "
                    f"job={result.job_key} "
                    f"attempt={result.attempt_no} "
                    f"score={result.score} "
                    f"artifact={artifact}"
                )

    best_by_job: dict[str, DPOAttemptResult] = {}

    for job in jobs:
        attempt_results = results_by_job.get(job.job_key, [])

        if not attempt_results:
            continue

        best = dpo_select_attempt_majority(attempt_results)
        best_by_job[job.job_key] = best

        selected_path = dpo_selected_path(job)

        dpo_write_json(
            selected_path,
            {
                "job_key": job.job_key,
                "process_name": job.process_name,
                "path_index": job.path_index,
                "function_name_to_traced": job.function_name_to_traced,
                "source_file": job.source_file,
                "line_number": job.line_number,
                "argument_numbers": job.argument_numbers,
                "path": job.path,
                "get_upper": job.get_upper,
                "selection_policy": "majority_answer_else_random_unique_else_random_failed",
                "selected_attempt": best.attempt_no,
                "selected_score": best.score,
                "selected_answer": best.answer,
                "selected_stats": best.stats,
                "selected_history_path": best.history_path,
                "selected_score_path": best.score_path,
                "selected_answer_path": best.answer_path,
                "all_attempts": [
                    {
                        "attempt_no": r.attempt_no,
                        "score": r.score,
                        "history_path": r.history_path,
                        "score_path": r.score_path,
                        "answer_path": r.answer_path,
                        "error_path": r.error_path,
                        "error": r.error,
                    }
                    for r in sorted(attempt_results, key=lambda x: x.attempt_no)
                ],
            },
        )

        if DPO_PRINT_SELECTED_ATTEMPTS:
            print(
                f"[DPO SELECTED] "
                f"job={job.job_key} "
                f"path={job.path_index} "
                f"function={job.function_name_to_traced} "
                f"attempt={best.attempt_no} "
                f"score={best.score} "
                f"selected={selected_path}"
            )

    return best_by_job


# =============================================================================
# MAIN DROP-IN BATCH RESOLVER
# =============================================================================

def resolve_paths_with_dpo(
    *,
    process_name: str,
    project_structure: dict[str, str],
    function_name_to_traced: str,
    argument_numbers: list[int],
    path_contexts: list[tuple[int, str, str]],
    get_upper: bool = True,
) -> dict[int, tuple[Any, Any, DPOAttemptResult]]:
    """
    Main helper you should call from your existing main/path loop.

    Args:
        process_name:
            Project/process name.

        project_structure:
            Same project_structure currently passed to llm_calls.

        function_name_to_traced:
            Target function name currently passed to llm_calls.

        argument_numbers:
            Same argument number list currently passed to llm_calls.

        path_contexts:
            List of tuples:
                [
                    (path_index, path_str, intial_context),
                    ...
                ]

        get_upper:
            Same get_upper currently passed to llm_calls.

    Returns:
        {
            path_index: (best_answer, best_stats, best_attempt_metadata)
        }

    The returned answer and stats are from ONLY the selected best attempt.
    """

    jobs: list[DPOLLMJob] = []

    for path_index, path_str, intial_context in path_contexts:
        path_hash = dpo_hash_text(path_str, length=8)
        context_hash = dpo_hash_text(intial_context, length=8)

        job_key = (
            f"{process_name}::"
            f"{function_name_to_traced}::"
            f"path_{path_index:04d}::"
            f"p_{path_hash}::"
            f"c_{context_hash}"
        )

        jobs.append(
            DPOLLMJob(
                job_key=job_key,
                process_name=process_name,
                path_index=path_index,
                project_structure=project_structure,
                function_name_to_traced=function_name_to_traced,
                argument_numbers=argument_numbers,
                intial_context=intial_context,
                path=path_str,
                get_upper=get_upper,
            )
        )

    best_by_job = dpo_run_llm_jobs_flat(jobs)

    selected_by_path: dict[int, tuple[Any, Any, DPOAttemptResult]] = {}

    for job in jobs:
        best = best_by_job.get(job.job_key)

        if best is None:
            continue

        selected_by_path[job.path_index] = (
            best.answer,
            best.stats,
            best,
        )

    return selected_by_path

def run_with_retry(func, args=(), timeout=180, retries=2):
    if not isinstance(args, (tuple, list)):
        args = (args,)

    for attempt in range(retries):
        # Create the pipe inside the loop so each attempt gets a fresh connection
        parent_conn, child_conn = multiprocessing.Pipe()

        def target_wrapper(conn, *func_args):
            try:
                result = func(*func_args)
                conn.send({"status": "success", "data": result})
            except Exception as e:
                conn.send({"status": "error", "data": str(e)})
            finally:
                conn.close()

        process = multiprocessing.Process(
            target=target_wrapper, args=(child_conn, *args)
        )
        process.start()

        process.join(timeout)

        # 1. Handle Timeout
        if process.is_alive():
            print(f"⚠️ Attempt {attempt + 1} timed out. Killing process...")
            process.terminate()
            process.join()
            parent_conn.close()  # Clean up pipe
            continue

        # 2. Retrieve Data Safely
        response = None
        try:
            if parent_conn.poll():
                response = parent_conn.recv()
        except EOFError:
            # This happens if the process dies after poll() but before recv()
            response = None
        finally:
            parent_conn.close()  # Always close the parent end after usage

        # 3. Validation - This fixes the 'NoneType' error
        if response is None:
            print(
                f"❌ Attempt {attempt + 1} failed: Process exited without sending data."
            )
            continue

        if response.get("status") == "success":
            return response.get("data")
        else:
            print(f"❌ Attempt {attempt + 1} failed with error: {response.get('data')}")
            continue

    return None


def make_llm_calls_for_function(
    function,
    trees: dict,
    functions_identified: dict[str, dict[str, any]],
    answers: dict[str, list[tuple[BaseModel, BaseModel]]],
    main_file_name: str,
    function_pointer_args,
    file_functions,
    project_structure,
    project_path,
) -> list | None:
    # will return list of dataframes containing all data to be saved in csv.

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # These are used inside DPO helper functions elsewhere.
    # Importing locally is not enough if those functions reference global names,
    # so we also inject into globals().
    # -------------------------------------------------------------------------
    import random as _random
    from collections import Counter as _Counter

    globals()["random"] = _random
    globals()["Counter"] = _Counter

    if ("(" in function) or (")" in function):
        answers[function] = [
            (
                "Not a valid function_n name:: [ENTER THE FUNCTION NAME WITHOUT '()']",
                None,
            )
        ]
        return None

    answers.setdefault(function, [])

    print(f"PROCESSING FUNCTION -->{BOLD}{GREEN}", function, f"{RESET}", end="\n\n")

    STATE = State()

    FILE_NAME_BYTES: dict[str, bytes] = {
        key: value[1] for key, value in STATE.get("TREES").items()
    }

    process_name = Path(project_path).name

    list_indices = functions_identified[function].get("indices")
    get_upper = functions_identified[function].get("get_upper")

    function_answer_csv = []

    dependent_functions = list(
        filter(
            lambda x: x in functions_identified,
            functions_identified[function].get("dependent_functions", []),
        )
    )

    check_other_functions: bool = (
        True if len(dependent_functions) > 0 and dependent_functions[0] != function else False
    )

    dependent_function_indices = None
    dependent_function_get_upper = None

    if check_other_functions:
        dependent_function_indices = functions_identified.get(
            dependent_functions[0]
        ).get("indices")

        dependent_function_get_upper = functions_identified.get(
            dependent_functions[0]
        ).get("get_upper")

        print(
            "DEPENDENT_FUNCTION_INDICES",
            dependent_function_indices,
            "GET_UPPER",
            dependent_function_get_upper,
        )

    stats_json_path = Path(
        f"/home/seigyo/c_repo/c_repo/results/csv_results/stats/{STATE.get('PROJECT_NAME')}_STATS.json"
    )
    stats_json_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_token = {"Input_tokens": 0, "Output_tokens": 0, "Total_tokens": 0}
    empty_stats = {
        "Tokens": dummy_token,
        "Iterations": 0,
        "Random_tool_calls": 0,
        "Other_tool_errors": 0,
        "Incorrect_details": [],
    }

    def write_json_file(data):
        with open(stats_json_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def clean_path_str(path_list):
        block_regex = r"\[([^\[\]]*)\]"
        return "->".join(map(lambda x: re.sub(block_regex, "", x), path_list))

    def zero_token_count():
        return TokenCount(Input_tokens=0, Output_tokens=0, Total_tokens=0)

    def append_path_tokens(path_index, tokens_dict):
        FUNCTION_DICT[function]["Each_Path_Tokens"].append({path_index: tokens_dict})

    def update_function_token_totals():
        FUNCTION_DICT[function].update(
            {
                "Total_Tokens": FUNCTION_INPUT_TOKEN + FUNCTION_OUTPUT_TOKEN,
                "Total_Output": FUNCTION_OUTPUT_TOKEN,
                "Total_Input": FUNCTION_INPUT_TOKEN,
            }
        )

    def parse_output_values(output_string: str):
        values_found: list[int | str | Literal["UNRESOLVED"]] = []

        if not output_string:
            return values_found

        splitted = output_string.split(",")

        for elements in splitted:
            try:
                value_part = elements.split(":")[1].strip('"')

                if '"' in elements.split(":")[1]:
                    values_found.append(value_part)
                else:
                    file_num = int(value_part)
                    values_found.append(file_num)

            except Exception:
                try:
                    values_found.append(elements.split(":")[1].strip('"'))
                except Exception:
                    values_found.append("UNRESOLVED")

        return values_found

    def extract_line_and_file_from_path(raw_path: list[str], target_function: str):
        """
        Extract file and line from path nodes like:

            [Dio860d.c]main[963:1035]
            [Dio860d.c:1024]Dio860dGsim1[657:958]
            [Dio860d.c:773]Dio860dTcsSend[514:573]
            [548]mpf_mfs_addque

        For final nodes like [548]mpf_mfs_addque, there is no file in that node,
        so it reuses the most recent file seen earlier in the path.
        """

        last_file = main_file_name
        found_file = None
        found_line = None

        for node in raw_path:
            if not node:
                continue

            # Case:
            #   [Dio860d.c:773]Dio860dTcsSend[514:573]
            m = re.match(
                r"^\[(?P<file>[^:\]\[]+\.[A-Za-z0-9_]+):(?P<line>\d+)\](?P<func>[A-Za-z_][A-Za-z0-9_]*)",
                node,
            )
            if m:
                last_file = m.group("file")
                if m.group("func") == target_function:
                    found_file = last_file
                    found_line = int(m.group("line"))
                continue

            # Case:
            #   [Dio860d.c]main[963:1035]
            m = re.match(
                r"^\[(?P<file>[^\]\[]+\.[A-Za-z0-9_]+)\](?P<func>[A-Za-z_][A-Za-z0-9_]*)",
                node,
            )
            if m:
                last_file = m.group("file")
                if m.group("func") == target_function:
                    found_file = last_file

                    # Try to recover start line from trailing [963:1035]
                    range_match = re.search(r"\[(?P<start>\d+):(?P<end>\d+)\]\s*$", node)
                    if range_match:
                        found_line = int(range_match.group("start"))
                continue

            # Case:
            #   [548]mpf_mfs_addque
            m = re.match(
                r"^\[(?P<line>\d+)\](?P<func>[A-Za-z_][A-Za-z0-9_]*)",
                node,
            )
            if m:
                if m.group("func") == target_function:
                    found_file = last_file
                    found_line = int(m.group("line"))
                continue

        return found_file, found_line

    def get_function_source_meta(fn: str):
        info = functions_identified.get(fn, {}) or {}

        source_file = (
            info.get("source_file")
            or info.get("file")
            or info.get("file_path")
            or info.get("filename")
            or main_file_name
        )

        line_number = (
            info.get("line_number")
            or info.get("line")
            or info.get("start_line")
        )

        try:
            line_number = int(line_number) if line_number is not None else None
        except Exception:
            line_number = None

        return source_file, line_number

    def extract_meta_from_path_node(node: str):
        """
        Handles:

            [Dio860d.c]main[963:1035]
            [Dio860d.c:1024]Dio860dGsim1[657:958]
            [Dio860d.c:773]Dio860dTcsSend[514:573]
            [548]mpf_mfs_addque

        Returns:
            source_file, line_number, function_name
        """

        if not node:
            return None, None, None

        # [Dio860d.c:773]Dio860dTcsSend[514:573]
        m = re.match(
            r"^\[(?P<file>[^:\]\[]+\.[A-Za-z0-9_]+):(?P<line>\d+)\](?P<func>[A-Za-z_][A-Za-z0-9_]*)",
            node,
        )
        if m:
            return m.group("file"), int(m.group("line")), m.group("func")

        # [Dio860d.c]main[963:1035]
        m = re.match(
            r"^\[(?P<file>[^\]\[]+\.[A-Za-z0-9_]+)\](?P<func>[A-Za-z_][A-Za-z0-9_]*)",
            node,
        )
        if m:
            source_file = m.group("file")
            func_name = m.group("func")

            range_match = re.search(r"\[(?P<start>\d+):(?P<end>\d+)\]\s*$", node)
            line_number = int(range_match.group("start")) if range_match else None

            return source_file, line_number, func_name

        # [548]mpf_mfs_addque
        m = re.match(
            r"^\[(?P<line>\d+)\](?P<func>[A-Za-z_][A-Za-z0-9_]*)",
            node,
        )
        if m:
            return None, int(m.group("line")), m.group("func")

        return None, None, None

    def infer_source_meta_from_raw_path(
        *,
        raw_path: list[str],
        target_function: str,
        fallback_source_file: str | None,
        fallback_line_number: int | None,
    ):
        """
        Infer source_file and line_number from raw path before clean_path_str()
        removes bracket metadata.
        """

        last_seen_file = fallback_source_file
        found_file = fallback_source_file
        found_line = fallback_line_number

        for node in raw_path:
            source_file, line_number, func_name = extract_meta_from_path_node(node)

            if source_file:
                last_seen_file = source_file

            if func_name == target_function:
                found_file = source_file or last_seen_file or fallback_source_file

                if line_number is not None:
                    found_line = line_number

        return found_file, found_line

    def make_dpo_spec(
        *,
        target_function: str,
        argument_numbers: list[int],
        path_index: int,
        raw_path: list[str],
        path_str: str,
        intial_context: str,
        get_upper_value: bool,
    ) -> dict:
        source_file, line_number = get_function_source_meta(target_function)

        source_file, line_number = infer_source_meta_from_raw_path(
            raw_path=raw_path,
            target_function=target_function,
            fallback_source_file=source_file,
            fallback_line_number=line_number,
        )

        return {
            "process_name": process_name,
            "project_structure": project_structure,
            "function_name_to_traced": target_function,
            "argument_numbers": argument_numbers,
            "path_index": path_index,
            "path": path_str,
            "raw_path": raw_path,
            "intial_context": intial_context,
            "get_upper": get_upper_value,
            "source_file": source_file,
            "line_number": line_number,
        }

    def dpo_spec_job_key(spec: dict) -> str:
        """
        Must match resolve_all_paths_with_dpo_flat() when no explicit job_key is used.
        """

        source_file = spec.get("source_file")
        src_stem = Path(source_file).stem if source_file else "unknown_src"

        line_number = spec.get("line_number")

        try:
            line_number = int(line_number) if line_number is not None else None
        except Exception:
            line_number = None

        line = line_number if line_number is not None else "unknown_line"

        path_hash = dpo_hash_text(spec["path"], length=8)
        context_hash = dpo_hash_text(spec["intial_context"], length=8)

        return (
            f"{spec['process_name']}::"
            f"{spec['function_name_to_traced']}::"
            f"{src_stem}::"
            f"line_{line}::"
            f"path_{int(spec['path_index']):04d}::"
            f"p_{path_hash}::"
            f"c_{context_hash}"
        )

    def run_flat_dpo_specs(specs_with_local_keys: list[tuple[tuple[str, int], dict]]):
        if not specs_with_local_keys:
            return {}

        specs = [spec for _, spec in specs_with_local_keys]

        local_key_to_job_key = {
            local_key: dpo_spec_job_key(spec)
            for local_key, spec in specs_with_local_keys
        }

        print(
            "[LOCAL DPO FLATTEN COLLECTED]",
            f"function={function}",
            f"jobs={len(specs)}",
            f"local_keys={list(local_key_to_job_key.keys())}",
        )

        results_by_job_key = resolve_all_paths_with_dpo_flat(specs)

        return {
            local_key: results_by_job_key.get(job_key)
            for local_key, job_key in local_key_to_job_key.items()
        }

    FUNCTION_DICT: dict[str, any] = {
        function: {
            "Total_Input": 0,
            "Total_Output": 0,
            "Total_Tokens": 0,
            "Each_Path_Tokens": [],
        }
    }

    if stats_json_path.exists():
        with open(stats_json_path, "r", encoding="utf-8") as f:
            loaded_dict = json.load(f)
            FUNCTION_DICT = {**FUNCTION_DICT, **loaded_dict}

    if function not in FUNCTION_DICT:
        FUNCTION_DICT[function] = {
            "Total_Input": 0,
            "Total_Output": 0,
            "Total_Tokens": 0,
            "Each_Path_Tokens": [],
        }

    FUNCTION_INPUT_TOKEN = FUNCTION_DICT[function].get("Total_Input", 0)
    FUNCTION_OUTPUT_TOKEN = FUNCTION_DICT[function].get("Total_Output", 0)

    if FUNCTION_DICT[function]["Each_Path_Tokens"] == []:
        PATH_TO_START_WITH = 1
    else:
        PATH_TO_START_WITH = (
            int(list(FUNCTION_DICT[function]["Each_Path_Tokens"][-1].keys())[0]) + 1
        )

    print("Need to start with the path", PATH_TO_START_WITH)

    macro_data = possible_paths_data = None

    call_graph_with_paths = orchestrate(
        project_strcuture=project_structure,
        trees=trees,
        required_func=function,
        main_file_name=main_file_name,
        function_pointer_args=function_pointer_args,
        file_functions=file_functions,
        return_whole_tree=check_other_functions,
    )

    if not call_graph_with_paths:
        print(f"{BOLD}{RED}No possible paths for: {GREEN}{function}{RESET}")
        return None

    macro_data, possible_paths_data = call_graph_with_paths

    if not macro_data:
        print("For ", function, " no macro data")

    call_graph_determined_datas = [path[1] for path in possible_paths_data]

    path_nodes: list[list[CallTreeNode]] | None = [
        path[0][1] for path in possible_paths_data if path[0][1] is not None
    ]

    path_strs: list[list[str]] = [path[0][0] for path in possible_paths_data]

    print("TOTAL PATHS FOR THIS FUNCTION: ", len(path_strs))

    make_graph(paths=path_strs)

    print(f"PARSING FOR FUNCTION -> {function}")

    parser = parseFiles(
        project_structure=project_structure,
        paths=path_strs,
        macro_data=macro_data,
        file_name_bytes=FILE_NAME_BYTES,
    )

    contexts = parser.get_parsed_results(get_upper=get_upper)

    # =============================================================================
    # CASE 1: THERE IS A DEPENDENT FUNCTION
    # =============================================================================
    if len(path_nodes) > 0:
        paths_to_dependent: list[list[str]] = []

        print_or_return_possible_paths_trees(
            paths=path_nodes,
            dependent_function=dependent_functions[0],
            result_path_list=paths_to_dependent,
        )

        print(f"PARSING FOR DEPENDENT FUNCTION -> {dependent_functions[0]}")

        dependent_function_parser = parseFiles(
            project_structure=project_structure,
            paths=paths_to_dependent,
            macro_data=macro_data,
            file_name_bytes=FILE_NAME_BYTES,
        )

        dependent_contexts = dependent_function_parser.get_parsed_results(
            get_upper=True
        )

        needs_dynamic_type = (
            STATE.get("FUNCTION_TYPES").get(function, {}).get("type", "NO DATA")
            == "WRITEF/READF"
        )

        dpo_specs_with_keys: list[tuple[tuple[str, int], dict]] = []

        if needs_dynamic_type:
            for index, (path, context) in enumerate(contexts, start=1):
                if index < PATH_TO_START_WITH:
                    continue

                path_str = clean_path_str(path)

                dpo_specs_with_keys.append(
                    (
                        ("type", index),
                        make_dpo_spec(
                            target_function=function,
                            argument_numbers=list_indices,
                            path_index=index,
                            raw_path=path,
                            path_str=path_str,
                            intial_context=context,
                            get_upper_value=get_upper,
                        )
                    )
                )

        for index, (new_path, context_new) in enumerate(dependent_contexts, start=1):
            if index < PATH_TO_START_WITH:
                continue

            new_path_str = clean_path_str(new_path)

            dpo_specs_with_keys.append(
                (
                    ("dependent", index),
                    make_dpo_spec(
                        target_function=dependent_functions[0],
                        argument_numbers=dependent_function_indices,
                        path_index=index,
                        raw_path=new_path,
                        path_str=new_path_str,
                        intial_context=context_new,
                        get_upper_value=dependent_function_get_upper,
                    )
                )
            )

        selected_flat = run_flat_dpo_specs(dpo_specs_with_keys)

        for index, (path, context) in enumerate(contexts, start=1):
            if index < PATH_TO_START_WITH:
                continue

            call_graph_data = call_graph_determined_datas[index - 1]

            print(f"{BOLD}{GREEN}PROCESS PATH_{index}{RESET}")
            print("-" * 20, "PATH AND CONTEXT", "-" * 20)

            print(highlight(context, CLexer(), TerminalFormatter()))
            print("-" * 56)

            stats_dict1 = None

            if needs_dynamic_type:
                call_graph_data = {
                    **call_graph_data,
                    "process_name": process_name,
                }

                print(" STEP - 1 DETERMINING THE CALL_TYPE")

                selected_type = selected_flat.get(("type", index))

                if not selected_type:
                    final_combined_data = {
                        **call_graph_data,
                        "target_number": {
                            "path_str": "->".join(path),
                            "ans": ["UNRESOLVED"],
                        },
                        "call_number": -1,
                        "type": "NO DATA",
                    }

                    tokens = zero_token_count()
                    append_path_tokens(index, tokens.model_dump())
                    update_function_token_totals()

                    combined_model = Combined.model_validate(final_combined_data)
                    save_dict_csv(data_dict=combined_model.model_dump(), save=True)
                    write_json_file(data=FUNCTION_DICT)
                    answers[function].append(
                        (combined_model, Stats.model_validate(empty_stats))
                    )

                    print(f"DONE WITH PATH {index}")
                    continue

                validated_model, stats, best_attempt = selected_type

                if not validated_model:
                    final_combined_data = {
                        **call_graph_data,
                        "target_number": {
                            "path_str": "->".join(path),
                            "ans": ["UNRESOLVED"],
                        },
                        "call_number": -1,
                        "type": "NO DATA",
                    }

                    tokens = zero_token_count()
                    append_path_tokens(index, tokens.model_dump())
                    update_function_token_totals()

                    combined_model = Combined.model_validate(final_combined_data)
                    save_dict_csv(data_dict=combined_model.model_dump(), save=True)
                    write_json_file(data=FUNCTION_DICT)
                    answers[function].append(
                        (combined_model, Stats.model_validate(empty_stats))
                    )

                    print(f"DONE WITH PATH {index}")
                    continue

                print(
                    f"[DPO USING SELECTED TYPE] path={index} "
                    f"attempt={best_attempt.attempt_no} "
                    f"score={best_attempt.score} "
                    f"history={best_attempt.history_path}"
                )

                stats_dict1 = stats.model_dump()
                FUNCTION_INPUT_TOKEN += stats_dict1.get("Tokens").get("Input_tokens")
                FUNCTION_OUTPUT_TOKEN += stats_dict1.get("Tokens").get("Output_tokens")

                validated_model_dict = validated_model.model_dump()

                print("VALIDATED MODEL CONVERTED TO DICT")
                console.print(validated_model_dict)

                output_string = validated_model_dict.get("output", "")
                call_number = validated_model_dict.get("call_number") or -1

                call_graph_data = {
                    **call_graph_data,
                    "type": output_string,
                    "call_number": call_number,
                    "process_name": process_name,
                }

            else:
                call_graph_data = {
                    **call_graph_data,
                    "process_name": process_name,
                    "type": STATE.get("FUNCTION_TYPES")
                    .get(function, {})
                    .get("type", "NO DATA"),
                }

            print("NOW RUNNING FOR THE DEPENDENT FUNCTION.")

            new_path, context_new = dependent_contexts[index - 1]

            print("CONTEXT FOR THE DEPENDENT FUNCTION")
            print(highlight(context_new, CLexer(), TerminalFormatter()))

            selected_dependent = selected_flat.get(("dependent", index))

            if not selected_dependent:
                final_combined_data = {
                    **call_graph_data,
                    "target_number": {
                        "path_str": "->".join(path),
                        "ans": ["UNRESOLVED"],
                    },
                }

                final_combined_data["call_number"] = -1

                combined_model = Combined.model_validate(final_combined_data)

                if stats_dict1:
                    append_path_tokens(index, stats_dict1["Tokens"])
                else:
                    tokens = zero_token_count()
                    append_path_tokens(index, tokens.model_dump())

                update_function_token_totals()
                write_json_file(data=FUNCTION_DICT)
                save_dict_csv(data_dict=combined_model.model_dump(), save=True)

                answers[function].append(
                    (combined_model, Stats.model_validate(empty_stats))
                )

                print(f"DONE WITH PATH {index}")
                continue

            validated_model, stats, best_attempt = selected_dependent

            if not validated_model:
                final_combined_data = {
                    **call_graph_data,
                    "target_number": {
                        "path_str": "->".join(path),
                        "ans": ["UNRESOLVED"],
                    },
                }

                final_combined_data["call_number"] = -1

                combined_model = Combined.model_validate(final_combined_data)

                if stats_dict1:
                    append_path_tokens(index, stats_dict1["Tokens"])
                else:
                    tokens = zero_token_count()
                    append_path_tokens(index, tokens.model_dump())

                update_function_token_totals()
                write_json_file(data=FUNCTION_DICT)
                save_dict_csv(data_dict=combined_model.model_dump(), save=True)

                answers[function].append(
                    (combined_model, Stats.model_validate(empty_stats))
                )

                print(f"DONE WITH PATH {index}")
                continue

            print(
                f"[DPO USING SELECTED DEPENDENT] path={index} "
                f"attempt={best_attempt.attempt_no} "
                f"score={best_attempt.score} "
                f"history={best_attempt.history_path}"
            )

            validated_model_dict = validated_model.model_dump()
            stats_dict2 = stats.model_dump()

            FUNCTION_INPUT_TOKEN += stats_dict2.get("Tokens").get("Input_tokens")
            FUNCTION_OUTPUT_TOKEN += stats_dict2.get("Tokens").get("Output_tokens")

            if stats_dict1:
                FUNCTION_DICT[function]["Each_Path_Tokens"].append(
                    {
                        index: {
                            key: (
                                stats_dict1["Tokens"][key] + stats_dict2["Tokens"][key]
                            )
                            for key in stats_dict2["Tokens"]
                        }
                    }
                )
            else:
                append_path_tokens(index, stats_dict2["Tokens"])

            update_function_token_totals()
            write_json_file(data=FUNCTION_DICT)

            print("VALIDATED MODEL CONVERTED TO DICT")
            console.print(validated_model_dict)

            output_string = validated_model_dict.get("output", "")
            call_number_dependent = validated_model_dict.get("call_number") or -1

            values_found = []

            for elements in output_string.split(","):
                try:
                    file_num = int(elements.split(":")[1])
                    values_found.append(file_num)
                except Exception:
                    file_num = "UNRESOLVED"
                    values_found.append(file_num)

            final_combined_data = {
                **call_graph_data,
                "target_number": {"path_str": "->".join(path), "ans": values_found},
            }

            if "call_number" not in final_combined_data:
                final_combined_data = {
                    **final_combined_data,
                    "call_number": call_number_dependent,
                }

            console.print(final_combined_data)

            combined_model = Combined.model_validate(final_combined_data)

            save_dict_csv(data_dict=combined_model.model_dump(), save=True)
            answers[function].append((combined_model, stats))

            print(f"DONE WITH PATH {index}")

    # =============================================================================
    # CASE 2: NO DEPENDENT FUNCTION
    # =============================================================================
    else:
        dpo_specs_with_keys: list[tuple[tuple[str, int], dict]] = []

        for index, (path, context) in enumerate(contexts, start=1):
            if index < PATH_TO_START_WITH:
                continue

            if "pmf" in function and len(list_indices) == 0:
                continue

            path_str = clean_path_str(path)

            dpo_specs_with_keys.append(
                (
                    ("direct", index),
                    make_dpo_spec(
                        target_function=function,
                        argument_numbers=list_indices,
                        path_index=index,
                        raw_path=path,
                        path_str=path_str,
                        intial_context=context,
                        get_upper_value=get_upper,
                    )
                )
            )

        selected_flat = run_flat_dpo_specs(dpo_specs_with_keys)

        for index, (path, context) in enumerate(contexts, start=1):
            if index < PATH_TO_START_WITH:
                continue

            call_graph_data = call_graph_determined_datas[index - 1]

            type_of_func = STATE.get("FUNCTION_TYPES").get(function).get("type")

            call_graph_data = {
                **call_graph_data,
                "process_name": process_name,
                "type": type_of_func if type_of_func else "NO DATA",
            }

            console.print(call_graph_data)

            print(f"{BOLD}{GREEN}PROCESS PATH_{index}{RESET}")
            print("-" * 20, "PATH AND CONTEXT", "-" * 20)

            print(highlight(context, CLexer(), TerminalFormatter()))
            print("-" * 56)

            if "pmf" in function and len(list_indices) == 0:
                launch = STATE.get("FUNCTION_TYPES").get(function).get("launch")

                final_combined_data = {
                    **call_graph_data,
                    "target_number": {
                        "path_str": "->".join(path),
                        "ans": ["NO TARGET"],
                    },
                    "call_number": -1,
                }

                if launch:
                    final_combined_data = {**final_combined_data, "launch_via": launch}

                combined_model = Combined.model_validate(final_combined_data)

                tokens = zero_token_count()

                append_path_tokens(index, tokens.model_dump())
                update_function_token_totals()

                save_dict_csv(data_dict=combined_model.model_dump(), save=True)
                write_json_file(data=FUNCTION_DICT)

                console.print(final_combined_data)

                answers[function].append(
                    (combined_model, Stats.model_validate(empty_stats))
                )

                print(f"DONE WITH PATH {index}")
                continue

            selected = selected_flat.get(("direct", index))

            if not selected:
                launch = STATE.get("FUNCTION_TYPES").get(function).get("launch")

                final_combined_data = {
                    **call_graph_data,
                    "target_number": {
                        "path_str": "->".join(path),
                        "ans": ["UNRESOLVED"],
                    },
                    "call_number": -1,
                }

                if "pmf" in function and launch is not None:
                    final_combined_data = {**final_combined_data, "launch_via": launch}

                combined_model = Combined.model_validate(final_combined_data)
                tokens = zero_token_count()

                append_path_tokens(index, tokens.model_dump())
                update_function_token_totals()

                write_json_file(data=FUNCTION_DICT)
                save_dict_csv(data_dict=combined_model.model_dump(), save=True)

                answers[function].append(
                    (combined_model, Stats.model_validate(empty_stats).model_dump())
                )

                print(f"DONE WITH PATH {index}")
                continue

            validated_model, stats, best_attempt = selected

            if not validated_model:
                launch = STATE.get("FUNCTION_TYPES").get(function).get("launch")

                final_combined_data = {
                    **call_graph_data,
                    "target_number": {
                        "path_str": "->".join(path),
                        "ans": ["UNRESOLVED"],
                    },
                    "call_number": -1,
                }

                if "pmf" in function and launch is not None:
                    final_combined_data = {**final_combined_data, "launch_via": launch}

                combined_model = Combined.model_validate(final_combined_data)
                tokens = zero_token_count()

                append_path_tokens(index, tokens.model_dump())
                update_function_token_totals()

                write_json_file(data=FUNCTION_DICT)
                save_dict_csv(data_dict=combined_model.model_dump(), save=True)

                answers[function].append(
                    (combined_model, Stats.model_validate(empty_stats).model_dump())
                )

                print(f"DONE WITH PATH {index}")
                continue

            print(
                f"[DPO USING SELECTED DIRECT] path={index} "
                f"attempt={best_attempt.attempt_no} "
                f"score={best_attempt.score} "
                f"history={best_attempt.history_path}"
            )

            validated_model_dict = validated_model.model_dump()

            print("VALIDATED MODEL CONVERTED TO DICT")
            console.print(validated_model_dict)

            stats_dict = stats.model_dump()

            FUNCTION_INPUT_TOKEN += stats_dict.get("Tokens").get("Input_tokens")
            FUNCTION_OUTPUT_TOKEN += stats_dict.get("Tokens").get("Output_tokens")

            append_path_tokens(index, stats_dict["Tokens"])
            update_function_token_totals()

            write_json_file(data=FUNCTION_DICT)

            output_string = validated_model_dict.get("output", "")

            values_found = parse_output_values(output_string)

            call_number = validated_model_dict.get("call_number") or -1

            launch = STATE.get("FUNCTION_TYPES").get(function).get("launch")

            final_combined_data = {
                **call_graph_data,
                "target_number": {
                    "path_str": "->".join(path),
                    "ans": values_found if len(values_found) > 0 else ["NO TARGET"],
                },
                "call_number": call_number,
            }

            if "pmf" in function and launch:
                final_combined_data = {**final_combined_data, "launch_via": launch}

            console.print(final_combined_data)

            combined_model = Combined.model_validate(final_combined_data)

            save_dict_csv(data_dict=combined_model.model_dump(), save=True)
            answers[function].append((combined_model, stats))

            print(f"DONE WITH PATH {index}")

    return function_answer_csv

@time_it()
def trace_variable(project_path):
    """
    Global-flat DPO version.

    Pipeline:

        1. Build project structure, trees, macros, FILE_FUNCTIONS.
        2. Identify all functions to trace.
        3. Collect DPO specs from every function without running DPO.
        4. Run resolve_all_paths_with_dpo_flat(all_specs) exactly once.
        5. Process every function using the global DPO results.

    Required make_llm_calls_for_function contract:

        Collect mode:

            prepared_data = make_llm_calls_for_function(
                ...,
                dpo_mode="collect",
            )

            It must return either None or dict like:

            {
                "function": function_name,
                "dpo_specs": list[dict],
                "prepared": any,
            }

        Process mode:

            function_dataframes = make_llm_calls_for_function(
                ...,
                dpo_mode="process",
                prepared_data=prepared_data,
                global_dpo_results=global_dpo_results,
            )

    """

    import pickle
    import inspect
    from pathlib import Path
    from collections import defaultdict

    STATE = State()

    pickle_dir = (
        Path(__file__).resolve().parent / "pickle_data/project_structures_pickle"
    )

    if not pickle_dir.exists():
        pickle_dir.mkdir(exist_ok=True, parents=True)

    project_structure_path = pickle_dir / f"{STATE.get('PROJECT_NAME')}.pkl"

    potential_main_files: list[str] | None = None

    # ==========================================================================
    # 1. LOAD OR BUILD PROJECT STRUCTURE
    # ==========================================================================
    if not project_structure_path.exists():
        print(
            "PROJECT STRUCTURE NEEDS TO BE RESOLVED. "
            "NO PICKLE FILE. IT WILL TAKE TIME...."
        )

        PROJECT_STRUCTURE, potential_main_files = return_project_mapping(
            show=False,
            project_path=project_path,
        )

        PROJECT_STRUCTURE = dict(
            sorted(PROJECT_STRUCTURE.items(), key=lambda x: str(x[0]))
        )

        STATE.set("PROJECT_STRUCTURE", PROJECT_STRUCTURE)

        with open(project_structure_path, "wb") as f:
            pickle.dump((PROJECT_STRUCTURE, potential_main_files), f)

    else:
        with open(project_structure_path, "rb") as f:
            PROJECT_STRUCTURE, potential_main_files = pickle.load(f)

        STATE.set("PROJECT_STRUCTURE", PROJECT_STRUCTURE)

    print("THE MAIN FILES ARE: ", potential_main_files)

    # ==========================================================================
    # 2. PREPROCESS PROJECT TREES
    # ==========================================================================
    trees = Preprocess().preprocess(
        project_structure=PROJECT_STRUCTURE,
    )

    STATE.set("TREES", trees)

    PROJECT_STRUCTURE = {
        key: str(PROJECT_STRUCTURE[key])
        for key in PROJECT_STRUCTURE.keys()
    }

    print("LENGTH OF PROJECT_STRUCTURE", len(PROJECT_STRUCTURE))

    FUNCTION_POINTER_ARGS = STATE.get("FUNCTION_POINTER_ARGS")

    FILE_FUNCTIONS = {}

    main_file_name = None
    bad_main_files = []
    macros = {}
    file_includes: dict[str, list] = {}

    # ==========================================================================
    # 3. EXTRACT MACROS, INCLUDES, FUNCTIONS
    # ==========================================================================
    for files in PROJECT_STRUCTURE.keys():
        macros[files] = extract_all_macros(PROJECT_STRUCTURE[files])
        file_includes[files] = extract_includes(PROJECT_STRUCTURE[files])

        if files.endswith(".h"):
            continue

        functions = get_local_function_definitions(
            code_bytes=trees[files][1],
        )

        if "main" in functions.keys() and any(
            files == x for x in potential_main_files
        ):
            main_file_name = files
            print("Found main file", main_file_name)

        if "main" in functions and not any(files == x for x in potential_main_files):
            bad_main_files.append(files)

        FILE_FUNCTIONS[files] = functions

    for bad_files in bad_main_files:
        del FILE_FUNCTIONS[bad_files]
        del PROJECT_STRUCTURE[bad_files]
        del trees[bad_files]

    STATE.set("FILE_FUNCTIONS", FILE_FUNCTIONS)
    STATE.set("FILE_INCLUDES", file_includes)
    STATE.set("MACROS", macros)

    # ==========================================================================
    # 4. IDENTIFY FUNCTIONS TO TRACE
    # ==========================================================================
    functions_identified = identify_funs_to_trace(
        project_structure=PROJECT_STRUCTURE,
        trees=trees,
    )

    if functions_identified == {}:
        print(
            f"{BOLD}{RED}NO FUNCTIONS IDENTIFIED IN THE PROJECT "
            f"{project_path.name}.{RESET}"
        )
        return None

    console.print(
        "-" * 10,
        "DETECTED FUNCTIONS NEEDS TO BE TRACED AND THEIR ARG. NUMS.",
        "-" * 10,
    )

    console.print(functions_identified)
    console.print("-" * 60)

    answers: dict[str, list[tuple[BaseModel, BaseModel]]] = defaultdict(list)

    print("This is the main file::", main_file_name)

    # ==========================================================================
    # 5. VERIFY make_llm_calls_for_function SUPPORTS GLOBAL MODE
    # ==========================================================================
    make_llm_signature = inspect.signature(make_llm_calls_for_function)

    required_extra_params = {
        "dpo_mode",
        "prepared_data",
        "global_dpo_results",
    }

    missing_params = [
        param
        for param in required_extra_params
        if param not in make_llm_signature.parameters
    ]

    if missing_params:
        raise TypeError(
            "Your make_llm_calls_for_function does not support global flat DPO yet. "
            "trace_variable alone cannot flatten globally because your current "
            "make_llm_calls_for_function starts DPO internally.\n\n"
            f"Missing parameters: {missing_params}\n\n"
            "Update make_llm_calls_for_function to accept:\n"
            "    dpo_mode='normal' | 'collect' | 'process'\n"
            "    prepared_data=None\n"
            "    global_dpo_results=None\n"
        )

    # ==========================================================================
    # 6. COLLECT PHASE
    # ==========================================================================
    prepared_by_function: dict[str, dict] = {}
    all_dpo_specs: list[dict] = []

    print()
    print("=" * 80)
    print("[GLOBAL DPO COLLECT PHASE START]")
    print("=" * 80)

    for function in functions_identified:
        STATE.set("CURRENT_PROCESSED_FUNCTION", function)

        print()
        print(f"{BOLD}{GREEN}[COLLECT] {function}{RESET}")

        prepared_data = make_llm_calls_for_function(
            function=function,
            trees=trees,
            functions_identified=functions_identified,
            answers=answers,
            main_file_name=main_file_name,
            function_pointer_args=FUNCTION_POINTER_ARGS,
            file_functions=FILE_FUNCTIONS,
            project_structure=PROJECT_STRUCTURE,
            project_path=project_path,
            dpo_mode="collect",
            prepared_data=None,
            global_dpo_results=None,
        )

        if prepared_data is None:
            print(f"[COLLECT SKIP] {function}: no prepared data")
            continue

        if not isinstance(prepared_data, dict):
            raise TypeError(
                f"Collect mode for {function} must return dict or None. "
                f"Got: {type(prepared_data)}"
            )

        function_dpo_specs = prepared_data.get("dpo_specs", [])

        if function_dpo_specs is None:
            function_dpo_specs = []

        if not isinstance(function_dpo_specs, list):
            raise TypeError(
                f"prepared_data['dpo_specs'] for {function} must be list. "
                f"Got: {type(function_dpo_specs)}"
            )

        prepared_by_function[function] = prepared_data
        all_dpo_specs.extend(function_dpo_specs)

        print(
            f"[COLLECT DONE] function={function} "
            f"specs={len(function_dpo_specs)} "
            f"total_specs_so_far={len(all_dpo_specs)}"
        )

    print()
    print("=" * 80)
    print("[GLOBAL DPO COLLECT PHASE DONE]")
    print(f"functions_prepared={len(prepared_by_function)}")
    print(f"total_dpo_jobs={len(all_dpo_specs)}")
    print("=" * 80)

    # ==========================================================================
    # 7. GLOBAL DPO PHASE
    # ==========================================================================
    if all_dpo_specs:
        print()
        print("=" * 80)
        print("[GLOBAL DPO START]")
        print(f"total_jobs={len(all_dpo_specs)}")
        print("=" * 80)

        global_dpo_results = resolve_all_paths_with_dpo_flat(all_dpo_specs)

        print()
        print("=" * 80)
        print("[GLOBAL DPO DONE]")
        print(f"results={len(global_dpo_results)}")
        print("=" * 80)

    else:
        print()
        print("=" * 80)
        print("[GLOBAL DPO SKIPPED]")
        print("No DPO specs collected.")
        print("=" * 80)

        global_dpo_results = {}

    # ==========================================================================
    # 8. PROCESS PHASE
    # ==========================================================================
    data_csvs = []

    print()
    print("=" * 80)
    print("[GLOBAL DPO PROCESS PHASE START]")
    print("=" * 80)

    for function in functions_identified:
        if function not in prepared_by_function:
            print(f"[PROCESS SKIP] {function}: no prepared data")
            continue

        STATE.set("CURRENT_PROCESSED_FUNCTION", function)

        print()
        print(f"{BOLD}{GREEN}[PROCESS] {function}{RESET}")

        function_dataframes: list | None = make_llm_calls_for_function(
            function=function,
            trees=trees,
            functions_identified=functions_identified,
            answers=answers,
            main_file_name=main_file_name,
            function_pointer_args=FUNCTION_POINTER_ARGS,
            file_functions=FILE_FUNCTIONS,
            project_structure=PROJECT_STRUCTURE,
            project_path=project_path,
            dpo_mode="process",
            prepared_data=prepared_by_function[function],
            global_dpo_results=global_dpo_results,
        )

        if function_dataframes is not None:
            data_csvs = [*data_csvs, *function_dataframes]

    print()
    print("=" * 80)
    print("[GLOBAL DPO PROCESS PHASE DONE]")
    print(f"data_csvs={len(data_csvs)}")
    print("=" * 80)

    console.print(data_csvs)

    return answers

# Run the tracer
if __name__ == "__main__":
    # c_folder_name = input("Enter c_folder name (in /src:)::")
    # region LOGGING
    import sys
    from datetime import datetime

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)

        def flush(self):
            for s in self.streams:
                s.flush()

    # endregion

    summaries = []
    # parent_folder = 'results/detailed_tool_description_changed_prompt'
    parent_folder = "logs/after_tool_modification"
    # functions_to_trace = functions_to_trace()
    from pathlib import Path

    folder = Path(__file__).resolve().parent / f"{parent_folder}"
    folder.mkdir(parents=True, exist_ok=True)
    apl_path = "/home/seigyo/c_repo/c_repo/src/src_analysis/src"
    rbt_path = "/home/seigyo/c_repo/c_repo/src/src_rbt/src"
    src_wh = "/home/seigyo/c_repo/c_repo/src/src_wh/wh-dio/src"
    list_project_paths = [apl_path, rbt_path, src_wh]
    to_exclude = {
        # 'wh-dio':['libdio','libdio_ora','libDioKyusei','libDioTrace','libLocal','tools','dio000d','dio100d','dio110d','dio110d_nobori','dio120d','dio130d','dio140d','dio150d','dio160d','dio170d','dio175d','dio210d','dio210d_nobori'],
        "wh-dio": [
            "libdio",
            "libdio_ora",
            "libDioKyusei",
            "libDioTrace",
            "libLocal",
            "tools",
        ],
        "src_analysis": ["libapl"],
        "src_rbt": ["libRbt"],
    }

    # WHOLE,whole_index = pick(options=['True','False'],title=f'Want to process all projects one by one (Like apl projects, rbt projects)?\n True = YES \n False = NO\n\n',indicator='==>>',default_index=0)
    # WHOLE = True if whole_index==0 else False
    WHOLE = True
    if not WHOLE:
        parent_project, parent_index = pick(
            options=[Path(path).resolve().parent.name for path in list_project_paths],
            title="Pick the parent project",
            indicator="==>>",
            default_index=0,
        )
        projects = [
            p.name
            for p in Path(list_project_paths[parent_index]).iterdir()
            if p.is_dir()
        ]
        projects = list(filter(lambda x: x not in to_exclude[parent_project], projects))

        project, _ = pick(
            options=projects,
            title="Pick the projects",
            indicator="==>>",
            default_index=0,
        )
        project_path = Path(list_project_paths[parent_index]) / project

        from tools.tools import set_tool_def

        set_tool_def()
        STATE = load_files()
        parent_folder = "/home/seigyo/c_repo/c_repo/logs"
        datetime_for_name = f"{datetime.now():%Y%m%d_%H%M%S}"
        STATE.set("TIME", datetime_for_name)
        logfile = f"{parent_folder}/{project}_{datetime_for_name}.txt"
        log = open(logfile, "w", buffering=1)
        sys.stdout = Tee(sys.__stdout__, log)
        sys.stderr = Tee(sys.__stderr__, log)
        # STATE.set('PROJECT_NAME','hehe')
        STATE.set("PROJECT_NAME", project)

        summary = trace_variable(project_path=project_path)  # list[ocombined,

        console.print(summary)

    else:
        # parent_project,project_index = pick(options=[Path(path).resolve().parent.name for path in list_project_paths],title='Pick the parent project to; process all its child projects.',indicator='==>>',default_index=0)
        parent_project, project_index = "wh-dio", 2
        projects = [
            p.name
            for p in Path(list_project_paths[project_index]).iterdir()
            if p.is_dir()
        ]
        # already_done = ["dio000d", "dio800d", "dio810d", "dio815d", "dio860d","dio"

        # TODO: Run it back
        # not_done = ['dio000d']
        # projects = list(filter(lambda x: x in not_done, projects))

        projects = [
            'dio000d', 'dio100d', 'dio110d', 'dio110d_nobori', 'dio120d',
            'dio130d', 'dio140d', 'dio150d', 'dio160d', 'dio170d',
            'dio175d', 'dio210d', 'dio210d_nobori', 'dio220d', 'dio260d',
            'dio260d_nobori', 'dio270d', 'dio310d', 'dio410d', 'dio600d',
            'dio690d', 'dio800d', 'dio810d', 'dio815d', 'dio860d'
        ]
        
        # projects = ['dio860d']
        # projects = ['dio860d', 'dio220d','dio260d','dio260d_nobori','dio270d','dio310d','dio410d','dio210d','dio210d_nobori','dio110d','dio110d_nobori']
        # projects = ['dio260d_nobori','dio270d','dio310d','dio410d','dio210d','dio210d_nobori','dio110d','dio110d_nobori','dio000d','dio100d','dio120d',
        #             'dio130d','dio140d','dio150d','dio160d','dio170d','dio175d','dio220d','dio260d','dio600d','dio690d']
        # projects = ['dio800d']
        # projects = ['dio000d','dio110d','dio110d_nobori','dio120d','dio130d','dio']
        # to_exclude = ['libdio','libdio_ora','libDioKyusei','libDioTrace','libLocal','tools']

        # TODO: Run it back
        # projects = list(filter(lambda x: x not in to_exclude[parent_project], projects))
        
        for project in reversed(projects):
            print("RUNNING FOR PROJECT", project)
            set_tool_def()
            STATE = load_files()
            console.print(STATE.__dict__.keys())
            # project_path = Path(apl_path)/project if 'apl' in project else Path(rbt_path)/project if 'rbt' in project else Path(src_wh)/project
            project_path = Path(list_project_paths[project_index]) / project
            datetime_for_name = f"{datetime.now():%Y%m%d_%H%M%S}"
            STATE.set("TIME", datetime_for_name)
            # STATE.set('TIME','20260306_093155')
            # logfile = f"{parent_folder}/{project}_{datetime_for_name}.txt"
            # os.environ['PROJECT_NAME'] = project
            STATE.set("PROJECT_NAME", project)
            # STATE.set('PROJECT_NAME','hehe')

            summary = trace_variable(project_path=project_path)  # list[ocombined,]
            # destroy the state...
            STATE.reset()
            console.print(STATE.__dict__.keys())
