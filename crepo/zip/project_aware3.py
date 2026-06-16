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

from call_graph.call_graph import orchestrate
from call_graph.data_classes import CallTreeNode, custom_tree
from call_graph.gen_graph import make_graph
from client.llm import OllamaClient
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

import hashlib
import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- DPO CONFIGURATION ---
MAX_DPO_ATTEMPTS = 5        # Hardcoded as requested
MAX_CONCURRENT_AGENTS = 100  # Adjust based on your API rate limits
csv_lock = threading.Lock() # Ensures thread-safe CSV writing

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


def extract_function_calls(code: bytes) -> list[str]:
    """
    RETURNS A LIST OF FUNCTION CALLED IN THIS CODE.
    """
    # import from tree_sitter_customustompp as tsc
    import tree_sitter_custom as tsc
    from tree_sitter import Language, Parser

    lang = Language(tsc.language())
    parser = Parser(lang)
    tree = parser.parse(code)

    calls: list[str] = []

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


# region HELPERS (Identifying funs to trace and printing trees, dfs.)
def identify_funs_to_trace(
    project_structure: dict[str, str],
    trees: dict,
    name_of_json: str = "json_data/mpf_data.json",
) -> (
    dict[str, dict[str, any]] | None
):  # will return {function_name, [list of indice of arguments to trace...]}

    # file_path = Path(name_of_json)
    STATE = State()
    functions_to_detect = STATE.get("FUNCTION_TYPES")
    ans = {}  # {function_name, [indices to trace for it....]}
    if not functions_to_detect:
        print(f"Data 'FUNCTION_TYPES' Not in state.")
    for file_name, file_path_str in project_structure.items():
        if file_name.endswith(".h"):
            continue  # as we don't look at the function declared in the header files...

        bytes_content = trees[file_name][1]  # content
        functions_called = extract_function_calls(bytes_content)
        # print('Functions called in ',file_name,functions_called)
        # sys.exit()
        for func in functions_to_detect:

            if func in functions_called:
                list_of_indices = [
                    ind for ind in functions_to_detect[func].get("indices")
                ]
                get_upper = functions_to_detect[func].get("get_upper")
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
) -> tuple[type[BaseModel], dict[str, any]]:
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

    ans, stats, messages = client.start_tool_chain(prompt_data=prompt_data)
    # ans, stats, messages = client.start_new_tool_chain(prompt_data=prompt_data)
    # endregion

    return ans, stats, messages


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

def calculate_dpo_score(stats: Stats, result_model) -> float:
    """Simple heuristic score for DPO collection. Higher is better."""
    score = 100.0
    if hasattr(result_model, 'output'):
        if 'UNRESOLVED' in str(result_model.output):
            score -= 50.0
    score -= stats.Iterations * 5.0
    score -= stats.Random_tool_calls * 20.0
    score -= stats.Other_tool_errors * 10.0
    score -= (stats.Tokens.Total_tokens / 1000.0) * 2.0
    return score

def process_single_path_dpo(job_data):
    """
    Processes a single path with MAX_DPO_ATTEMPTS tries concurrently.
    Saves history and score for each attempt, picks the best, and saves to CSV.
    """
    process_name = job_data['process_name']
    function_name = job_data['function_name']
    context = job_data['context']
    path_str = job_data['path_str']
    call_graph_data = job_data['call_graph_data']
    list_indices = job_data['list_indices']
    get_upper = job_data['get_upper']
    project_structure = job_data['project_structure']
    
    dep_context = job_data.get('dep_context')
    dep_path_str = job_data.get('dep_path_str')
    dep_indices = job_data.get('dep_indices')
    dep_get_upper = job_data.get('dep_get_upper')
    dep_function = job_data.get('dep_function')

    # FIX: Use context hash to ensure same context = same folder (fixes your input token bug)
    context_hash = hashlib.md5(context.encode('utf-8')).hexdigest()[:8]
    base_dir = Path("dpo_llm_data") / process_name / f"{function_name}_{context_hash}"
    base_dir.mkdir(parents=True, exist_ok=True)

    best_score = float('-inf')
    best_attempt_data = None

    # Suppress prints during attempts so you only see the final summary
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        for attempt in range(1, MAX_DPO_ATTEMPTS + 1):
            attempt_dir = base_dir / f"attempt_{attempt:02d}"
            
            # --- CONTINUATION LOGIC ---
            if (attempt_dir / "history.json").exists() and (attempt_dir / "score.json").exists():
                with open(attempt_dir / "score.json", "r") as f:
                    s_data = json.load(f)
                    if s_data.get("score", float('-inf')) > best_score:
                        best_score = s_data["score"]
                        # Note: To fully resume CSV writing, you'd also need to 
                        # reload the best history.json here. Otherwise, it just 
                        # skips the LLM call for finished attempts.
                continue 
            # --------------------------

            attempt_dir.mkdir(exist_ok=True)
            # ... rest of the code ...

            type_of_func = job_data.get('type_of_func', "NO DATA")
            
            ans1, stats1, msgs1 = None, None, []
            ans2, stats2, msgs2 = None, None, []

            try:
                if type_of_func == "WRITEF/READF":
                    res1 = llm_calls(project_structure, function_name, list_indices, context, path_str, get_upper=True)
                    ans1, stats1 = res1[0], res1[1]
                    msgs1 = res1[2] if len(res1) > 2 else [] # Safely unpack
                    if ans1:
                        type_of_func = ans1.model_dump().get("output", "NO DATA")

                if dep_function:
                    res2 = llm_calls(project_structure, dep_function, dep_indices, dep_context, dep_path_str, dep_get_upper)
                    ans2, stats2 = res2[0], res2[1]
                    msgs2 = res2[2] if len(res2) > 2 else [] # Safely unpack
            except Exception as e:
                # Write to a file so you can see errors even if stdout is suppressed
                with open("dpo_errors.log", "a") as err_log:
                    err_log.write(f"Attempt {attempt} for {function_name} crashed: {e}\n")

            final_ans = ans2 if dep_function else ans1
            final_stats = stats2 if dep_function else stats1
            
            all_messages = []
            if msgs1: all_messages.extend([{"role": "system", "content": "MAIN_FUNC_CALL"}, *msgs1])
            if msgs2: all_messages.extend([{"role": "system", "content": "DEP_FUNC_CALL"}, *msgs2])
            
            if not final_stats:
                final_stats = Stats.model_validate({
                    "Iterations": 0, "Random_tool_calls": 0, "Other_tool_errors": 0,
                    "Incorrect_details": [], "Tokens": {"Input_tokens": 0, "Output_tokens": 0, "Total_tokens": 0}
                })

            score = calculate_dpo_score(final_stats, final_ans)

            # Save history.json
            history_data = {
                "job_key": f"{process_name}::{function_name}::{context_hash}",
                "process_name": process_name,
                "path_index": context_hash,
                "function_name_to_traced": function_name,
                "argument_numbers": list_indices,
                "context_hash": context_hash,
                "path_str": path_str,
                "messages": all_messages
            }
            with open(attempt_dir / "history.json", "w") as f:
                json.dump(history_data, f, indent=2)

            # Save score.json
            score_data = {
                "score": score,
                "iterations": final_stats.Iterations,
                "tokens": final_stats.Tokens.Total_tokens,
                "errors": final_stats.Other_tool_errors + final_stats.Random_tool_calls,
                "resolved": 'UNRESOLVED' not in str(final_ans.output) if final_ans else False
            }
            with open(attempt_dir / "score.json", "w") as f:
                json.dump(score_data, f, indent=2)

            if score > best_score:
                best_score = score
                best_attempt_data = {
                    "ans": final_ans, "stats": final_stats, "type": type_of_func,
                    "ans1": ans1, "ans2": ans2
                }

    finally:
        sys.stdout = old_stdout # Restore stdout

    # Process the best attempt
    if best_attempt_data:
        final_ans = best_attempt_data['ans']
        values_found = []
        call_number = -1
        
        if final_ans:
            out_dict = final_ans.model_dump()
            output_string = out_dict.get("output", "")
            call_number = out_dict.get("call_number") or -1
            
            for elements in output_string.split(","):
                try:
                    if ':' in elements:
                        val = elements.split(":")[1].strip('"')
                        values_found.append(int(val) if val.isdigit() else val)
                    else:
                        values_found.append(elements.strip('"'))
                except:
                    values_found.append("UNRESOLVED")
                    
        if not values_found: values_found = ["UNRESOLVED"]

        final_combined_data = {
            **call_graph_data, "process_name": process_name,
            "type": best_attempt_data['type'],
            "target_number": {"path_str": path_str, "ans": values_found},
            "call_number": call_number
        }
        
        launch = job_data.get('launch_via')
        if "pmf" in function_name and launch:
            final_combined_data["launch_via"] = launch

        combined_model = Combined.model_validate(final_combined_data)
        
        # Thread-safe CSV save
        with csv_lock:
            save_dict_csv(data_dict=combined_model.model_dump(), save=True)
        
        print(f"{GREEN}✓ COMPLETED:{RESET} {process_name} | {function_name} | Hash: {context_hash} | Best Score: {best_score:.2f} | Resolved: {values_found}")
        return combined_model, best_attempt_data['stats']
    
    return None, None

@time_it()
def trace_variable(project_path):
    STATE = State()
    pickle_dir = Path(__file__).resolve().parent / "pickle_data/project_structures_pickle"
    if not pickle_dir.exists(): pickle_dir.mkdir(exist_ok=True, parents=True)

    project_structure_path = pickle_dir / f"{STATE.get('PROJECT_NAME')}.pkl"
    potential_main_files = None
    if not project_structure_path.exists():
        print("PROJECT STRUCTURE NEEDS TO BE RESOLVED. NO PICKLE FILE. IT WILL TAKE TIME....")
        PROJECT_STRUCTURE, potential_main_files = return_project_mapping(show=False, project_path=project_path)
        PROJECT_STRUCTURE = dict(sorted(PROJECT_STRUCTURE.items(), key=lambda x: str(x[0])))
        STATE.set("PROJECT_STRUCTURE", PROJECT_STRUCTURE)
        with open(project_structure_path, "wb") as f: pickle.dump((PROJECT_STRUCTURE, potential_main_files), f)
    else:
        with open(project_structure_path, "rb") as f: PROJECT_STRUCTURE, potential_main_files = pickle.load(f)
        STATE.set("PROJECT_STRUCTURE", PROJECT_STRUCTURE)

    print("THE MAIN FILES ARE: ", potential_main_files)
    trees = Preprocess().preprocess(project_structure=PROJECT_STRUCTURE)
    STATE.set("TREES", trees)
    
    PROJECT_STRUCTURE = {key: str(PROJECT_STRUCTURE[key]) for key in PROJECT_STRUCTURE.keys()}
    FUNCTION_POINTER_ARGS = STATE.get("FUNCTION_POINTER_ARGS")
    FILE_FUNCTIONS = {}
    main_file_name = None
    bad_main_files = []
    macros = {}
    file_includes = {}

    for files in PROJECT_STRUCTURE.keys():
        macros[files] = extract_all_macros(PROJECT_STRUCTURE[files])
        file_includes[files] = extract_includes(PROJECT_STRUCTURE[files])
        if files.endswith(".h"): continue
        file_path = PROJECT_STRUCTURE[files]
        functions = get_local_function_definitions(code_bytes=trees[files][1])
        if "main" in functions.keys() and any(files == x for x in potential_main_files):
            main_file_name = files
            print("Found main file", main_file_name)
        if "main" in functions and not any(files == x for x in potential_main_files):
            bad_main_files.append(files)
        FILE_FUNCTIONS[files] = functions

    for bad_files in bad_main_files:
        del FILE_FUNCTIONS[bad_files]; del PROJECT_STRUCTURE[bad_files]; del trees[bad_files]
        
    STATE.set("FILE_FUNCTIONS", FILE_FUNCTIONS)
    STATE.set("FILE_INCLUDES", file_includes)
    STATE.set("MACROS", macros)

    functions_identified = identify_funs_to_trace(project_structure=PROJECT_STRUCTURE, trees=trees)
    if functions_identified == {}:
        print(f"{BOLD}{RED}NO FUNCTIONS IDENTIFIED IN THE PROJECT {project_path.name}.{RESET}")
        return None

    console.print("-" * 10, "DETECTED FUNCTIONS NEEDS TO BE TRACED", "-" * 10)
    console.print(list(functions_identified.keys()))

    # --- FLATTEN ALL PATHS INTO JOBS ---
    all_jobs = []
    FILE_NAME_BYTES = {key: value[1] for key, value in STATE.get("TREES").items()}

    for function in functions_identified:
        STATE.set("CURRENT_PROCESSED_FUNCTION", function)
        list_indices = functions_identified[function].get("indices")
        get_upper = functions_identified[function].get("get_upper")
        type_of_func = STATE.get("FUNCTION_TYPES").get(function, {}).get("type", "NO DATA")
        launch = STATE.get("FUNCTION_TYPES").get(function, {}).get("launch")
        
        dependent_functions = list(filter(lambda x: x in functions_identified, functions_identified[function].get("dependent_functions", [])))
        check_other_functions = True if dependent_functions and dependent_functions[0] != function else False
        dep_function = dependent_functions[0] if check_other_functions else None
        dep_indices = functions_identified.get(dep_function, {}).get("indices") if dep_function else None
        dep_get_upper = functions_identified.get(dep_function, {}).get("get_upper") if dep_function else None

        call_graph_with_paths = orchestrate(
            project_strcuture=PROJECT_STRUCTURE, trees=trees, required_func=function,
            main_file_name=main_file_name, function_pointer_args=FUNCTION_POINTER_ARGS,
            file_functions=FILE_FUNCTIONS, return_whole_tree=check_other_functions
        )
        if not call_graph_with_paths: continue
            
        macro_data, possible_paths_data = call_graph_with_paths
        path_nodes = [path[0][1] for path in possible_paths_data if path[0][1] is not None]
        path_strs = [path[0][0] for path in possible_paths_data]
        call_graph_determined_datas = [path[1] for path in possible_paths_data]

        parser = parseFiles(project_structure=PROJECT_STRUCTURE, paths=path_strs, macro_data=macro_data, file_name_bytes=FILE_NAME_BYTES)
        contexts = parser.get_parsed_results(get_upper=get_upper)

        dep_contexts = []
        if len(path_nodes) > 0 and dep_function:
            paths_to_dependent = []
            print_or_return_possible_paths_trees(paths=path_nodes, dependent_function=dep_function, result_path_list=paths_to_dependent)
            dep_parser = parseFiles(project_structure=PROJECT_STRUCTURE, paths=paths_to_dependent, macro_data=macro_data, file_name_bytes=FILE_NAME_BYTES)
            dep_contexts = dep_parser.get_parsed_results(get_upper=True)

        for index, (path, context) in enumerate(contexts, start=1):
            block_regex = r"\[([^\[\]]*)\]"
            path_str = "->".join(map(lambda x: re.sub(block_regex, "", x), path))
            call_graph_data = call_graph_determined_datas[index - 1]
            
            job = {
                'process_name': STATE.get('PROJECT_NAME'), 'function_name': function,
                'context': context, 'path_str': path_str, 'call_graph_data': call_graph_data,
                'list_indices': list_indices, 'get_upper': get_upper, 'project_structure': PROJECT_STRUCTURE,
                'type_of_func': type_of_func, 'launch_via': launch, 'dep_function': dep_function,
                'dep_indices': dep_indices, 'dep_get_upper': dep_get_upper
            }
            if dep_contexts and index - 1 < len(dep_contexts):
                dep_path, dep_context = dep_contexts[index - 1]
                job['dep_context'] = dep_context
                job['dep_path_str'] = "->".join(map(lambda x: re.sub(block_regex, "", x), dep_path))
            all_jobs.append(job)

    print(f"{BOLD}{ORANGE}Total paths flattened for concurrent DPO collection: {len(all_jobs)}{RESET}")

    # --- EXECUTE CONCURRENTLY ---
    answers = defaultdict(list)
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_AGENTS) as executor:
        futures = {executor.submit(process_single_path_dpo, job): job for job in all_jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                combined_model, stats = future.result()
                if combined_model:
                    answers[job['function_name']].append((combined_model, stats))
            except Exception as e:
                print(f"{RED}Job failed for {job['function_name']}: {e}{RESET}")

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

        projects = ['dio000d']

        # projects = [
        #     'dio000d', 'dio100d', 'dio110d', 'dio110d_nobori', 'dio120d',
        #     'dio130d', 'dio140d', 'dio150d', 'dio160d', 'dio170d',
        #     'dio175d', 'dio210d', 'dio210d_nobori', 'dio220d', 'dio260d',
        #     'dio260d_nobori', 'dio270d', 'dio310d', 'dio410d', 'dio600d',
        #     'dio690d', 'dio800d', 'dio810d', 'dio815d', 'dio860d'
        # ]
        
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
