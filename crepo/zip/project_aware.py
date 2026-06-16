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

    ans, stats = client.start_tool_chain(prompt_data=prompt_data)
    # ans,stats = client.start_new_tool_chain(prompt_data=prompt_data)
    # endregion

    return ans, stats


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
) -> (
    list | None
):  # will return list of dataframes containing all data to be saved in csv.

    if ("(" in function) or (")" in function):
        answers[function] = [
            (
                "Not a valid function_n name:: [ENTER THE FUNCTION NAME WITHOUT '()']",
                None,
            )
        ]
        return None
    print(f"PROCESSING FUNCTION -->{BOLD}{GREEN}", function, f"{RESET}", end="\n\n")
    # region INTIALIZING STATE AND OTHER VARIABLES.
    STATE = State()
    FILE_NAME_BYTES: dict[str, bytes] = {
        key: value[1] for key, value in STATE.get("TREES").items()
    }
    list_indices = functions_identified[function].get("indices")
    get_upper = functions_identified[function].get("get_upper")
    function_answer_csv = []  # where will we store the function's csv rows.

    dependent_functions = list(
        filter(
            lambda x: x in functions_identified,
            functions_identified[function].get("dependent_functions"),
        )
    )  # filtering out the dependent function based on what are in the project.
    check_other_functions: bool = (
        True if dependent_functions[0] != function else False
    )  # assuming that after filtering we are left only with one function
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
    # return None
    stats_json_path = Path(
        f"/home/seigyo/c_repo/c_repo/results/csv_results/stats/{STATE.get('PROJECT_NAME')}_STATS.json"
    )
    (stats_json_path.parent / "stats").mkdir(parents=True, exist_ok=True)
    # endregion

    # region INTIALIZING EMPTY TOKEN VARIABLES
    dummy_token = {"Input_tokens": 0, "Output_tokens": 0, "Total_tokens": 0}
    empty_stats = {
        "Tokens": dummy_token,
        "Iterations": 0,
        "Random_tool_calls": 0,
        "Other_tool_errors": 0,
        "Incorrect_details": [],
    }
    # endregion

    def write_json_file(data):
        with open(stats_json_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    # region INTIALIZING THE FUNCTION_DICT
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

    FUNCTION_INPUT_TOKEN = FUNCTION_DICT[function].get("Total_Input", 0)
    FUNCTION_OUTPUT_TOKEN = FUNCTION_DICT[function].get("Total_Output", 0)
    # endregion

    # region FINDING OUT FROM WHICH PART WE NEED TO RESUME
    if FUNCTION_DICT[function]["Each_Path_Tokens"] == []:
        PATH_TO_START_WITH = 1
    else:
        PATH_TO_START_WITH = (
            int(list(FUNCTION_DICT[function]["Each_Path_Tokens"][-1].keys())[0]) + 1
        )
    print("Need to start with the path", PATH_TO_START_WITH)
    # endregion

    # region MAKING CALL_GRAPH AND GETTING DATA
    macro_data = possible_paths_data = None
    call_graph_with_paths = orchestrate(
        project_strcuture=project_structure,
        trees=trees,
        required_func=function,
        main_file_name=main_file_name,
        function_pointer_args=function_pointer_args,
        file_functions=file_functions,
        return_whole_tree=check_other_functions,  # this will tell if path_nodes will be returned or not.
    )

    if not call_graph_with_paths:
        print(f"{BOLD}{RED}No possible paths for: {GREEN}{function}{RESET}")
        return None
    macro_data, possible_paths_data = call_graph_with_paths
    if not macro_data:
        print("For ", function, " no macro data")
    call_graph_determined_datas = [
        path[1] for path in possible_paths_data
    ]  # tuple of [list[str],dict]
    path_nodes: list[list[CallTreeNode]] | None = [
        path[0][1] for path in possible_paths_data if path[0][1] is not None
    ]  # NONE WHEN NO DEPENDENT FUNCTION OTHER THEN ITSSELF.
    path_strs: list[list[str]] = [path[0][0] for path in possible_paths_data]
    # print_or_return_possible_paths_trees(paths=path_strs) FOR JUST PRINTING ALL PATH TREES BEFOREHAND FOR OBSERVATION..
    # endregion

    print("TOTAL PATHS FOR THIS FUNCTION: ", len(path_strs))
    # return

    make_graph(
        paths=path_strs
    )  # for storing the this function's call_graph. (mermaid and pyvis)

    # region PARSING FOR THE FUNCTION AND GETTING CONTEXTS FOR ALL PATHS.
    print(f"PARSING FOR FUNCTION -> {function}")
    parser = parseFiles(
        project_structure=project_structure,
        paths=path_strs,
        macro_data=macro_data,
        file_name_bytes=FILE_NAME_BYTES,
    )
    contexts = parser.get_parsed_results(get_upper=get_upper)
    # endregion

    if len(path_nodes) > 0:

        # region GETTING DEPENDENT FUNCTION PATH FROM EACH FUNCTION PATH (APPLYING DFS IN IT AND GETTING LAST PATH)
        paths_to_dependent: list[list[str]] = []
        print_or_return_possible_paths_trees(
            paths=path_nodes,
            dependent_function=dependent_functions[0],
            result_path_list=paths_to_dependent,
        )  # returns the tree made from the nodes.
        # set the paths to dependent in this function ...
        #  endregion

        # region PARSING FOR DEPENDENT FUNCTION
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
        # endregion

        for index, (path, context) in enumerate(contexts, start=1):
            if index < PATH_TO_START_WITH:
                continue
            call_graph_data = call_graph_determined_datas[
                index - 1
            ]  # data received for this path from call_graph
            # It has [function_name,launch_via,call_function,function_name_src,target_name_src] -> NEEDED process_name(folder_name), type + aidetermined...
            print(f"{BOLD}{GREEN}PROCESS PATH_{index}{RESET}")
            print("-" * 20, "PATH AND CONTEXT", "-" * 20)

            block_regex = r"\[([^\[\]]*)\]"
            path_str = "->".join(
                map(lambda x: re.sub(block_regex, "", x), path)
            )  # path like func_a->func_b ...

            print(
                highlight(context, CLexer(), TerminalFormatter())
            )  # for highlighting the c code in terminal..
            print("-" * 56)
            stats_dict1 = None
            # region WHEN WE NEED TO DETERMINE FUNCTION_TYPE LIKE (mpf_mfs_getrec etc)
            if (
                STATE.get("FUNCTION_TYPES").get(function, {}).get("type", "NO DATA")
                == "WRITEF/READF"
            ):
                call_graph_data = {
                    **call_graph_data,
                    "process_name": Path(project_path).name,
                }
                print(" STEP - 1 DETERMINING THE CALL_TYPE")

                ans = run_with_retry(
                    llm_calls,
                    args=(
                        project_structure,
                        function,
                        list_indices,
                        context,
                        path_str,
                        get_upper,
                    ),
                )
                if ans:
                    validated_model, stats = ans
                else:
                    # get out of the loop..
                    final_combined_data = {
                        **call_graph_data,
                        "target_number": {
                            "path_str": "->".join(path),
                            "ans": ["UNRESOLVED"],
                        },
                        "call_number": -1,
                        "type": "NO DATA",
                    }
                    tokens = TokenCount(Input_tokens=0, Output_tokens=0, Total_tokens=0)
                    FUNCTION_DICT[function]["Each_Path_Tokens"].append(
                        {index: tokens.model_dump()}
                    )
                    FUNCTION_DICT[function].update(
                        {
                            "Total_Tokens": FUNCTION_INPUT_TOKEN
                            + FUNCTION_OUTPUT_TOKEN,
                            "Total_Output": FUNCTION_OUTPUT_TOKEN,
                            "Total_Input": FUNCTION_INPUT_TOKEN,
                        }
                    )
                    combined_model = Combined.model_validate(final_combined_data)
                    save_dict_csv(data_dict=combined_model.model_dump(), save=True)
                    write_json_file(data=FUNCTION_DICT)
                    answers[function].append(
                        (combined_model, Stats.model_validate(empty_stats))
                    )
                    print(f"DONE WITH PATH {index}")
                    continue

                stats_dict1 = stats.model_dump()
                FUNCTION_INPUT_TOKEN += stats_dict1.get("Tokens").get("Input_tokens")
                FUNCTION_OUTPUT_TOKEN += stats_dict1.get("Tokens").get("Output_tokens")
                validated_model_dict = (
                    validated_model.model_dump()
                )  # will give as dict {'output': str_value...} str_value = arg_number:value,...
                print("VALIDATED MODEL CONVERTED TO DICT")
                console.print(validated_model_dict)
                output_string = validated_model_dict.get("output", "")  # arg_num:value
                call_number = validated_model_dict.get("call_number") or -1
                call_graph_data = {
                    **call_graph_data,
                    "type": output_string,
                    "call_number": call_number,
                    "process_name": Path(project_path).name,
                }  # determined the type.
            # endregion
            else:
                # region WHEN FUNCTION_TYPE IS ALREADY DECIDED (ONLY CALL_GRAPH DATA CHANGES)
                call_graph_data = {
                    **call_graph_data,
                    "process_name": Path(project_path).name,
                    "type": STATE.get("FUNCTION_TYPES")
                    .get(function, {})
                    .get("type", "NO DATA"),  # here type is fixed.
                }
            # endregion

            print("NOW RUNNING FOR THE DEPENDENT FUNCTION.")
            new_path, context_new = dependent_contexts[index - 1]
            print("CONTEXT FOR THE DEPENDENT FUNCTION")
            print(highlight(context_new, CLexer(), TerminalFormatter()))

            new_path_str = "->".join(
                map(lambda x: re.sub(block_regex, "", x), new_path)
            )
            ans = run_with_retry(
                llm_calls,
                args=(
                    project_structure,
                    dependent_functions[0],
                    dependent_function_indices,
                    context_new,
                    new_path_str,
                    dependent_function_get_upper,
                ),
            )  # as the functions right now depend only on mpf_mfs_open...

            if ans:
                validated_model, stats = ans
            else:
                final_combined_data = {
                    **call_graph_data,
                    "target_number": {
                        "path_str": "->".join(path),
                        "ans": ["UNRESOLVED"],
                    },
                }
                final_combined_data["call_number"] = -1
                combined_model = Combined.model_validate(final_combined_data)
                tokens = TokenCount(Input_tokens=0, Output_tokens=0, Total_tokens=0)
                FUNCTION_DICT[function]["Each_Path_Tokens"].append(
                    {index: tokens.model_dump()}
                )
                FUNCTION_DICT[function].update(
                    {
                        "Total_Tokens": FUNCTION_INPUT_TOKEN + FUNCTION_OUTPUT_TOKEN,
                        "Total_Output": FUNCTION_OUTPUT_TOKEN,
                        "Total_Input": FUNCTION_INPUT_TOKEN,
                    }
                )
                write_json_file(data=FUNCTION_DICT)
                save_dict_csv(data_dict=combined_model.model_dump(), save=True)
                answers[function].append(
                    (combined_model, Stats.model_validate(empty_stats))
                )
                print(f"DONE WITH PATH {index}")
                continue

            validated_model_dict = (
                validated_model.model_dump()
            )  # will give as dict {'output': str_value...} str_value = arg_number:value,...
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
                FUNCTION_DICT[function]["Each_Path_Tokens"].append(
                    {index: stats_dict2["Tokens"]}
                )

            FUNCTION_DICT[function].update(
                {
                    "Total_Tokens": FUNCTION_INPUT_TOKEN + FUNCTION_OUTPUT_TOKEN,
                    "Total_Output": FUNCTION_OUTPUT_TOKEN,
                    "Total_Input": FUNCTION_INPUT_TOKEN,
                }
            )
            write_json_file(data=FUNCTION_DICT)
            print("VALIDATED MODEL CONVERTED TO DICT")
            console.print(validated_model_dict)

            # extracting data from ai_output.
            output_string = validated_model_dict.get("output", "")  # arg_num:value
            call_number_dependent = validated_model_dict.get("call_number") or -1
            values_found = []
            for elements in output_string.split(","):
                try:
                    file_num = int(elements.split(":")[1])
                    values_found.append(file_num)
                except Exception as e:
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
            # function_answer_csv.append(save_dict_csv(data_dict=combined_model.model_dump(),save=False))
            save_dict_csv(data_dict=combined_model.model_dump(), save=True)
            answers[function].append((combined_model, stats))
            print(f"DONE WITH PATH {index}")

    else:
        for index, (path, context) in enumerate(contexts, start=1):
            if index < PATH_TO_START_WITH:
                continue
            call_graph_data = call_graph_determined_datas[
                index - 1
            ]  # data received for this path from call_graph
            # It has [function_name,launch_via,call_function,function_name_src,target_name_src] -> NEEDED process_name(folder_name), type + aidetermined...
            type_of_func = STATE.get("FUNCTION_TYPES").get(function).get("type")
            call_graph_data = {
                **call_graph_data,
                "process_name": Path(project_path).name,
                "type": type_of_func if type_of_func else "NO DATA",
            }

            console.print(call_graph_data)  # now need to add the target(list[strs])
            print(f"{BOLD}{GREEN}PROCESS PATH_{index}{RESET}")
            print("-" * 20, "PATH AND CONTEXT", "-" * 20)

            block_regex = r"\[([^\[\]]*)\]"
            path_str = "->".join(map(lambda x: re.sub(block_regex, "", x), path))
            # for i,p in enumerate(path):
            #     path_str += re.sub(block_regex,'',p)
            #     if i<len(path)-1:
            #         path_str+='->'

            # pprint(path_str)
            print(
                highlight(context, CLexer(), TerminalFormatter())
            )  # for highlighting the c code in terminal..
            print("-" * 56)
            if (
                "pmf" in function and len(list_indices) == 0
            ):  # in case of pmf functions and no list_indices.
                # then we don't need to call llm..
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
                tokens = TokenCount(Input_tokens=0, Output_tokens=0, Total_tokens=0)
                FUNCTION_DICT[function]["Each_Path_Tokens"].append(
                    {index: tokens.model_dump()}
                )
                FUNCTION_DICT[function].update(
                    {
                        "Total_Tokens": FUNCTION_INPUT_TOKEN + FUNCTION_OUTPUT_TOKEN,
                        "Total_Output": FUNCTION_OUTPUT_TOKEN,
                        "Total_Input": FUNCTION_INPUT_TOKEN,
                    }
                )
                save_dict_csv(data_dict=combined_model.model_dump(), save=True)
                # answers[function].append((combined_model,stats))
                write_json_file(data=FUNCTION_DICT)
                console.print(final_combined_data)
                answers[function].append(
                    (combined_model, Stats.model_validate(empty_stats))
                )
                print(f"DONE WITH PATH {index}")
                continue
            ans = run_with_retry(
                llm_calls,
                args=(
                    project_structure,
                    function,
                    list_indices,
                    context,
                    path_str,
                    get_upper,
                ),
            )
            if ans:
                validated_model, stats = ans
            else:
                # get out of the loop..
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
                tokens = TokenCount(Input_tokens=0, Output_tokens=0, Total_tokens=0)

                FUNCTION_DICT[function]["Each_Path_Tokens"].append(
                    {index: tokens.model_dump()}
                )
                FUNCTION_DICT[function].update(
                    {
                        "Total_Tokens": FUNCTION_INPUT_TOKEN + FUNCTION_OUTPUT_TOKEN,
                        "Total_Output": FUNCTION_OUTPUT_TOKEN,
                        "Total_Input": FUNCTION_INPUT_TOKEN,
                    }
                )
                write_json_file(data=FUNCTION_DICT)
                save_dict_csv(data_dict=combined_model.model_dump(), save=True)
                # answers[function].append((combined_model,stats))
                answers[function].append(
                    (combined_model, Stats.model_validate(empty_stats).model_dump())
                )
                print(f"DONE WITH PATH {index}")
                continue
            # console.print(stats)
            validated_model_dict = (
                validated_model.model_dump()
            )  #       will give as dict {'output': str_value...} str_value = arg_number:value,...
            print("VALIDATED MODEL CONVERTED TO DICT")
            console.print(validated_model_dict)
            stats_dict = stats.model_dump()
            FUNCTION_INPUT_TOKEN += stats_dict.get("Tokens").get("Input_tokens")
            FUNCTION_OUTPUT_TOKEN += stats_dict.get("Tokens").get("Output_tokens")

            FUNCTION_DICT[function]["Each_Path_Tokens"].append(
                {index: stats_dict["Tokens"]}
            )
            FUNCTION_DICT[function].update(
                {
                    "Total_Tokens": FUNCTION_INPUT_TOKEN + FUNCTION_OUTPUT_TOKEN,
                    "Total_Output": FUNCTION_OUTPUT_TOKEN,
                    "Total_Input": FUNCTION_INPUT_TOKEN,
                }
            )

            write_json_file(data=FUNCTION_DICT)
            # extracting data from ai_output.
            output_string = validated_model_dict.get("output", "")  # arg_num:value
            # print("The final output_string", output_string)

            values_found: list[int | str | Literal["UNRESOLVED"]] = []
            call_number = validated_model_dict.get("call_number") or -1
            splitted = output_string.split(",")
            for elements in splitted:
                first_one = elements.split(":")[0]
                try:
                    if '"' in elements.split(":")[1]:
                        # string
                        values_found.append(elements.split(":")[1].strip('"'))
                    else:
                        file_num = int(elements.split(":")[1].strip('"'))
                        values_found.append(file_num)
                except Exception as e:
                    # values_found.append('UNRESOLVED')\
                    values_found.append(elements.split(":")[1].strip('"'))
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
            # function_answer_csv.append(save_dict_csv(data_dict=combined_model.model_dump(),save=False))
            save_dict_csv(data_dict=combined_model.model_dump(), save=True)
            answers[function].append((combined_model, stats))
            print(f"DONE WITH PATH {index}")

    return function_answer_csv


@time_it()
def trace_variable(project_path):
    STATE = State()
    pickle_dir = (
        Path(__file__).resolve().parent / "pickle_data/project_structures_pickle"
    )
    if not pickle_dir.exists():
        pickle_dir.mkdir(exist_ok=True, parents=True)

    project_structure_path = pickle_dir / f"{STATE.get('PROJECT_NAME')}.pkl"
    potential_main_files: list[str] | None = None
    if not project_structure_path.exists():
        print(
            "PROJECT STRUCTURE NEEDS TO BE RESOLVED. NO PICKLE FILE. IT WILL TAKE TIME...."
        )
        PROJECT_STRUCTURE, potential_main_files = return_project_mapping(
            show=False, project_path=project_path
        )
        PROJECT_STRUCTURE = dict(
            sorted(PROJECT_STRUCTURE.items(), key=lambda x: str(x[0]))
        )
        STATE.set("PROJECT_STRUCTURE", PROJECT_STRUCTURE)
        with open(project_structure_path, "wb") as f:
            pickle.dump((PROJECT_STRUCTURE, potential_main_files), f)
    else:
        with open(project_structure_path, "rb") as f:
            PROJECT_STRUCTURE, potential_main_files = pickle.load(f)  # already sored.

        STATE.set("PROJECT_STRUCTURE", PROJECT_STRUCTURE)
    print("THE MAIN FILES ARE: ", potential_main_files)
    # project_tree_path = pickle_dir/f'{STATE.get('PROJECT_NAME')}_tree.pkl'
    trees = Preprocess().preprocess(
        project_structure=PROJECT_STRUCTURE
    )  # str, tuple[Tree,bytes]

    STATE.set("TREES", trees)
    # print(extract_includes(filepath=PROJECT_STRUCTURE['dio000d.c']))
    # sys.exit()
    PROJECT_STRUCTURE = {
        key: str(PROJECT_STRUCTURE[key]) for key in PROJECT_STRUCTURE.keys()
    }
    # PROJECT_STRUCTURE = dict(sorted(PROJECT_STRUCTURE.items(),key=lambda x: str(x[0])))
    # console.print(PROJECT_STRUCTURE)
    print("LENGTH OF PROJECT_STRUCTURE", len(PROJECT_STRUCTURE))
    FUNCTION_POINTER_ARGS = STATE.get("FUNCTION_POINTER_ARGS")
    FILE_FUNCTIONS = {}

    main_file_name = None
    bad_main_files = []
    macros = {}
    file_includes: dict[str, list] = {}
    for files in PROJECT_STRUCTURE.keys():
        macros[files] = extract_all_macros(PROJECT_STRUCTURE[files])
        file_includes[files] = extract_includes(PROJECT_STRUCTURE[files])
        if files.endswith(".h"):
            continue

        file_path = PROJECT_STRUCTURE[files]
        functions = get_local_function_definitions(
            code_bytes=trees[files][1]
        )  # function_name:dict(info of this function.)
        # if files == 'dio000d.c':
        #     print('This is the files and functions for the dio000d.c',files,functions)
        if "main" in functions.keys() and any(files == x for x in potential_main_files):
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
    # STATE.set('TREES',trees)

    functions_identified = identify_funs_to_trace(
        project_structure=PROJECT_STRUCTURE, trees=trees
    )  # functions and
    if functions_identified == {}:
        print(
            f"{BOLD}{RED}NO FUNCTIONS IDENTIFIED IN THE PROJECT {project_path.name}.{RESET}"
        )
        return None
    console.print(
        "-" * 10, "DETECTED FUNCTIONS NEEDS TO BE TRACED AND THEIR ARG. NUMS.", "-" * 10
    )

    console.print(functions_identified)
    # sys.exit()
    console.print("-" * 60)
    import time

    answers: dict[str, list[tuple[BaseModel, BaseModel]]] = defaultdict(
        list
    )  # funct_name [(combinedModel,Stats)]

    # print('dio000d.c' in PROJECT_STRUCTURE)
    print("This is the main file::", main_file_name)
    # sys.exit()
    data_csvs = []
    for function in functions_identified:
        # if function == 'mpf_mfs_open' or function=='mpf_mfs_close' or function=='mpf_mfs_getrec': continue
        # if 'pmf_addevent' not in function: continue
        # if function != 'mpf_mfs_open':continue
        # if project_path.name == 'dio120d' : continue
        STATE.set("CURRENT_PROCESSED_FUNCTION", function)
        function_dataframes: list | None = make_llm_calls_for_function(
            function=function,
            trees=trees,
            functions_identified=functions_identified,
            answers=answers,  # will be modified in place
            main_file_name=main_file_name,
            function_pointer_args=FUNCTION_POINTER_ARGS,  # TODO: need the same for these or make a state class holding these vars together.
            file_functions=FILE_FUNCTIONS,
            project_structure=PROJECT_STRUCTURE,
            project_path=project_path,
        )
        if function_dataframes is not None:
            data_csvs = [*data_csvs, *function_dataframes]

    console.print(data_csvs)
    # save_dicts_to_csv(data_csvs)

    return answers  # full answer plus+ stats


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
        not_done = ['dio110d','dio100d']
        projects = list(filter(lambda x: x in not_done, projects))
        # projects = ['dio860d']
        # projects = ['dio860d', 'dio220d','dio260d','dio260d_nobori','dio270d','dio310d','dio410d','dio210d','dio210d_nobori','dio110d','dio110d_nobori']
        # projects = ['dio260d_nobori','dio270d','dio310d','dio410d','dio210d','dio210d_nobori','dio110d','dio110d_nobori','dio000d','dio100d','dio120d',
        #             'dio130d','dio140d','dio150d','dio160d','dio170d','dio175d','dio220d','dio260d','dio600d','dio690d']
        # projects = ['dio800d']
        # projects = ['dio000d','dio110d','dio110d_nobori','dio120d','dio130d','dio']
        # to_exclude = ['libdio','libdio_ora','libDioKyusei','libDioTrace','libLocal','tools']
        projects = list(filter(lambda x: x not in to_exclude[parent_project], projects))
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
