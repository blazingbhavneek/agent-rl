import re
from pathlib import Path

import clang.cindex

from helpers.extract_functions_from_c import get_local_function_definitions

comment_regex = r"(^|\s)(\/\/.*|\/\*[\s\S]*?\*\/)"


def tool(func):
    from state.state import State

    STATE = State()
    tools = STATE.get("TOOLS", {})
    tools[func.__name__] = func
    STATE.set("TOOLS", tools)
    return func


def set_tool_def() -> None:
    """It sets the tool definition in the state and also will populate the functions in the state."""
    # returns a dict of dict{function_name: function}
    tools = [
        {
            "type": "function",  # ✅ REQUIRED wrapper
            "function": {  # ✅ REQUIRED nested object
                "name": "read_file_content",
                "description": "Reads from a file. **Can't read entire files**",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "name of the file to open. Like a.c, b.h",
                        },
                        "function_name": {
                            "type": "string",
                            "description": "**Required only for .c files**. The name of the function to read. eg: abc",
                        },
                        "line_number": {
                            "type": "integer",
                            "description": "**Required only for .h files**. The line number to center the snippet on. eg: 20",
                        },
                    },
                    "required": ["file_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_definition",
                "description": "Searches all project files to locate where a symbol is defined. DON'T USE THIS TOOL AGAIN AND AGAIN WITH THE SAME **symbol_name** AS IT RETURNS ALL THE OCCURENCES AT ONCE.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol_name": {
                            "type": "string",
                            "description": "The exact name of the function or variable to find. eg. abc",
                        }
                    },
                    "required": ["symbol_name"],
                },
            },
        },
    ]
    from state.state import State

    STATE = State()
    STATE.set("TOOL_DEFINITION", tools)


@tool
def read_file_content(
    file_name: str,
    function_name=None,
    line_number=None,
    project_structure: dict[str, str] = None,
):
    from state.state import State

    STATE = State()

    def resolve_file_name(
        file_name: str, project_structure: dict[str, str]
    ) -> Path | None:
        # search in the project_files.\
        if file_name not in project_structure.keys():
            # but if not found then it may be a relative path so look for this path in the paths of the files ...
            name = file_name.split("/")[-1]
            if name in project_structure.keys():
                return Path(project_structure.get(name))
            else:
                return None
            # return Path(PROJECT_STRUCTURE.get(name,None)) # we are expecting the paths in the project_structure to be absolute (SO we are not using .resolve() else it will append the cwd in front....)
        else:
            return Path(str(project_structure.get(file_name)))

    input_func = {
        "file_name": file_name,
        "function_name": function_name,
        "line_number": line_number,
    }
    if file_name.strip() == "":
        # Then we need to check for function_name....
        if function_name and type(find_definition(symbol_name=function_name)) == list:
            return (
                f"You need to provide fine_name to find the function. Use find_defition with this function_name to find the file",
                -1,
                input_func,
            )
        else:
            return (
                f"**file_name** is a required parameter to read a file",
                -1,
                input_func,
            )
    else:

        file_path = resolve_file_name(
            file_name=file_name, project_structure=project_structure
        )  # path object...
        # print(file_path)

        if not file_path or not file_path.exists():
            # then the file_path is none for the file_name or it doesn't exist..
            if function_name:

                return (
                    f"Given file_name not found in project.To get the info for this function you can use the find_definition tool to get file_path for this function.",
                    -1,
                    input_func,
                )

            return f"**file_name** is required to access a file", -1, input_func

    # Use your robust encoding-aware reader to get all lines
    # raw_content = read_file_content_raw(file_path)
    raw_content = STATE.get("TREES")[file_name][1].decode("latin-1")
    # print(raw_content)
    if raw_content is None:
        return (
            f"Error: Could not decode file {file_path} with supported encodings.",
            0,
            input_func,
        )

    lines = raw_content.splitlines()
    is_c_file = file_name.endswith(".c")
    is_h_file = file_name.endswith(".h")

    # --- Header File Logic: Line-based snippet ---
    if is_h_file:
        if line_number is None:
            return (
                f"Error: To read a header file you need to provide the line_number which you want to read.",
                -1,
                input_func,
            )

        line_idx = int(line_number) - 1
        if 0 <= line_idx < len(lines):
            start = max(0, line_idx - 10)
            end = min(len(lines), line_idx + 11)
            # snip = [clear_comments(line.strip()).strip() for line in lines[start:end]]
            snip = lines[start:end]
            snippet = ""
            for line in snip:
                if line != "":
                    snippet += line + "\n"
                    # snippet+='\n'

            # snippet = "".join(lines[start:end])
            return f"--- Snippet (around Line {line_number}) ---\n{snippet}"
        return f"Error: Line {line_number} out of range.", 0, input_func

    # --- Source File Logic: Function-based extraction ---
    elif is_c_file:

        # TODO: FIX THIS PART..
        FILE_FUNCTIONS = STATE.get("FILE_FUNCTIONS")
        function_infos = FILE_FUNCTIONS.get(file_name)
        start_line = function_infos.get(function_name, {}).get("start_line")
        end_line = function_infos.get(function_name, {}).get("end_line")
        # raw_content_lines = raw
        print("STart line and end lines", start_line, end_line)
        if not function_name:

            return (
                f"For accessing .c files **function_name** is necessary. In this c_file the function names are: \n{function_infos.keys()}",
                -3,
                input_func,
            )
        # if functon_name is given..
        if function_name not in function_infos:
            return (
                f"Error: Function '{function_name}' not found in {file_name}, The only functions in this file are: \n{function_infos.keys()}.",
                0,
                input_func,
            )

        # byte_string: bytes = TREES[file_name][1] # bytes
        code_string = lines[
            function_infos[function_name]["start_line"]
            - 1 : function_infos[function_name]["end_line"]
        ]
        return "\n".join(code_string)

    # return raw_content # Fallback for other files
    return (
        "The file_name is without extension. (**only .h and .c files are supported**)",
        -1,
        input_func,
    )


@tool
def find_definition(symbol_name: str, project_structure: dict[str, str]):
    from state.state import State

    STATE = State()

    input_func = {"symbol_name": symbol_name}
    if symbol_name and (".c" in symbol_name or ".h" in symbol_name):
        return (
            "This tool is to search for symbols in the files not the files itself.",
            -1,
            input_func,
        )

    results = []

    def lookup_function_in_registry(function_name):
        input_func = {"function_name": function_name}
        if STATE.get("FUNCTION_MAP") is not None:
            if STATE.get("FUNCTION_MAP").get(function_name) is None:
                return (
                    "The function is not stored look up in the headers or c files associated.",
                    0,
                    input_func,
                )
            res = f"Prototype: {STATE.get('FUNCTION_MAP').get(function_name).get('prototype')}\n"
            # res += f"Description: {FUNCTION_MAP.get(function_name).get('description')}\n"
            return res
        else:
            return (
                "Lookup in the headers or c files no functions are stored right now.",
                0,
                input_func,
            )

    # --- Step 1: Lookup in Registry ---
    registry_result = lookup_function_in_registry(symbol_name)

    # If the result is a string, it means the lookup was successful.
    # If it returns a tuple (error message, code, input_func), we ignore it and proceed to file search.
    if isinstance(registry_result, str):
        results.append(registry_result)

    # --- Step 2: Search in Files ---
    index = clang.cindex.Index.create()
    # print(project_structure)
    # sys.exit()
    for file_name in project_structure.keys():
        file_path = project_structure[file_name]
        is_c_file = file_name.endswith(".c")

        content = STATE.get("TREES")[file_name][1].decode("latin-1")
        if is_c_file:
            unsaved_file_name = "unsaved.c"
            # Use libclang to get structural context
            tu = index.parse(
                unsaved_file_name,
                args=["-fsyntax-only"],
                unsaved_files=[(unsaved_file_name, content)],
            )

            # Simple line-by-line search first to find the hit
            # content = read_file_content_raw(Path(file_path))
            # content = STATE.get('TREES')[file_name][1].decode('latin-1')
            if not content:
                continue
            lines = content.splitlines()

            for i, line in enumerate(lines):
                commented_out_line = line
                if commented_out_line == "":
                    continue
                if re.search(
                    rf"\b{re.escape(symbol_name)}\b", commented_out_line
                ) or re.search(
                    rf"\b{re.escape(symbol_name.lower())}\b", commented_out_line
                ):
                    line_num = i + 1
                    # Find which function contains this line
                    container_func = "Global Scope"
                    for node in tu.cursor.walk_preorder():
                        if (
                            node.kind == clang.cindex.CursorKind.FUNCTION_DECL
                            and node.is_definition()
                        ):
                            if (
                                node.extent.start.line
                                <= line_num
                                <= node.extent.end.line
                            ):
                                container_func = node.spelling
                                break

                    results.append(
                        f"file_name: **{file_name}** | function_name: **{container_func}** | line: {commented_out_line}"
                    )
        else:
            # Standard logic for headers/other files
            # content = read_file_content_raw(Path(file_path))
            if content:
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    commented_out_line = line
                    if commented_out_line == "":
                        continue
                    if re.search(
                        rf"\b{re.escape(symbol_name)}\b", commented_out_line
                    ) or re.search(
                        rf"\b{re.escape(symbol_name.lower())}\b", commented_out_line
                    ):
                        results.append(
                            f"file_name: **{file_name}** | line_number: **{i+1}** | line: {commented_out_line}"
                        )

    if results:
        return (
            f"ALL OCCURENCES OF {symbol_name} ARE: **DONT'T USE THIS TOOL WITH THIS SYMBOL AGAIN IF YOU CAN'T FIGURE OUT THEN REPORT ANSWER DON'T INVOKE ANY OTHER TOOLS**\n"
            + "\n".join(results)
        )

    return (
        f"SYMBOL '{symbol_name}' NOT THERE IN PROJECT. REPORT YOUR ANSWER",
        0,
        input_func,
    )
