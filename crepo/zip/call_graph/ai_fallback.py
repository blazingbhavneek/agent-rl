import re
from pathlib import Path

# from .call_graph import find_highest_parent_of_node,get_fn_name,dfs_for_finding_path
# from .call_graph import FILE_NAME_REGEX,RED,ORANGE,GREEN,BOLD,RESET
import clang.cindex
from ollama import Client

from .data_classes import CallTreeNode, FunctionNode

FILE_NAME_REGEX = r"\[(.*?)\]"
GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"


def get_fn_name(name: str) -> str:
    return re.sub(FILE_NAME_REGEX, "", name)


def dfs_for_finding_path(
    node: CallTreeNode, required_func: str, paths: list[list[str]], curr_path: list[str]
) -> None:
    """
    DFS on the logical CallTreeNode structure to find paths.
    """
    if not node:
        return

    # Check if current node matches the required function name
    # We check the function name directly, ignoring file context/macro expansion for the match
    # unless the user specifically wants the macro entry point (handled by the name check).
    if node.func_node.name == required_func:
        curr_path.append(node.func_node.label)
        paths.append(curr_path[:])
        curr_path.pop()
        return

    curr_path.append(node.func_node.label)
    for child in node.children:
        dfs_for_finding_path(
            node=child, required_func=required_func, paths=paths, curr_path=curr_path
        )
    curr_path.pop()
    return


def find_highest_parent_of_node(
    tree_objects: dict[str, CallTreeNode], function_name: str
):
    def return_parent_key(fn_name: str) -> str | None:
        fn_name = get_fn_name(fn_name)
        for nodes in tree_objects.keys():
            if get_fn_name(nodes) == fn_name:
                continue
            paths = []
            dfs_for_finding_path(
                node=tree_objects[nodes],
                required_func=fn_name,
                paths=paths,
                curr_path=[],
            )
            if len(paths) > 0:
                return nodes
        return None

    highest_parent_keys = []
    function_name = get_fn_name(function_name)
    for nodes in tree_objects.keys():
        if get_fn_name(nodes) == function_name:
            continue
        paths: list[list[str]] = []
        dfs_for_finding_path(
            node=tree_objects[nodes],
            required_func=function_name,
            paths=paths,
            curr_path=[],
        )
        if len(paths) > 0:
            parent = return_parent_key(fn_name=get_fn_name(nodes))
            if parent is None:
                highest_parent_keys.append(nodes)
            else:
                continue

    return highest_parent_keys


class fallbackAi:
    def __init__(
        self,
        project_structure: dict[str, str],
        tree_objects: dict[str, CallTreeNode],
        target_func: str,
        function_map: dict,
    ):
        self.model = "gpt-oss:20b"
        self.temp = 0.0
        self.function_map = function_map
        self.host = "http://10.160.152.38:11434"
        self.client = Client(self.host)
        self.project_structure = project_structure
        self.tree_objects = tree_objects  # Now holds CallTreeNode
        self.target_func = target_func
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "find_all_occurences",
                    "description": "Searches all project files to locate where the given symbol_name is used or defined.",
                    "parameters": {
                        "type": "object",
                        "properties": {"symbol_name": {"type": "string"}},
                        "required": ["symbol_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_highest_parent",
                    "description": "Finds the highest-level callers (entry points) for a specific function.",
                    "parameters": {
                        "type": "object",
                        "properties": {"function_name": {"type": "string"}},
                        "required": ["function_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "trace_path",
                    "description": "Attempts to find a call path from a start function to an end function.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_func": {"type": "string"},
                            "end_func": {"type": "string"},
                        },
                        "required": ["start_func", "end_func"],
                    },
                },
            },
        ]

    def clear_comments(self, content: str) -> str:
        comment_regex = r"(^|\s)(\/\/.*|\/\*[\s\S]*?\*\/)"
        return re.sub(comment_regex, "", content)

    def read_file_content_raw(self, file_path: Path):
        if not file_path.exists():
            return None
        encodings = ["utf-8", "shift_jis", "euc-jp", "cp932", "latin-1"]
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except:
                continue
        return None

    def lookup_function_in_registry(self, function_name):
        input = {"function_name": function_name}
        FUNCTION_MAP = self.function_map
        if FUNCTION_MAP is not None:
            if FUNCTION_MAP.get(function_name) is None:
                return (
                    "The function is not stored look up in the headers or c files associated.",
                    0,
                    input,
                )
            res = f"Prototype: {FUNCTION_MAP.get(function_name).get('prototype')}\n"
            return res
        else:
            return (
                "Lookup in the headers or c files no functions are stored right now.",
                0,
                input,
            )

    def find_all_occurences(self, symbol_name: str):
        PROJECT_STRUCTURE = self.project_structure
        input_data = {"symbol_name": symbol_name}
        symbol_name = re.sub(FILE_NAME_REGEX, "", symbol_name)

        if symbol_name and (".c" in symbol_name or ".h" in symbol_name):
            return (
                "This tool is to search for symbols in the files not the files itself.",
                -1,
                input_data,
            )

        results = []
        registry_result = self.lookup_function_in_registry(symbol_name)

        if isinstance(registry_result, str):
            results.append(registry_result)
        index = clang.cindex.Index.create()

        for file_name in PROJECT_STRUCTURE.keys():
            file_path = PROJECT_STRUCTURE[file_name]
            is_c_file = file_name.endswith(".c")

            if is_c_file:
                try:
                    tu = index.parse(file_path, args=["-fsyntax-only"])
                    content = self.read_file_content_raw(Path(file_path))
                    if not content:
                        continue
                    lines = content.splitlines()

                    for i, line in enumerate(lines):
                        if self.clear_comments(line.strip()).strip() == "":
                            continue
                        if re.search(
                            rf"\b{re.escape(symbol_name)}\b", line
                        ) or re.search(rf"\b{re.escape(symbol_name.lower())}\b", line):
                            line_num = i + 1
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
                                f"file_name: **{file_name}** | function_name: **{container_func}** | line: {line.strip()}"
                            )
                except Exception as e:
                    continue
            else:
                content = self.read_file_content_raw(Path(file_path))
                if content:
                    lines = content.splitlines()
                    for i, line in enumerate(lines):
                        if self.clear_comments(line.strip()).strip() == "":
                            continue
                        if re.search(
                            rf"\b{re.escape(symbol_name)}\b", line
                        ) or re.search(rf"\b{re.escape(symbol_name.lower())}\b", line):
                            results.append(
                                f"file_name: **{file_name}** | line_number: **{i+1}** | line: {line.strip()}"
                            )

        if results:
            return (
                f"ALL OCCURENCES OF {symbol_name} ARE (INCLUDING FUNCTION_PROTOTYPES IF ANY): (**NO OTHER DATA RELATED TO THIS SYMBOL**)\n"
                + "\n".join(results)
            )

        return (f"Symbol '{symbol_name}' not there in the project.", 0, input_data)

    def get_highest_parent(self, function_name: str) -> str:
        try:
            parents = find_highest_parent_of_node(
                tree_objects=self.tree_objects, function_name=function_name
            )
            if not parents:
                return f"No parents found for function '{function_name}'. It might be a top-level entry point or disconnected."
            return f"The highest parents (entry points) calling '{function_name}' are: {', '.join(parents)}"
        except Exception as e:
            return f"Error finding highest parent: {str(e)}"

    def trace_path(self, start_func: str, end_func: str) -> str:
        start_key = None
        if start_func in self.tree_objects:
            start_key = start_func
        else:
            for key in self.tree_objects.keys():
                if get_fn_name(key) == start_func:
                    start_key = key
                    break

        if not start_key:
            return f"Start function '{start_func}' not found in the processed call graph trees."

        paths = []
        # Updated to use CallTreeNode
        dfs_for_finding_path(
            node=self.tree_objects[start_key],
            required_func=end_func,
            paths=paths,
            curr_path=[],
        )

        if not paths:
            return f"No path found from '{start_func}' to '{end_func}'."

        formatted_paths = []
        for p in paths:
            formatted_paths.append(" -> ".join(p))

        return f"Found {len(paths)} path(s):\n" + "\n".join(formatted_paths)

    def complete_path(self, path) -> str:
        required_function = self.target_func
        system_prompt = f"""
        You are a C-project call graph resolution assistant. Your goal is to find ALL possible execution paths from main() to {required_function} by tracing backward through callback/event registrations.
        ... (Prompt remains same as original) ...
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"The given path that our analyzer can resolve is {path}",
            },
        ]

        max_iterations = 10
        iteration = 1

        while True:
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    options={"temperature": 0, "num_ctx": 8192},
                )
            except Exception as e:
                return f"Error communicating with LLM: {e}"

            msg = response.get("message", {})
            messages.append(msg)
            print(f"-" * 80)
            print(f"ITERATION {iteration}")
            print(f"-" * 80)
            print(f"{GREEN}:::::::LLM RESPONSE:::::::{RESET}\n{response.message}")
            if not msg.get("tool_calls"):
                print(
                    f"\n{BOLD}{GREEN}--- FINAL RESULT ---\n",
                    response.message.content,
                    f"{RESET}",
                )
                break

            for tool_call in msg.get("tool_calls", []):
                function_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]

                if isinstance(arguments, str):
                    try:
                        import json

                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                tool_result = ""
                try:
                    if function_name == "find_all_occurences":
                        tool_result = self.find_all_occurences(
                            arguments.get("symbol_name")
                        )
                    elif function_name == "get_highest_parent":
                        tool_result = self.get_highest_parent(
                            arguments.get("function_name")
                        )
                    elif function_name == "trace_path":
                        tool_result = self.trace_path(
                            arguments.get("start_func"), arguments.get("end_func")
                        )
                    else:
                        tool_result = f"Unknown tool: {function_name}"
                except Exception as e:
                    tool_result = f"Error executing {function_name}: {str(e)}"
                print(f"{BOLD}{ORANGE}Tool call result\n{tool_result}{RESET}")
                if isinstance(tool_result, tuple):
                    messages.append(
                        {
                            "role": "tool",
                            "name": function_name,
                            "content": tool_result[0],
                        }
                    )
                else:
                    messages.append(
                        {"role": "tool", "name": function_name, "content": tool_result}
                    )
            iteration += 1
        final_message = messages[-1].get("content", "")
        return final_message
