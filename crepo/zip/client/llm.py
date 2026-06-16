import json
import re
from pprint import pprint

from ollama import Client
from openai import AsyncOpenAI, AzureOpenAI, OpenAI
from pydantic import BaseModel, Field
from tiktoken import get_encoding

from models import (
    Stats,# for reporting the stats.
    TokenCount,
    outputModel,  
    outputModelForReturn,
)

# from models import outputModel # for the llm's response in a string so is not too heavy.
# from models import outputModelForReturn # for the functions that only demand their return types to be studied..

FILE_NAME_REGEX = r"\[(.*?)\]"
GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"

# error codes 0 for non-llm errors, -1 for llm errors.


class OllamaClient:

    # later save to .env file
    PRINT_CONSOLE = True  #
    GEN_LOGS = True
    PRODUCTION_MODE = False  # will set the output to just logs

    def __init__(self, data):  # in the data we'll have the config as well as t
        # self.model = data.get('model','gpt-oss:20b')
        self.model = data.get("model", "gpt-oss:20b")
        if self.model == "gpt-oss:20b":
            self.enc = get_encoding("cl100k_base")
        else:
            self.enc = get_encoding("gpt4")
        self.temp = data.get("temp", 0.0)
        self.tool_functions = data.get("tool_functions", None)
        self.tools = data.get("tools", None)  # tool description provided to the mode.
        # self.num_ctx = data.get('num_ctx',40000)
        self.user_prompt = data.get("user_prompt", "")
        self.async_Openai = data.get("async", False)

        self.async_Openai = True
        self.system_prompt = data.get("system_prompt", "")
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]
        self.project_structure = data.get("project_structure", {})
        self.function_map = data.get("function_map", None)
        if not self.function_map:
            print("No function map")
            # sys.exit()
        self.output_model: outputModelForReturn | outputModel | None = data.get(
            "output_model", None
        )
        print(self.output_model)
        if not self.output_model:
            raise ValueError("OUTPUT MODEL NOT PROVIDED TO LLM CLASS.")
        # self.openai_model: bool = data.get('openai_model', None)

        self.openai_model: bool = False
        self.num_ctx = data.get("num_ctx", 110000)
        if self.openai_model:
            # self.client = (
            #     AzureOpenAI(
            #     api_key= 'sk-sXRoS3fmlU7bW797ulQysg',
            #     api_version = "2025-04-01-preview",
            #     azure_endpoint = "https://mesw-openai-api.azurewebsites.net/"
            #     )
            #         if not self.async_Openai else
            #             AsyncOpenAI(
            #             api_key='asdf',
            #             base_url='http://10.160.152.38:8000/v1'
            #             )
            # )
            self.client = OpenAI(
                api_key="asdf", base_url="http://10.160.152.38:42067/v1"
            )
        else:
            # self.client = Client(host = data.get('host','http://10.160.144.101:51021'))
            self.client = Client(host=data.get("host", "http://127.0.0.1:11434"))
            # self.c
            # print('')

        if self.PRODUCTION_MODE:
            self.PRINT_CONSOLE = False
            self.GEN_LOGS = True

    def give_client(self):
        return self.client

    @staticmethod
    def extract_argument_names(function) -> list[str]:
        import inspect

        signature = inspect.signature(function)
        return list(signature.parameters.keys())

    def give_json_format(self) -> dict | None:
        return self.output_model.model_json_schema()

    def _parse_response(self, raw_content: str) -> BaseModel | str:
        """Parse and validate LLM response against the Pydantic model."""
        if self.output_model is None:
            return raw_content
        try:
            parsed = json.loads(raw_content)
            return self.output_model.model_validate(parsed)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {raw_content}")
        except Exception as e:
            raise ValueError(f"Validation failed: {e}\nRaw: {raw_content}")

    def normal_chat(self) -> str:  # for just chatting won't support iterations...
        try:
            response = self.client.chat(
                model=self.model,
                messages=self.messages,
                options={"temperature": self.temp, "num_ctx": self.num_ctx},
            )
        except Exception as e:
            print("Error in llm calling ", e)

        return response.message.content

    def get_token_count(
        self, messages: list[dict] | None = None, message: str | None = None
    ) -> int:
        count = 0
        if messages:
            for m in messages:
                count += len(self.enc.encode(m.get("content", "")))
        elif message:
            count += len(self.enc.encode(message))
        return count

    def start_tool_chain(
        self, prompt_data: dict[str, dict]
    ) -> tuple[
        outputModel | outputModelForReturn, Stats
    ]:  # {'user or system_prompt' : {'string_vars': 'Their values'}}

        if not prompt_data:
            print("Error no system prompt_data")
            return (None, None)
        MAX_RETRY_ATTEMPTS = 5
        MAX_ITERATIONS_ALLOWED = 10
        ANS_FOUND = False
        INPUT_TOKEN = 0
        OUTPUT_TOKEN = 0
        random_tool_calls = 0
        incorrect_calls = []  # that has some king of erros.
        # region combine prompt_data with prompts
        argument_number_to_track = prompt_data.get("user_prompt").get(
            "argument_numbers"
        )
        for key in prompt_data:
            d = prompt_data.get(key)
            if len(d) == 0:
                continue
            # print(d)
            if key == "user_prompt":
                self.user_prompt = self.user_prompt.format(**d)
            else:
                self.system_prompt = self.system_prompt.format(**d)
        # endregion
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]
        # print("formated prompts")
        # print(self.user_prompt)
        # region extract the arguments for the tools
        fun_arg_names = {}
        for tool in self.tool_functions.keys():
            tool_def = self.tool_functions[tool]
            arg_names = OllamaClient.extract_argument_names(function=tool_def)
            fun_arg_names[tool] = arg_names
        # endregion
        format_schema = self.give_json_format()
        # pprint(json.dumps(format_schema,indent=2))
        final_validated_model = None
        if not self.openai_model:
            for attempt in range(0, MAX_RETRY_ATTEMPTS):
                if ANS_FOUND:
                    break
                iteration = 1
                self.messages = self.messages[:2]  # delete the history...
                INPUT_TOKEN += self.get_token_count(messages=self.messages)
                print(f"RETRY {attempt+1}/{MAX_RETRY_ATTEMPTS}")
                while iteration <= MAX_ITERATIONS_ALLOWED:
                    try:
                        response = self.client.chat(
                            model=self.model,
                            messages=self.messages,
                            tools=self.tools,
                            options={"temperature": self.temp, "num_ctx": self.num_ctx},
                            keep_alive=-1,
                        )
                        # print(messages,end='\n\n')
                    except Exception as e:
                        if attempt < MAX_RETRY_ATTEMPTS - 1:
                            print(f"Error communicating with LLM: {e}\nRETRYING..")
                            print(f"WAITING FOR 10s")
                            time.sleep(10)  # wait for 10 seconds
                            continue
                        else:
                            print(
                                f"Error communicating with LLM and RETRY LIMIT EXCEEDED."
                            )
                            # TODO:  RETURN A EMPTY OBJECT..
                        # return f"Error communicating with LLM: {e}",{}

                    msg = response.get("message", {})
                    OUTPUT_TOKEN += self.get_token_count(message=msg.get("content", ""))
                    OUTPUT_TOKEN += self.get_token_count(
                        message=msg.get("thinking", "")
                    )
                    self.messages.append(msg)
                    # print(f'{BOLD}{GREEN}',msg,f'{RESET}')
                    if not msg.get("tool_calls"):
                        last_message = self.messages[-1]

                        # ─────────────────────────────────────────────────
                        #  BUILD THE FORMATTING PROMPT (once)
                        # ─────────────────────────────────────────────────
                        format_prompt_content = (
                            f"Analyze: {last_message}\n"
                            f"Return ONLY valid JSON string like this json schema "
                            f"inside ```json``` block. JSON_SCHEMA: {json.dumps(format_schema)}.\n"
                        )
                        if self.output_model.__name__ != "outputModelForReturn":
                            format_prompt_content += f"""
                                "**DON'TS**:"
                                - For this function we are only tracking these arguments (1 based index) {argument_number_to_track} **don't report any other arguments**
                                - For answer use argument number and its value like 1:value,2:value (1,2 are the argument numbers asked to resolved in sorted argument number order)
                                - If argument's value is string report it without "" like lets say argument 1 is string then 1:value
                                - If argument's value is int then report it as it is like lets say argument 1 is int then 1: 2003 (If 2003 is the value.)
                                - If argument's value is not resolved then report as UNRESOLVED
                                 
                                - For call_number we can have both int or NONE
                                - DONT RETURN A LIST.
                            """
                        else:
                            format_prompt_content += (
                                "- **YOU JUST HAVE TO RETURN WHETHER ITS A READ OR WRITE "
                                "OPERATION ON THE RETURN POINTER NOTHING ELSE.**\n"
                            )

                        # ─────────────────────────────────────────────────
                        #  ★ RETRY LOOP — up to MAX_RETRIES attempts
                        # ─────────────────────────────────────────────────
                        MAX_RETRIES = 5
                        format_messages = [
                            {"role": "user", "content": format_prompt_content}
                        ]

                        for attempt in range(1, MAX_RETRIES + 1):
                            try:
                                print(
                                    f"\n{BOLD}JSON parse attempt {attempt}/{MAX_RETRIES}{RESET}"
                                )
                                INPUT_TOKEN += self.get_token_count(
                                    messages=format_messages
                                )
                                response = self.client.chat(
                                    model=self.model,
                                    messages=format_messages,
                                    think=True,
                                    options={
                                        "temperature": self.temp,
                                        "num_ctx": self.num_ctx,
                                    },
                                )
                                OUTPUT_TOKEN += self.get_token_count(
                                    message=response.message.content
                                )
                                OUTPUT_TOKEN += self.get_token_count(
                                    message=response.get("message", "").get(
                                        "thinking", ""
                                    )
                                )
                                raw_content = response.message.content
                                print(
                                    f"\n{BOLD}{GREEN}--- RAW RESPONSE ---{RESET}\n{raw_content}"
                                )

                                # ── Step 1: Extract ```json``` block ─────
                                json_match = re.search(
                                    r"```json\s*(.*?)\s*```", raw_content, re.DOTALL
                                )

                                if not json_match:
                                    raise ValueError(
                                        "No ```json``` block found in response"
                                    )

                                json_str = json_match.group(1).strip()

                                # ── Step 2: Validate JSON syntax ─────────
                                try:
                                    json.loads(json_str)  # syntax check
                                except json.JSONDecodeError as je:
                                    raise ValueError(f"Invalid JSON syntax: {je}")

                                # ── Step 3: Validate against Pydantic ────
                                parsed = self.output_model.model_validate_json(json_str)

                                # ── Success! ─────────────────────────────
                                print(
                                    f"{BOLD}{GREEN}JSON VALIDATED on "
                                    f"attempt {attempt}{RESET}"
                                )
                                final_validated_model = parsed
                                ANS_FOUND = True
                                break

                            except (ValueError, Exception) as e:
                                print(
                                    f"{BOLD}{RED}Attempt {attempt} failed: "
                                    f"{e}{RESET}"
                                )

                                if attempt < MAX_RETRIES:
                                    # ── Feed error back so LLM can fix it ─
                                    format_messages.append(
                                        {"role": "assistant", "content": raw_content}
                                    )
                                    format_messages.append(
                                        {
                                            "role": "user",
                                            "content": (
                                                f"Your previous response failed "
                                                f"validation:\n{e}\n\n"
                                                f"Please fix and return ONLY valid "
                                                f"JSON inside a ```json``` block "
                                                f"matching this schema:\n"
                                                f"{json.dumps(format_schema)}"
                                            ),
                                        }
                                    )
                                else:
                                    print(
                                        f"{BOLD}{RED}All {MAX_RETRIES} "
                                        f"attempts exhausted.{RESET}"
                                    )
                                    final_validated_model = None

                        break

                    print(f"-" * 80)
                    print(f"ITERATION {iteration}/{MAX_ITERATIONS_ALLOWED}")
                    print(f"-" * 80)
                    print(
                        f"{GREEN}:::::::LLM RESPONSE:::::::{RESET}\n{response.message}"
                    )
                    #  region Execute tool calls
                    for tool_call in msg.get("tool_calls", []):
                        function_name = tool_call["function"]["name"]
                        arguments = tool_call["function"]["arguments"]
                        arguments = {
                            **arguments,
                            "project_structure": self.project_structure,
                        }

                        if isinstance(arguments, str):
                            try:
                                # import json
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                arguments = {}

                        tool_result = ""
                        try:
                            if function_name in self.tool_functions.keys():
                                # the function is correct
                                args = fun_arg_names[
                                    function_name
                                ]  # arguments of the function...
                                tool_result = self.tool_functions[function_name](
                                    **arguments
                                )
                                print(
                                    f"{BOLD}{ORANGE}Tool call result\n{tool_result}{RESET}"
                                )
                            else:
                                # tool _request but not provided..
                                tool_result = (
                                    f"Unknown tool called. Please use only the given tools",
                                    -1,
                                    function_name,
                                )
                                random_tool_calls += 1

                        except Exception as e:
                            tool_result = (
                                f"Error executing {function_name}: {str(e)}",
                                0,
                                e,
                            )
                            incorrect_calls.append(tool_result)
                        if isinstance(tool_result, tuple):
                            INPUT_TOKEN += self.get_token_count(message=tool_result[0])
                            self.messages.append(
                                {
                                    "role": "tool",
                                    "name": function_name,
                                    "content": tool_result[0],
                                }
                            )
                        else:
                            INPUT_TOKEN += self.get_token_count(message=tool_result)
                            self.messages.append(
                                {
                                    "role": "tool",
                                    "name": function_name,
                                    "content": tool_result,
                                }
                            )
                    # endregion
                    iteration += 1
                    if iteration > MAX_ITERATIONS_ALLOWED:
                        print(f"MAX ITERATIONS REACHED, BREAKING.")
                        # iteration = 1
                        break

                # final_message = self.messages[-1].get('content', '')
        else:
            print("PROCESSING BY OPENAI CHAIN.")
            import time

            for attempt in range(0, MAX_RETRY_ATTEMPTS):
                if ANS_FOUND:
                    break
                iteration = 1
                self.messages = self.messages[:2]  # reset history
                INPUT_TOKEN += self.get_token_count(messages=self.messages)
                print(f"RETRY {attempt+1}/{MAX_RETRY_ATTEMPTS}")

                while iteration <= MAX_ITERATIONS_ALLOWED:
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=self.messages,
                            tools=self.tools,
                            tool_choice="auto",
                            temperature=self.temp,
                            max_tokens=self.num_ctx,
                        )
                    except Exception as e:
                        if attempt < MAX_RETRY_ATTEMPTS - 1:
                            print(f"Error communicating with OpenAI: {e}\nRETRYING..")
                            print(f"WAITING FOR 10s")
                            time.sleep(10)
                            continue
                        else:
                            print(
                                f"Error communicating with OpenAI and RETRY LIMIT EXCEEDED."
                            )
                            break

                    print(response)
                    msg = response.choices[0].message
                    print(msg)
                    # Track tokens
                    if response.usage:
                        INPUT_TOKEN += response.usage.prompt_tokens
                        OUTPUT_TOKEN += response.usage.completion_tokens

                    self.messages.append(msg)

                    if not msg.tool_calls:
                        last_message = msg.content if msg.content else "No content"

                        # ─────────────────────────────────────────────────
                        #  BUILD THE FORMATTING PROMPT (once)
                        # ─────────────────────────────────────────────────
                        format_prompt_content = (
                            f"Analyze: {last_message}\n"
                            f"Return ONLY valid JSON string like this json schema "
                            f"inside ```json``` block. JSON_SCHEMA: {json.dumps(format_schema)}.\n"
                        )
                        if self.output_model.__name__ != "outputModelForReturn":
                            format_prompt_content += f"""
                                "**DON'TS**:"
                                - For this function we are only tracking these arguments (1 based index) {argument_number_to_track} **don't report any other arguments**
                                - For answer use argument number and its value like 1:value,2:value (1,2 are the argument numbers asked to resolved in sorted argument number order)
                                - The only possible value for an argument is either an int or UNRESOLVED, don't report long string explaining something like:
                                    output: "Trace:\n1. main → Dio110dDataGet: call site `Dio110dDataGet();` (no mpf_mfs_open)\n2. Dio110dDataGet → Dio110dDtKind:.... final resolved_value:8604", Incorrect
                                - If argument's value is not resolved then report as UNRESOLVED
                                
                                - For call_number we can have both int or NONE
                                - DONT RETURN A LIST.
                            """
                        else:
                            format_prompt_content += (
                                "- **YOU JUST HAVE TO RETURN WHETHER ITS A READ OR WRITE "
                                "OPERATION ON THE RETURN POINTER NOTHING ELSE.**\n"
                            )

                        # ─────────────────────────────────────────────────
                        #  ★ JSON FORMATTING RETRY LOOP
                        # ─────────────────────────────────────────────────
                        MAX_RETRIES = 5
                        format_messages = [
                            {"role": "user", "content": format_prompt_content}
                        ]

                        for fmt_attempt in range(1, MAX_RETRIES + 1):
                            try:
                                print(
                                    f"\n{BOLD}JSON parse attempt {fmt_attempt}/{MAX_RETRIES}{RESET}"
                                )
                                time.sleep(2)

                                fmt_response = self.client.chat.completions.create(
                                    model=self.model,
                                    messages=format_messages,
                                    temperature=self.temp,
                                    max_tokens=1000,
                                )

                                if fmt_response.usage:
                                    INPUT_TOKEN += fmt_response.usage.prompt_tokens
                                    OUTPUT_TOKEN += fmt_response.usage.completion_tokens

                                raw_content = fmt_response.choices[0].message.content
                                print(
                                    f"\n{BOLD}{GREEN}--- RAW RESPONSE ---{RESET}\n{raw_content}"
                                )

                                # ── Step 1: Extract ```json``` block ─────
                                json_match = re.search(
                                    r"```json\s*(.*?)\s*```", raw_content, re.DOTALL
                                )

                                if not json_match:
                                    raise ValueError(
                                        "No ```json``` block found in response"
                                    )

                                json_str = json_match.group(1).strip()

                                # ── Step 2: Validate JSON syntax ─────────
                                try:
                                    json.loads(json_str)
                                except json.JSONDecodeError as je:
                                    raise ValueError(f"Invalid JSON syntax: {je}")

                                # ── Step 3: Validate against Pydantic ────
                                parsed = self.output_model.model_validate_json(json_str)

                                # ── Success! ─────────────────────────────
                                print(
                                    f"{BOLD}{GREEN}JSON VALIDATED on "
                                    f"attempt {fmt_attempt}{RESET}"
                                )
                                final_validated_model = parsed
                                ANS_FOUND = True
                                break

                            except (ValueError, Exception) as e:
                                print(
                                    f"{BOLD}{RED}Attempt {fmt_attempt} failed: "
                                    f"{e}{RESET}"
                                )

                                if fmt_attempt < MAX_RETRIES:
                                    # ── Feed error back so LLM can fix it ─
                                    format_messages.append(
                                        {"role": "assistant", "content": raw_content}
                                    )
                                    format_messages.append(
                                        {
                                            "role": "user",
                                            "content": (
                                                f"Your previous response failed "
                                                f"validation:\n{e}\n\n"
                                                f"Please fix and return ONLY valid "
                                                f"JSON inside a ```json``` block "
                                                f"matching this schema:\n"
                                                f"{json.dumps(format_schema)}"
                                            ),
                                        }
                                    )
                                else:
                                    print(
                                        f"{BOLD}{RED}All {MAX_RETRIES} "
                                        f"attempts exhausted.{RESET}"
                                    )
                                    final_validated_model = None

                        break  # break out of iteration while loop

                    # ─────────────────────────────────────────────────
                    #  TOOL CALLS — execute them
                    # ─────────────────────────────────────────────────
                    print(f"-" * 80)
                    print(f"ITERATION {iteration}/{MAX_ITERATIONS_ALLOWED} (OpenAI)")
                    print(f"-" * 80)
                    print(f"{GREEN}:::::::LLM RESPONSE:::::::{RESET}\n{msg}")

                    for i, tool_call in enumerate(msg.tool_calls):
                        function_name = tool_call.function.name
                        try:
                            arguments = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            arguments = {}

                        arguments = {
                            **arguments,
                            "project_structure": self.project_structure,
                        }

                        tool_result = ""
                        try:
                            if function_name in self.tool_functions:
                                tool_result = self.tool_functions[function_name](
                                    **arguments
                                )
                                print(
                                    f"{BOLD}{RED}TOOL_CALL({i}){RESET}{BOLD}{ORANGE}"
                                    f"Tool call result ({function_name})\n{tool_result}{RESET}"
                                )
                            else:
                                tool_result = "Unknown tool called. Please use only the given tools."
                                random_tool_calls += 1
                        except Exception as e:
                            tool_result = f"Error executing {function_name}: {str(e)}"
                            incorrect_calls.append((tool_result, 0, e))

                        if isinstance(tool_result, tuple):
                            INPUT_TOKEN += self.get_token_count(message=tool_result[0])
                            self.messages.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": tool_result[0],
                                }
                            )
                        else:
                            INPUT_TOKEN += self.get_token_count(message=tool_result)
                            self.messages.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": tool_result,
                                }
                            )

                    iteration += 1
                    if iteration > MAX_ITERATIONS_ALLOWED:
                        print(f"MAX ITERATIONS REACHED, BREAKING.")
                        break
            # endregion
            # final_message = ''
            # final_valida
        stats = {
            "Iterations": iteration,
            "Random_tool_calls": random_tool_calls,
            "Other_tool_errors": len(incorrect_calls),
            "Incorrect_details": incorrect_calls,
        }
        TokenCount = {
            "Input_tokens": INPUT_TOKEN,
            "Output_tokens": OUTPUT_TOKEN,
            "Total_tokens": INPUT_TOKEN + OUTPUT_TOKEN,
        }
        if not final_validated_model:
            final_validated_model = (
                outputModel(output="1:UNRESOLVED")
                if isinstance(self.output_model, outputModel)
                else outputModelForReturn(output="UNRESOLVED")
            )
        return (
            final_validated_model,
            Stats.model_validate({**stats, "Tokens": TokenCount}),
        )

    async def start_new_tool_chain(
        self, prompt_data: dict[str, dict]
    ) -> tuple[outputModel | outputModelForReturn, Stats]:
        """Tool chain targeting the OpenAI-compatible endpoint (zai-org/GLM-4.7-Flash)."""
        import time

        if not prompt_data:
            print("Error no system prompt_data")
            return (None, None)

        iteration = 1
        random_tool_calls = 0
        incorrect_calls = []

        # ── Format the prompts with the supplied data ────────────────────
        for key in prompt_data:
            d = prompt_data.get(key)
            if len(d) == 0:
                continue
            if key == "user_prompt":
                self.user_prompt = self.user_prompt.format(**d)
            else:
                self.system_prompt = self.system_prompt.format(**d)

        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]

        # ── Extract argument names for every registered tool ─────────────
        fun_arg_names = {}
        for tool in self.tool_functions.keys():
            tool_def = self.tool_functions[tool]
            arg_names = OllamaClient.extract_argument_names(function=tool_def)
            fun_arg_names[tool] = arg_names

        format_schema = self.give_json_format()
        final_validated_model = None

        # ═══════════════════════════════════════════════════════════════════
        #  TOOL-CALLING LOOP
        # ═══════════════════════════════════════════════════════════════════
        while True:
            try:
                response = await self.client.chat.completions.create(
                    model=model_name,
                    messages=self.messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=self.temp,
                )
            except Exception as e:
                print(e)
                return f"Error communicating with {model_name}: {e}", {}

            msg = response.choices[0].message
            self.messages.append(msg)

            print(f"-" * 80)
            print(f"ITERATION {iteration} {model_name}")
            print(f"-" * 80)
            print(f"{GREEN}:::::::LLM RESPONSE:::::::{RESET}\n{msg}")

            # ── No more tool calls → format the final answer ─────────────
            if not msg.tool_calls:
                last_content = msg.content or ""

                # ─────────────────────────────────────────────────────────
                #  Try response_format first; fall back to prompt-retry
                # ─────────────────────────────────────────────────────────

                # ─────────────────────────────────────────────────────
                #  FALLBACK: prompt-based retry (same logic as Ollama)
                # ─────────────────────────────────────────────────────
                format_prompt_content = (
                    f"Analyze: {last_content}\n"
                    f"Return ONLY valid JSON string like this json schema "
                    f"inside ```json``` block. JSON_SCHEMA: {json.dumps(format_schema)}.\n"
                )
                if self.output_model.__name__ != "outputModelForReturn":
                    format_prompt_content += (
                        "**DON'TS**:\n"
                        "- For answer use number for answer_no like 1 for first answer, 2 second answer...\n"
                        "- For call_numbers use `call_number` instead of a number like call_number: answer\n"
                        "- For argument_numbers the possible values are INTEGER or UNRESOLVED only\n"
                        "- For call_number we can have both int or NONE\n"
                        "- DONT RETURN A LIST.\n"
                        "Example:\n"
                        "- When we have call_number:\n"
                        "  output = '1:val,2:val2,..,call_number:val3'\n"
                        "- When we don't have call_number:\n"
                        "  output = '1:val,2:val2,..,call_number:None'\n"
                    )
                else:
                    format_prompt_content += (
                        "- **YOU JUST HAVE TO RETURN WHETHER ITS A READ OR WRITE "
                        "OPERATION ON THE RETURN POINTER NOTHING ELSE.**\n"
                    )

                MAX_RETRIES = 5
                format_messages = [
                    {
                        "role": "system",
                        "content": "You are a json formatter according to a give schema and data.",
                    },
                    {"role": "user", "content": format_prompt_content},
                ]

                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        print(
                            f"\n{BOLD}JSON parse attempt {attempt}/{MAX_RETRIES}{RESET}"
                        )
                        time.sleep(1)

                        retry_resp = await self.client.chat.completions.create(
                            model=model_name,
                            messages=format_messages,
                            temperature=self.temp,
                        )

                        raw_content = retry_resp.choices[0].message.content
                        print(
                            f"\n{BOLD}{GREEN}--- RAW RESPONSE ---{RESET}\n{raw_content}"
                        )

                        # Step 1: extract ```json``` block
                        json_match = re.search(
                            r"```json\s*(.*?)\s*```", raw_content, re.DOTALL
                        )
                        if not json_match:
                            raise ValueError("No ```json``` block found in response")

                        json_str = json_match.group(1).strip()

                        # Step 2: syntax check
                        try:
                            json.loads(json_str)
                        except json.JSONDecodeError as je:
                            raise ValueError(f"Invalid JSON syntax: {je}")

                        # Step 3: validate with Pydantic
                        parsed = self.output_model.model_validate_json(json_str)

                        print(
                            f"{BOLD}{GREEN}JSON VALIDATED on attempt {attempt}{RESET}"
                        )
                        final_validated_model = parsed
                        break

                    except (ValueError, Exception) as e:
                        print(f"{BOLD}{RED}Attempt {attempt} failed: {e}{RESET}")
                        if attempt < MAX_RETRIES:
                            format_messages.append(
                                {"role": "assistant", "content": raw_content}
                            )
                            format_messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        f"Your previous response failed validation:\n{e}\n\n"
                                        f"Please fix and return ONLY valid JSON inside a "
                                        f"```json``` block matching this schema:\n"
                                        f"{json.dumps(format_schema)}"
                                    ),
                                }
                            )
                        else:
                            print(
                                f"{BOLD}{RED}All {MAX_RETRIES} attempts exhausted.{RESET}"
                            )
                            final_validated_model = None

                break  # exit the main while-loop

            # ── Execute every tool call the model requested ───────────────
            for i, tool_call in enumerate(msg.tool_calls):
                function_name = tool_call.function.name

                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                arguments = {**arguments, "project_structure": self.project_structure}

                tool_result = ""
                try:
                    if function_name in self.tool_functions:
                        tool_result = self.tool_functions[function_name](**arguments)
                        print(
                            f"{BOLD}{RED}TOOL_CALL({i}){RESET}{BOLD}{ORANGE}"
                            f"Tool call result ({function_name})\n{tool_result}{RESET}"
                        )
                    else:
                        tool_result = (
                            "Unknown tool called. Please use only the given tools."
                        )
                        random_tool_calls += 1
                except Exception as e:
                    tool_result = f"Error executing {function_name}: {str(e)}"
                    incorrect_calls.append((tool_result, 0, e))

                content = (
                    tool_result[0] if isinstance(tool_result, tuple) else tool_result
                )
                self.messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": content,
                    }
                )

            iteration += 1

        # ═══════════════════════════════════════════════════════════════════
        stats = {
            "Iterations": iteration,
            "Random_tool_calls": random_tool_calls,
            "Other_tool_errors": len(incorrect_calls),
            "Incorrect_details": incorrect_calls,
        }
        return (final_validated_model, Stats.model_validate(stats))
