from llm_sdk import Small_LLM_Model  # type: ignore
from parse import ParsingPompt, ParsngFunctions, Color
from typing import Any, Iterator, Literal
import numpy as np
from numpy.typing import NDArray
import json

   
def build_prompt_func(data_function: list[ParsngFunctions]) -> str:
    prompt_func: str = ''
    for func in data_function:
        prompt_func = prompt_func + f"name: {func.name}\n"
        prompt_func = prompt_func + f"description: {func.description}\n"
        prompt_func = prompt_func + "parameters:\n"
        for name, param in func.parameters.items():
            prompt_func = prompt_func + f"  - {name}: {param.type}\n"
        prompt_func = prompt_func + f"returns: {func.returns.type}\n\n"
    return prompt_func


def build_struct() -> str:
    return """exemple:
    [
        {
            "prompt": "What is the sum of 2 and 3?",
            "name": "fn_add_numbers",
            "parameters": {
                "a": 2.0,
                "b": 3.0
            }
        },
        {
            "prompt": "Reverse the string 'hello'",
            "name": "fn_reverse_string",
            "parameters": {
                "s": "hello"
            }
        }
    ]"""


def charge_vocab(llm) -> Any:
    file_vocab: str = llm.get_path_to_vocab_file()

    try:
        with open(file_vocab, "r", encoding="utf-8") as content:
            return json.load(content)
    except FileNotFoundError:
        raise ValueError(f'The file "{Color.YELLOW.value}{file_vocab}'
                         f'{Color.RST.value}" could not be found.')
    except json.JSONDecodeError:
        raise ValueError(f'The file "{Color.YELLOW.value}{file_vocab}'
                         f'"{Color.RST.value}" is not valid JSON.')


def run_inference(data_prompt: list, data_function: list, output: str) -> None:

    llm: Any = Small_LLM_Model()
    vocab: Any = charge_vocab(llm)

    print(data_prompt[0].prompt)

    function: list[int] = llm.encode(build_prompt_func(
        data_function))[0].tolist()

    exemple: list[int] = llm.encode(build_struct())[0].tolist()

    for line in data_prompt:
        prompt: str = llm.encode(line.prompt)[0].tolist()
        send_prompt: list[int] = function + prompt + exemple

        while True:
            logits: NDArray[Any] = np.array(llm.get_logits_from_input_ids(
                send_prompt))
            logits_origin = logits
            logits[:] = -float("inf")
            logits[58] = logits_origin[58]
            next_token = int(np.argmax(logits))
            send_prompt.append(next_token)

            result: str = llm.decode([next_token])
            print(result, end="", flush=True)
            
            if send_prompt[-1] == 60:
                break
            