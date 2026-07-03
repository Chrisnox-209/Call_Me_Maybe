from parse import (json_to_data, ParsingPompt,
                   ParsngFunctions, check_output,
                   check_argument, OutputPathError,
                   Color)

from typing import Any
from inference import run_inference
import sys


def main(input: str, output: str, functions_definition: str) -> None:
    try:
        check_output(output)
    except OutputPathError as e:
        print(f"{Color.RED.value}[ERROR]{Color.RST.value} {e}")
        sys.exit(1)

    try:
        data_prompt: list[dict[str, Any]] = (
            json_to_data(input))
        data_function: list[dict[str, Any]] = (
            json_to_data(functions_definition))
    except ValueError as e:
        print(f"{Color.RED.value}[ERROR]{Color.RST.value} {e}")
        sys.exit(1)

    try:
        parse_prompt: list[ParsingPompt] = (
            ParsingPompt.parse_prompts(data_prompt))
        parse_function: list[ParsngFunctions] = (
            ParsngFunctions.parse_functions(data_function))

    except ValueError as e:
        print(f"{Color.RED.value}[ERROR]{Color.RST.value} {e}")
        sys.exit(1)
    run_inference(parse_prompt, parse_function, output)


if __name__ == "__main__":
    input: str
    output: str
    functions_definition: str

    input, output, functions_definition = check_argument()
    main(input, output, functions_definition)
