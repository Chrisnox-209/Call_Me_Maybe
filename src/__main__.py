"""Main entry point for the Call_Me_Maybe application."""

from .parse import (
    json_to_data, ParsingPompt, ParsngFunctions, check_output,
    check_argument, OutputPathError, Color
)

from typing import Any
from .inference import run_inference
from .utils import add_none, select_model_interactive
import sys


def main(input: str, output: str, functions_definition: str,
         model: str) -> None:
    """
    Execute the main application flow.

    Args:
        input: Path to the input JSON file containing prompts.
        output: Path to the output JSON file.
        functions_definition: Path to the JSON file defining functions.
        model: The LLM model name to use.
    """
    try:
        check_output(output)
    except OutputPathError as e:
        print(f"{Color.RED.value}[ERROR]{Color.RST.value} {e}")
        sys.exit(1)

    try:
        add_none(functions_definition)
    except ValueError as e:
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
    run_inference(parse_prompt, parse_function, output, model)


if __name__ == "__main__":
    input_file: str
    output_file: str
    functions_definition: str
    model: str
    multi: bool

    (input_file, output_file, functions_definition,
     model, multi) = check_argument()

    if multi:
        model = select_model_interactive(model)

    main(input_file, output_file, functions_definition, model)
