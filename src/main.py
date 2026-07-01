from parse import json_to_data, ParsingPompt, ParsngFunctions
import argparse
from typing import Any


def check_argument() -> tuple[Any, Any, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calls.json")
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    args: argparse.Namespace = parser.parse_args()
    return args.input, args.output, args.functions_definition


def main(input: str, output: str, functions_definition: str) -> None:
    data_prompt: list[dict[str, Any]] = json_to_data(input)
    data_function: list[dict[str, Any]] = json_to_data(functions_definition)
    ParsingPompt.parse_prompts(data_prompt)
    ParsngFunctions.parse_functions(data_function)


if __name__ == "__main__":
    input: str
    output: str
    functions_definition: str
    input, output, functions_definition = check_argument()
    main(input, output, functions_definition)
