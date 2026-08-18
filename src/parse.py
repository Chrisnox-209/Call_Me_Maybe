"""Parsing utilities and Pydantic models for the application."""

from pydantic import BaseModel, Field, ValidationError
from enum import Enum
import json
from typing import Any, Literal
from pathlib import Path
import argparse

from pydantic_core import ErrorDetails


class Color(Enum):
    """
    Enum representing ANSI color escape codes for terminal formatting.
    """
    BLUE = "\033[34m"
    ORANGE = "\033[38;5;208m"
    RED = "\033[31m"
    WHITE = "\033[37m"
    YELLOW = "\033[33m"
    GREEN = "\033[92m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RST = "\033[0m"


class OutputPathError(Exception):
    """
    Exception raised for errors related to output file paths.
    """
    pass


def create_Folder() -> bool:
    """
    Prompt the user for permission to create a missing folder.

    Returns:
        True if the user accepts, False otherwise.
    """
    while True:
        reponse: str = input("Create the folder ? (Y/n) ").lower()

        if reponse in ("", "y"):
            return True

        elif reponse == "n":
            return False

        else:
            print(f"{Color.ORANGE.value}[WARNING]: {Color.RST.value}"
                  f"invalid answer, only '{Color.ORANGE.value}y"
                  f"{Color.RST.value}' or '{Color.ORANGE.value}n"
                  f"{Color.RST.value}'.")


def check_output(output: str) -> bool:
    """
    Verify the output directory exists, prompting to create it if missing.

    Args:
        output: The target output file path.

    Returns:
        True if the directory exists or was created successfully.

    Raises:
        OutputPathError: If user refuses to create it or not a dir.
    """
    path = Path(output)

    if not path.parent.exists():
        print(f"{Color.ORANGE.value}[WARNING]: {Color.RST.value}"
              f"The folder {Color.YELLOW.value}{path.parent}"
              f"{Color.RST.value} does not exist.")

        if create_Folder():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"{Color.BLUE.value}[INFO]: {Color.RST.value}folder created")
        else:
            raise OutputPathError("User refused to create folder: "
                                  f"{Color.YELLOW.value}{path.parent}"
                                  f"{Color.RST.value}")

    if not path.parent.is_dir():
        raise OutputPathError(f"Not a directory: {path.parent}")

    return True


def check_argument() -> tuple[Any, Any, Any, str, bool]:
    """
    Parse command line arguments for the application.

    Returns:
        Tuple: (input, output, functions, model_name, is_multi).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json"
    )
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        choices=[
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-1.7B",
            "Qwen/Qwen3-0.6B-Base"
        ]
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Interactively select the model"
    )
    args: argparse.Namespace = parser.parse_args()
    return (args.input, args.output, args.functions_definition,
            args.model, args.multi)


def json_to_data(file: str) -> Any:
    """
    Load JSON data from a given file path.

    Args:
        file: The path to the JSON file.

    Returns:
        The parsed JSON content as a Python object.

    Raises:
        ValueError: If the file is not found or is invalid JSON.
    """
    try:
        with open(file, "r", encoding="utf-8") as content:
            return json.load(content)
    except FileNotFoundError:
        raise ValueError(f'The file "{Color.YELLOW.value}{file}'
                         f'{Color.RST.value}" could not be found.')
    except json.JSONDecodeError:
        raise ValueError(f'The file "{Color.YELLOW.value}{file}'
                         f'"{Color.RST.value}" is not valid JSON.')


class TypeDef(BaseModel):
    """
    Pydantic model representing a supported parameter or return type.
    """
    type: Literal["number", "string", "boolean", "integer", "float",
                  "object", "none"]


class ParsingPompt(BaseModel):
    """
    Pydantic model representing a user prompt entry.
    """
    prompt: str

    @classmethod
    def parse_prompts(cls, data: list[dict[str, Any]]) -> list['ParsingPompt']:
        """
        Parse a list of dictionaries into ParsingPompt objects.

        Args:
            data: List of raw prompt dictionaries.

        Returns:
            List of validated ParsingPompt instances.

        Raises:
            ValueError: If validation fails.
        """
        valid_data: list['ParsingPompt'] = []

        for item in data:
            try:
                prompt: Any = cls(**item)
                valid_data.append(prompt)
            except ValidationError as error:
                err: ErrorDetails = error.errors()[0]
                loc_str = ".".join(str(x) for x in err['loc'])
                raise ValueError(f"Invalid prompt: {loc_str}: "
                                 f"{err['msg']}")

        return valid_data


class ParsngFunctions(BaseModel):
    """
    Pydantic model representing a function definition.
    """
    name: str = Field(min_length=1, max_length=50)
    description: str = Field(min_length=1, max_length=255)
    parameters: dict[str, TypeDef]
    returns: TypeDef

    @classmethod
    def parse_functions(cls, data: list[dict[str, Any]]) -> list[
            'ParsngFunctions']:
        """
        Parse a list of dictionaries into ParsngFunctions objects.

        Args:
            data: List of raw function definition dictionaries.

        Returns:
            List of validated ParsngFunctions instances.

        Raises:
            ValueError: If validation fails.
        """
        valid_data: list['ParsngFunctions'] = []

        for item in data:
            try:
                result: Any = cls(**item)
                valid_data.append(result)
            except ValidationError as error:
                err: ErrorDetails = error.errors()[0]
                loc_str = ".".join(str(x) for x in err['loc'])
                raise ValueError(f"Invalid function definition: {loc_str}: "
                                 f"{err['msg']}")
        return valid_data
