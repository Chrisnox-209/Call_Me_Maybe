from pydantic import BaseModel, Field, ValidationError
from enum import Enum
import json
import sys
from typing import Any, Literal
from pathlib import Path
import argparse

from pydantic_core import ErrorDetails


class Color(Enum):
    BLUE = "\033[34m"
    ORANGE = "\033[38;5;208m"
    RED = "\033[31m"
    WHITE = "\033[37m"
    YELLOW = "\033[33m"
    RST = "\033[0m"


class OutputPathError(Exception):
    pass


def create_Folder() -> bool:
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
    path = Path(output)

    if not path.parent.exists():
        print(f"{Color.ORANGE.value}[WARNING]: {Color.RST.value}"
              f"The folder {Color.YELLOW.value}{path.parent}"
              f"{Color.RST.value} does not exist.")

        if create_Folder():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"{Color.BLUE.value}[INFO]: {Color.RST.value}folder create")
        else:
            raise OutputPathError("User refused to create folder: "
                                  f"{Color.YELLOW.value}{path.parent}"
                                  f"{Color.RST.value}")

    if not path.parent.is_dir():
        raise OutputPathError(f"Not a directory: {path.parent}")

    return True


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


def json_to_data(file: str) -> Any:
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
    type: Literal["number", "string", "boolean", "object"]


class ParsingPompt(BaseModel):
    prompt: str

    @classmethod
    def parse_prompts(cls, data: list[dict[str, Any]]) -> list['ParsingPompt']:
        valid_data: list['ParsingPompt'] = []

        for item in data:
            try:
                prompt: Any = cls(**item)
                valid_data.append(prompt)
            except ValidationError as error:
                err: ErrorDetails = error.errors()[0]
                raise ValueError(f"Invalid prompt: {err['loc'][0]}: "
                                 f"{err['msg']}")

        return valid_data


class ParsngFunctions(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    description: str = Field(min_length=1, max_length=255)
    parameters: dict[str, TypeDef]
    returns: TypeDef

    @classmethod
    def parse_functions(cls, data: list[dict[str, Any]]) -> list[
            'ParsngFunctions']:
        valid_data: list['ParsngFunctions'] = []

        for item in data:
            try:
                result: Any = cls(**item)
                valid_data.append(result)
            except ValidationError as error:
                err: ErrorDetails = error.errors()[0]
                raise ValueError(f"{err['loc'][0]}.{err['loc'][1]}: "
                                 f"{err['msg']}")
        return valid_data
