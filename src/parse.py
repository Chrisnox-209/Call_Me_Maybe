from pydantic import BaseModel, Field, ValidationError
import json
from typing import Any
from pathlib import Path


def check_output(output: str) -> bool:
    path = Path(output)
    if not path.parent.exists():
        print(f"[ERROR]: The folder '{path.parent}' does not exist.")
        return False
    if not path.parent.is_dir():
        print(f"[ERROR]: '{path.parent}' is not a folder.")
        return False
    return True


def json_to_data(file: str) -> Any:
    try:
        with open(file, "r", encoding="utf-8") as content:
            return json.load(content)
    except FileNotFoundError:
        print(f'[ERROR]: The file "{file}" could not be found.')
        return []
    except json.JSONDecodeError:
        print(f'[ERROR] : The file "{file}" is not valid JSON.')
        return []


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
                print(f"[ERROR]: Validation failed for a prompt: {error}")

        return valid_data


class TypeDef(BaseModel):
    type: str


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
                print(f"[ERROR]: Validation failed for function: {error}")
        return valid_data
