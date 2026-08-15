from parse import Color, ParsngFunctions
import json
from typing import Any


def charge_vocab(llm: Any) -> Any:
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


def reverse_vocab(file_vocab: dict[str, int]) -> dict[int, str]:
    reverse_vocab: dict[int, str] = {}
    for keys, values in file_vocab.items():
        reverse_vocab[values] = keys
    return reverse_vocab


def lst_name_fonction(
    data_function: list[ParsngFunctions],
) -> list[str]:
    name: list[str] = []

    for function in data_function:
        name.append(function.name)
    return name


def add_none(data_function: str) -> None:
    new_function: dict[str, Any] = {
        "name": "fn_none",
        "description": (
            "Fallback function. USE THIS FUNCTION if the user prompt is completely "
            "unrelated, impossible, about colors, or does not fit any other function."
        ),
        "parameters": {},
        "returns": {
            "type": "none"
        }
    }

    with open(data_function, "r", encoding="utf-8") as f:
        data: Any = json.load(f)

    if not any(function.get("name") == "fn_none" for function in data):
        data.insert(0, new_function)

    with open(data_function, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
