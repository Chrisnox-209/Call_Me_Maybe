from parse import Color
import json
from typing import Any


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


def reverse_vocab(file_vocab: dict) -> dict:
    reverse_vocab: dict = {}
    for keys, values in file_vocab.items():
        reverse_vocab[values] = keys
    return reverse_vocab


def lst_name_fonction(data_function) -> list[str]:
    name: list[str] = []
    for function in data_function:
        name.append(function.name)
    return name
