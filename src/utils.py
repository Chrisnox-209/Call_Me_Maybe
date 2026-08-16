from parse import Color, ParsngFunctions
import json
from typing import Any
from enum import Enum


class Color(Enum):
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    ORANGE = "\033[38;5;208m"


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
            "Fallback function. USE THIS FUNCTION if "
            "the user prompt is completely "
            "unrelated, impossible, about colors, or does "
            "not fit any other function."
        ),
        "parameters": {},
        "returns": {
            "type": "none"
        }
    }

    try:
        with open(data_function, "r", encoding="utf-8") as f:
            data: Any = json.load(f)

        if not any(function.get("name") == "fn_none" for function in data):
            data.insert(0, new_function)

        with open(data_function, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    except FileNotFoundError:
        print(f"\n[ERREUR] Le fichier '{data_function}' est introuvable.")
    except json.JSONDecodeError:
        print(f"\n[ERREUR] Le fichier '{data_function}' "
              "contient un JSON invalide.")
    except Exception as e:
        print(f"\n[ERREur] Une erreur inattendue est survenue : {e}")


def output(file_name: str, resultats_finaux: list[Any]) -> None:
    try:
        with open(file_name, "w", encoding="utf-8") as output_file:
            json.dump(resultats_finaux, output_file, indent=4,
                      ensure_ascii=False)
        print("\n[SUCCÈS] Résultats sauvegardés avec succès dans "
              f"'{file_name}'.")
    except Exception as e:
        print(f"\n[ERREUR] Échec de l'écriture du fichier de sortie : {e}")
