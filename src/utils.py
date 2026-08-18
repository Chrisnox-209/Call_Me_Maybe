"""Utility functions for file handling and data manipulation."""

from .parse import Color, ParsngFunctions
import json
import sys
from typing import Any


def charge_vocab(llm: Any) -> Any:
    """
    Load the vocabulary file from the given LLM model.

    Args:
        llm: The large language model instance.

    Returns:
        A dictionary representing the loaded vocabulary.

    Raises:
        ValueError: If the file is not found or is invalid JSON.
    """
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
    """
    Reverse the mapping of a vocabulary dictionary.

    Args:
        file_vocab: Dictionary mapping string tokens to integer IDs.

    Returns:
        Dictionary mapping integer IDs to string tokens.
    """
    reverse_vocab: dict[int, str] = {}
    for keys, values in file_vocab.items():
        reverse_vocab[values] = keys
    return reverse_vocab


def lst_name_fonction(
    data_function: list[ParsngFunctions],
) -> list[str]:
    """
    Extract the names of all parsing functions.

    Args:
        data_function: List of ParsngFunctions objects.

    Returns:
        A list of function name strings.
    """
    func_names: list[str] = []

    for function in data_function:
        func_names.append(function.name)
    return func_names


def select_model_interactive(model: str) -> str:
    """
    Prompt the user interactively to select an LLM model.

    Args:
        model: The default model to fall back on or modify.

    Returns:
        The string name of the selected model.
    """
    print("\nAvailable models:")
    print("1) Qwen/Qwen3-0.6B")
    print("2) Qwen/Qwen3-1.7B")
    print("3) Qwen/Qwen3-0.6B-Base")
    choice = input("Select a model (1-3): ")
    if choice == "1":
        model = "Qwen/Qwen3-0.6B"
    elif choice == "2":
        model = "Qwen/Qwen3-1.7B"
    elif choice == "3":
        model = "Qwen/Qwen3-0.6B-Base"
    else:
        print(f"{Color.RED.value}[ERROR]{Color.RST.value} Invalid choice.")
        sys.exit(1)
    return model


def add_none(data_function: str) -> None:
    """
    Ensure the fallback function 'fn_none' is present in the functions file.

    Reads the given JSON file containing function definitions, and if
    'fn_none' is missing, it is inserted at the beginning.

    Args:
        data_function: Path to the JSON file with function definitions.
    """
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
        print(f"\n{Color.RED.value}[ERROR]{Color.RST.value} The file "
              f"'{Color.YELLOW.value}{data_function}{Color.RST.value}' "
              "could not be found.")
    except json.JSONDecodeError:
        print(f"\n{Color.RED.value}[ERROR]{Color.RST.value} The file "
              f"'{Color.YELLOW.value}{data_function}{Color.RST.value}' "
              "contains invalid JSON.")
    except Exception as e:
        print(f"\n{Color.RED.value}[ERROR]{Color.RST.value} An unexpected "
              f"error occurred: {e}")


def output(file_name: str, resultats_finaux: list[Any]) -> None:
    """
    Write the final results to a JSON output file.

    Args:
        file_name: Path to the output JSON file.
        resultats_finaux: List of result dictionaries to save.
    """
    try:
        with open(file_name, "w", encoding="utf-8") as output_file:
            json.dump(resultats_finaux, output_file, indent=4,
                      ensure_ascii=False)
        print(f"\n{Color.GREEN.value}[SUCCESS]{Color.RST.value} "
              "Results successfully saved to "
              f"'{Color.BLUE.value}{file_name}{Color.RST.value}'.\n")
    except Exception as e:
        print(f"\n{Color.RED.value}[ERROR]{Color.RST.value} "
              f"Failed to write output file: {e}\n")
