"""Vocabulary management and tokenization testing script."""

from llm_sdk import Small_LLM_Model  # type: ignore
import os
import json
from typing import Any
from torch import Tensor


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
        raise ValueError(
            f'The file "{file_vocab}" could not be found.'
        )
    except json.JSONDecodeError:
        raise ValueError(
            f"The file {file_vocab} is not valid JSON."
        )


def main() -> None:
    """
    Run the vocabulary test based on the ARGS environment variable.
    """
    llm: Any = Small_LLM_Model()
    vocab: Any = charge_vocab(llm)

    word: str = os.environ["ARGS"]

    try:
        tokens: Tensor = llm.encode(word)
        subword: list[str] = [
            llm.decode([tok])
            for tok in tokens[0].tolist()
        ]

        id_lst: list[int] = []

        for text in vocab.keys():
            for w in subword:
                if w == text:
                    id_lst.append(vocab[text])

        print(
            f"\033[1m\033[32m[{word}]: \n"
            f"\033[1m\033[35mTOKEN(S) \033[34m--> "
            f"\033[0m\033[36m{tokens[0].tolist()}\n"
            f"\033[1m\033[35mSUBWORD\033[34m--> "
            f"\033[0m\033[36m{subword}\n"
            f"\033[1m\033[35mID_VOCAB\033[34m--> "
            f"\033[0m\033[36m{id_lst}\n\n"
        )

    except Exception as error:
        print(
            f"\033[1m\033[31m[ERROR]:\033[0m Could not tokenize "
            f"\033[1m{word}\033[0m ({error})"
        )


if __name__ == "__main__":
    main()
