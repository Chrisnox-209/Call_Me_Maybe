import json
import sys
from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Any


def arguments() -> list[str]:
    search: list[str] = []
    if len(sys.argv) > 1:
        for i in range(len(sys.argv)):
            if i != 0:
                search.append(sys.argv[i])
    return search


def main(arguments: list[str]) -> None:
    llm = Small_LLM_Model()
    path: str = llm.get_path_to_vocab_file()

    with open(path, 'r', encoding="utf8") as file:
        vocabulaire: Any = json.load(file)

    for char in arguments:
        if char not in vocabulaire:
            print(f"\033[1m\033[31m[ERROR]:\033[0m \033[1m{char} "
                  f"\033[0m\033[90m is not in vocab.json\033[0m")
        else:
            token: int = vocabulaire[char]
            print(f"\033[1m\033[32m[TOKEN]:\033[0m \033[36m{char} "
                  f"\033[0m\033[1m--> \033[0m\033[36m{token}\033[0m")


if __name__ == "__main__":
    main(arguments())
