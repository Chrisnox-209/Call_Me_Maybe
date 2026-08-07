from llm_sdk import Small_LLM_Model  # type: ignore
import sys

from torch import Tensor


def arguments() -> list[str]:
    search: list[str] = []
    if len(sys.argv) > 1:
        for i in range(len(sys.argv)):
            if i != 0:
                search.append(sys.argv[i])
    return search


def main(arguments: list[str]) -> None:
    llm = Small_LLM_Model()

    for word in arguments:
        try:
            tokens: Tensor = llm.encode(word)
            subword: list[str] = [llm.decode([tok]) for tok
                                  in tokens[0].tolist()]

            print(f"\033[1m\033[32m[{word}]: \n"
                  f"\033[1m\033[35mTOKEN(S) \033[34m--> "
                  f"\033[0m\033[36m{tokens[0].tolist()}\n"
                  f"\033[1m\033[35mSUBWORD\033[34m--> "
                  f"\033[0m\033[36m{subword}\n\n")
        except Exception as error:
            print(f"\033[1m\033[31m[ERROR]:\033[0m Could not tokenize "
                  f"\033[1m{word}\033[0m ({error})")


if __name__ == "__main__":
    main(arguments())
