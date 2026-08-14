from llm_sdk import Small_LLM_Model  # type: ignore
from parse import ParsingPompt, ParsngFunctions
from typing import Any, Iterator, Literal
import numpy as np
from numpy.typing import NDArray
from utils import charge_vocab, reverse_vocab, lst_name_fonction


def build_prompt_func(data_function: list[ParsngFunctions]) -> str:
    prompt_func: str = ''
    for func in data_function:
        prompt_func = prompt_func + f"name: {func.name}\n"
        prompt_func = prompt_func + f"description: {func.description}\n"
        prompt_func = prompt_func + "parameters:\n"
        for name, param in func.parameters.items():
            prompt_func = prompt_func + f"  - {name}: {param.type}\n"
        prompt_func = prompt_func + f"returns: {func.returns.type}\n\n"
    return prompt_func


def step_name(logits: NDArray[Any], logits_origin: NDArray[Any], vocab: dict,
              name_fonc: list[str], llm: Any, send_prompt: list[int]) -> None:
    logits[:] = -float("inf")
    characters_authorized: str = ("abcdefghijklmnopqrstuvwxyz"
                                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    ids_authorized: list[int] = []

    current_text: str = llm.decode(send_prompt)
    current_name: str = current_text.split('"name": "')[-1]

    for token_id, token_text in vocab.items():
        if token_text not in characters_authorized and token_text != '"':
            continue

        candidat: str = current_name + token_text
        valide = False

        for name in name_fonc:
            if name.startswith(candidat):
                valide = True
                break

            if name == current_name and token_text == '"':
                valide = True
                break

        if valide:
            token_id_int = int(token_id)
            ids_authorized.append(token_id_int)
            logits[token_id_int] = logits_origin[token_id_int]


def step_parameters(logits: NDArray[Any], logits_origin: NDArray[Any],
                    vocab: dict) -> None:
    logits[:] = -float("inf")
    caracteres_autorises = ('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNO'
                            'PQRSTUVWXYZ0123456789_"{},. :[]\n\t-')

    for token_id, token_text in vocab.items():
        valide = True
        for char in token_text:
            if char not in caracteres_autorises:
                valide = False
                break

        if valide:
            token_id_int = int(token_id)
            logits[token_id_int] = logits_origin[token_id_int]


def run_inference(data_prompt: list, data_function: list, output: str) -> None:
    llm: Any = Small_LLM_Model()
    vocab: dict = reverse_vocab(charge_vocab(llm))

    function: list[int] = llm.encode(build_prompt_func(
        data_function))[0].tolist()

    for line in data_prompt:
        starter: str = f'[\n{{\n  "prompt": "{line.prompt}",\n  "name": "'
        send_prompt: Any = function + llm.encode(starter)[0].tolist()
        state = 1

        while True:
            logits: NDArray[Any] = np.array(
                llm.get_logits_from_input_ids(send_prompt))
            logits_origin: NDArray[Any] = logits.copy()

            if state == 1:
                name_fonc: list[str] = lst_name_fonction(data_function)
                step_name(logits, logits_origin, vocab, name_fonc, llm, send_prompt)

            elif state == 2:
                step_parameters(logits, logits_origin, vocab)

            next_token = int(np.argmax(logits))
            send_prompt.append(next_token)

            result: str = llm.decode([next_token])
            print(result, end="", flush=True)

            if state == 1 and result == '"':
                state = 2
            if state == 1 and result == '}':
                state = 3


            if send_prompt[-1] == 60:
                break
    