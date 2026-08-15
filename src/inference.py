from llm_sdk import Small_LLM_Model  # type: ignore
from parse import ParsngFunctions, ParsingPompt
from typing import Any
import numpy as np
from numpy.typing import NDArray
from utils import charge_vocab, reverse_vocab, lst_name_fonction


def build_prompt_func(data_function: list[ParsngFunctions]) -> str:
    prompt_func = (
        "You are an expert AI API router. You must map the user's task "
        "to the correct function.\n\n"
        "AVAILABLE FUNCTIONS:\n"
    )

    for func in data_function:
        prompt_func += f"- name: {func.name}\n"
        prompt_func += f"  description: {func.description}\n"
        prompt_func += "  parameters:\n"
        for name, param in func.parameters.items():
            prompt_func += f"    - {name}: {param.type}\n"
        prompt_func += f"  returns: {func.returns.type}\n\n"

    prompt_func += "- name: fn_none\n"
    prompt_func += "  description: USE THIS FUNCTION if the task is about "
    "colors, completely unrelated, or impossible.\n"
    prompt_func += "  parameters:\n"
    prompt_func += "  returns: none\n\n"

    prompt_func += (
        "EXAMPLE:\n"
        "Task: What is the weather like in Paris?\n"
        "JSON:\n"
        "{\n"
        "  \"name\": \"fn_get_weather\",\n"
        "  \"parameters\": {\n"
        "    \"city\": \"Paris\"\n"
        "  }\n"
        "}\n\n"
        "Now process the following task.\n"
    )
    return prompt_func


def step_name(logits: NDArray[Any], logits_origin: NDArray[Any],
              vocab: dict[int, str], name_fonc: list[str], llm: Any,
              send_prompt: list[int]) -> None:
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


def get_authorized_chars(type_attendu: str, valeur_terminee: bool) -> str:
    if valeur_terminee or type_attendu is None:
        return ('abcdefghijklmnopqrstuvwxyz'
                'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"{},. :\n\t-')

    if type_attendu == "number":
        return '0123456789.-, \n}'

    elif type_attendu == "string":
        return ('abcdefghijklmnopqrstuvwxyz'
                'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ -.,!?\'"\n}')

    return ('abcdefghijklmnopqrstuvwxyz'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"{},. :\n\t-')


def step_parameters(logits: NDArray[Any], logits_origin: NDArray[Any],
                    vocab: dict[int, str], llm: Any, send_prompt: list[int],
                    chosen_function: Any) -> None:
    logits[:] = -float("inf")
    texte_actuel: Any = llm.decode(send_prompt)

    derniere_cle = None
    dernier_index: Any = -1
    type_attendu: Any = None
    valeur_terminee: bool = True

    if chosen_function and hasattr(chosen_function, 'parameters'):
        for key in chosen_function.parameters.keys():
            idx: Any = texte_actuel.rfind(f'"{key}"')
            if idx > dernier_index:
                dernier_index = idx
                derniere_cle = key

    if derniere_cle is not None:
        texte_depuis_cle: Any = texte_actuel[dernier_index:]
        idx_colon: Any = texte_depuis_cle.find(":")

        if idx_colon != -1:
            texte_apres_colon: Any = texte_depuis_cle[idx_colon + 1:]
            type_attendu = chosen_function.parameters[derniere_cle].type
            valeur_terminee = False

            if type_attendu == "number":
                if ("," in texte_apres_colon or "}"
                   in texte_apres_colon or "\n"
                   in texte_apres_colon):
                    valeur_terminee = True
            elif type_attendu == "string":
                if texte_apres_colon.count('"') >= 2:
                    valeur_terminee = True

    caracteres_autorises: str = get_authorized_chars(type_attendu,
                                                     valeur_terminee)

    for token_id, token_text in vocab.items():
        clean_text: Any = token_text.replace(
            ' ', '').replace('Ġ', '').replace('Ċ', '\n').replace('<0x00>', '')

        if not clean_text:
            logits[int(token_id)] = logits_origin[int(token_id)]
            continue

        valide = True
        for char in clean_text:
            if char not in caracteres_autorises:
                valide = False
                break

        if valide:
            logits[int(token_id)] = logits_origin[int(token_id)]


def run_inference(
    data_prompt: list[ParsingPompt],
    data_function: list[ParsngFunctions],
    output: str,
) -> None:
    llm: Any = Small_LLM_Model()
    vocab: dict[int, str] = reverse_vocab(charge_vocab(llm))

    function: list[ParsngFunctions] = llm.encode(build_prompt_func(
        data_function))[0].tolist()

    # for line in data_prompt:
    #     starter: str = f'[\n{{\n  "prompt": "{line.prompt}",\n  "name": "'
    #     send_prompt: Any = function + llm.encode(starter)[0].tolist()

    for line in data_prompt:
        starter: str = f'Task: {line.prompt}\nJSON:\n{{\n  "name": "'
        send_prompt: Any = function + llm.encode(starter)[0].tolist()

        state = 1
        chosen_func_obj = None

        print(f"---> {line.prompt}")

        while True:
            logits: NDArray[Any] = np.array(
                llm.get_logits_from_input_ids(send_prompt))
            logits_origin: NDArray[Any] = logits.copy()

            if state == 1:
                name_fonc: list[str] = lst_name_fonction(data_function)
                step_name(logits, logits_origin, vocab,
                          name_fonc, llm, send_prompt)

            elif state == 2:
                step_parameters(logits, logits_origin, vocab,
                                llm, send_prompt, chosen_func_obj)

            next_token = int(np.argmax(logits))
            send_prompt.append(next_token)

            result: str = llm.decode([next_token])
            print(result, end="", flush=True)

            if state == 1 and result == '"':
                texte_complet: Any = llm.decode(send_prompt)
                nom_genere: Any = texte_complet.split(
                    '"name": "')[-1].replace('"', '')

                if nom_genere == "fn_none":
                    print("\n\n[INFO] Le modèle a déterminé "
                          "qu'aucune fonction ne correspond.")
                    break

                state = 2

                for f in data_function:
                    if f.name == nom_genere:
                        chosen_func_obj = f
                        break

                if chosen_func_obj is None:
                    print(f"\n\n[ERREUR] La fonction '{nom_genere}' n'a pas "
                          "été trouvée dans la liste !")
                    break

                print(f"\n[INFO] Fonction détectée : {chosen_func_obj.name}.")

                aide_json = ',\n  "parameters": {\n    "'
                aide_tokens: Any = llm.encode(aide_json)[0].tolist()
                send_prompt.extend(aide_tokens)
                print(aide_json, end="", flush=True)

            if state == 2:
                texte_actuel: Any = llm.decode(send_prompt)
                nb_ouvertes: Any = texte_actuel.count('{')
                nb_fermees: Any = texte_actuel.count('}')

                if nb_ouvertes > 0 and nb_ouvertes == nb_fermees:
                    print("\n\n-----------------\n\n")
                    break
