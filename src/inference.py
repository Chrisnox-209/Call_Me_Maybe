from llm_sdk import Small_LLM_Model  # type: ignore
from parse import ParsngFunctions, ParsingPompt
from typing import Any
import numpy as np
from numpy.typing import NDArray
import json
from utils import charge_vocab, reverse_vocab, lst_name_fonction, output


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

    prompt_func += (
        "EXAMPLES:\n"
        "Task: What is the weather like in Paris?\n"
        "JSON:\n"
        "{\n"
        "  \"name\": \"fn_get_weather\",\n"
        "  \"parameters\": {\n"
        "    \"city\": \"Paris\"\n"
        "  }\n"
        "}\n\n"
        "Task: change blue in white\n"
        "JSON:\n"
        "{\n"
        "  \"name\": \"fn_none\"\n"
        "}\n\n"
        "Task: Bake me a chocolate cake\n"
        "JSON:\n"
        "{\n"
        "  \"name\": \"fn_none\"\n"
        "}\n\n"
        "Now process the following task.\n"
    )
    return prompt_func


def step_name(logits: NDArray[Any], logits_origin: NDArray[Any],
              vocab: dict[int, str], name_fonc: list[str], llm: Any,
              send_prompt: list[int]) -> None:
    logits[:] = -float("inf")

    characters_authorized: str = ("abcdefghijklmnopqrstuvwxyz"
                                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_\"")

    current_text: str = llm.decode(send_prompt)
    current_name: str = current_text.split('"name": "')[-1]

    for token_id, token_text in vocab.items():

        clean_text: str = token_text.replace(
            ' ', '').replace('Ġ', '').replace('Ċ', '').replace('<0x00>', '')

        if not clean_text:
            continue

        valide_char = True
        for char in clean_text:
            if char not in characters_authorized:
                valide_char = False
                break

        if not valide_char:
            continue

        candidat: str = current_name + clean_text
        valide_candidat = False

        for name in name_fonc:

            if name.startswith(candidat):
                valide_candidat = True
                break

            if name == current_name and clean_text == '"':
                valide_candidat = True
                break

            if candidat == name + '"':
                valide_candidat = True
                break

        if valide_candidat:
            logits[int(token_id)] = logits_origin[int(token_id)]


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
    output_filename: str,
) -> None:
    llm: Any = Small_LLM_Model()
    vocab: dict[int, str] = reverse_vocab(charge_vocab(llm))

    function: list[int] = llm.encode(build_prompt_func(
        data_function))[0].tolist()

    resultats_finaux: list[dict[str, str]] = []

    for line in data_prompt:
        starter: str = f'Task: {line.prompt}\nJSON:\n{{\n  "name": "'
        send_prompt: Any = function + llm.encode(starter)[0].tolist()

        state = 1
        chosen_func_obj: ParsngFunctions | None = None

        token_count = 0
        max_tokens = 150

        print(f"---> {line.prompt}")

        while True:
            token_count += 1
            if token_count > max_tokens:
                print("\n\n[ERREUR] Boucle infinie. Arrêt forcé !")
                resultats_finaux.append({"name": "fn_none"})
                print("-----------------\n")
                break

            logits: NDArray[Any] = np.array(llm.get_logits_from_input_ids(
                send_prompt))
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

            if state == 1:
                texte_complet: Any = llm.decode(send_prompt)
                nom_en_cours: str = texte_complet.split('"name": "')[-1]

                if '"' in nom_en_cours:
                    nom_genere: str = nom_en_cours.replace('"', '').strip()

                    if nom_genere == "fn_none":
                        print(
                            "\n[INFO] Le modèle a déterminé qu'aucune "
                            "fonction ne correspond.")
                        aide_json = '\n}'
                        aide_tokens: Any = llm.encode(aide_json)[0].tolist()
                        send_prompt.extend(aide_tokens)
                        print(aide_json, end="", flush=True)

                        json_str = "{\n  \"name\": \"fn_none\"\n}"
                        resultats_finaux.append(json.loads(json_str))
                        print("\n\n-----------------\n\n")
                        break

                    state = 2

                    for func in data_function:
                        if func.name == nom_genere:
                            chosen_func_obj = func
                            break

                    if chosen_func_obj is None:
                        print(
                            f"\n\n[ERREUR] La fonction '{nom_genere}' n'a "
                            "pas été trouvée !")
                        break

                    print(
                        f"\n[INFO] Fonction détectée : {
                            chosen_func_obj.name}.")

                    aide_json = ',\n  "parameters": {\n    '
                    aide_tokens = llm.encode(aide_json)[0].tolist()
                    send_prompt.extend(aide_tokens)
                    print(aide_json, end="", flush=True)

                    if chosen_func_obj.parameters:
                        premiere_cle = list(
                            chosen_func_obj.parameters.keys())[0]
                        aide_cle = f'"{premiere_cle}": '
                        aide_cle_tokens = llm.encode(aide_cle)[0].tolist()
                        send_prompt.extend(aide_cle_tokens)
                        print(aide_cle, end="", flush=True)

            elif state == 2:
                texte_actuel: Any = llm.decode(send_prompt)
                nb_ouvertes: Any = texte_actuel.count('{')
                nb_fermees: Any = texte_actuel.count('}')

                if nb_ouvertes > 0 and nb_ouvertes == nb_fermees:
                    json_str = "{\n" + texte_actuel.split("JSON:\n{")[-1]

                    import re
                    json_str = re.sub(r':\s*r"', ': "', json_str)

                    try:
                        objet_json = json.loads(json_str)
                        resultats_finaux.append(objet_json)
                    except json.JSONDecodeError as e:
                        print("\n[ERREUR] Impossible de parser le "
                              f"JSON généré : {e}")
                        print(f"JSON fautif :\n{json_str}")

                    print("\n\n-----------------\n\n")
                    break

    output(output_filename, resultats_finaux)
