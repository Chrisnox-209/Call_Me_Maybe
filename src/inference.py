from typing import Any
import json
import numpy as np
from numpy.typing import NDArray
from llm_sdk import Small_LLM_Model  # type: ignore
from parse import ParsngFunctions, ParsingPompt
from utils import charge_vocab, reverse_vocab, lst_name_fonction, output, Color


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
        '  "name": "fn_get_weather",\n'
        '  "parameters": {\n'
        '    "city": "Paris"\n'
        "  }\n"
        "}\n\n"
        "Task: change blue in white\n"
        "JSON:\n"
        "{\n"
        '  "name": "fn_none"\n'
        "}\n\n"
        "Task: Bake me a chocolate cake\n"
        "JSON:\n"
        "{\n"
        '  "name": "fn_none"\n'
        "}\n\n"
        "Now process the following task.\n"
    )
    return prompt_func


def post_process_types(parsed_data: dict[str, Any], chosen_func_obj: Any) -> dict[str, Any]:
    if not chosen_func_obj or not hasattr(chosen_func_obj, 'parameters'):
        return parsed_data

    if "parameters" in parsed_data and isinstance(parsed_data["parameters"], dict):
        for param_name, param_val in parsed_data["parameters"].items():
            if param_name in chosen_func_obj.parameters:
                expected_type = chosen_func_obj.parameters[param_name].type
                try:
                    if expected_type in ("number", "float"):
                        parsed_data["parameters"][param_name] = float(
                            param_val)
                    elif expected_type == "int":
                        parsed_data["parameters"][param_name] = int(param_val)
                    elif expected_type == "string":
                        parsed_data["parameters"][param_name] = str(param_val)
                except (ValueError, TypeError):
                    pass

    return parsed_data


def step_name(logits: NDArray[Any], logits_origin: NDArray[Any],
              vocab: dict[int, str], name_fonc: list[str], llm: Any,
              send_prompt: list[int]) -> None:
    logits[:] = -float("inf")
    characters_authorized: str = ('abcdefghijklmnopqrstuvwxyz'
                                  'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"')

    current_text: str = llm.decode(send_prompt)
    current_name: str = current_text.split('"name": "')[-1]

    for token_id, token_text in vocab.items():
        clean_text: str = token_text.replace(
            ' ', '').replace('Ġ', '').replace('Ċ', '').replace('<0x00>', '')

        if not clean_text:
            continue

        test_char = True
        for char in clean_text:
            if char not in characters_authorized:
                test_char = False
                break

        if not test_char:
            continue

        string: str = current_name + clean_text
        valide_string = False

        for name in name_fonc:
            if name.startswith(string):
                valide_string = True
                break

            if name == current_name and clean_text == '"':
                valide_string = True
                break

            if string == name + '"':
                valide_string = True
                break

        if valide_string:
            logits[int(token_id)] = logits_origin[int(token_id)]


def get_authorized_chars_dynamic(type_name: str, write: bool) -> str:
    if type_name == "key":
        return 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" ,:\n\t}'

    if not write or type_name is None:
        return ('abcdefghijklmnopqrstuvwxyz'
                'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"{},. :\n\t-')

    if type_name in ("number", "float"):
        return '0123456789.-, \n}'

    elif type_name == "int":
        return '0123456789-, \n}'

    elif type_name == "string":
        return ('abcdefghijklmnopqrstuvwxyz'
                'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ -.,!?\'"\n}')

    return ('abcdefghijklmnopqrstuvwxyz'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"{},. :\n\t-')


def should_continue_writing(type_name: str, texte_apres_colon: str) -> bool:
    if type_name in ("number", "float", "int"):
        if any(char in texte_apres_colon for char in [",", "}"]):
            return False
    else:
        if texte_apres_colon.count('"') >= 2:
            return False
    return True


def step_parameters(logits: NDArray[Any], logits_origin: NDArray[Any],
                    vocab: dict[int, str], llm: Any, send_prompt: list[int],
                    chosen_function: Any) -> None:
    logits[:] = -float("inf")
    current_text: Any = llm.decode(send_prompt)

    last_key_json = None
    key_position_json: Any = -1
    type_name: Any = None
    write: bool = True

    if chosen_function and hasattr(chosen_function, 'parameters'):
        for key in chosen_function.parameters.keys():
            idx: Any = current_text.rfind(f'"{key}"')
            if idx > key_position_json:
                key_position_json = idx
                last_key_json = key

    if last_key_json is not None:
        text_from_key: Any = current_text[key_position_json:]
        idx_colon: Any = text_from_key.find(":")

        if idx_colon != -1:
            texte_apres_colon: Any = text_from_key[idx_colon + 1:]
            type_name = chosen_function.parameters[last_key_json].type
            write = should_continue_writing(type_name, texte_apres_colon)
        else:
            type_name = "key"
    else:
        type_name = None

    caracteres_autorises: str = get_authorized_chars_dynamic(type_name, write)

    for token_id, token_text in vocab.items():
        clean_text: Any = token_text.replace(
            ' ', '').replace('Ġ', '').replace('Ċ', '\n').replace('<0x00>', '')

        if not clean_text:
            logits[int(token_id)] = logits_origin[int(token_id)]
            continue

        valide = all(char in caracteres_autorises for char in clean_text)

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

    resultats_finaux: list[dict[str, Any]] = []

    for line in data_prompt:
        starter: str = f'Task: {line.prompt}\nJSON:\n{{\n  "name": "'
        send_prompt: Any = function + llm.encode(starter)[0].tolist()

        state = 1
        chosen_func_obj: ParsngFunctions | None = None

        token_count = 0
        max_tokens = 45

        print(f"\n{Color.GREEN.value}\n\n[PROMPT] "
              f"{Color.BLUE.value}{line.prompt}"
              f"{Color.RESET.value}\n")

        size_start_prompt: int = len(send_prompt)

        while True:
            token_count += 1
            if token_count > max_tokens:
                print(f"{Color.RED.value}\n\n[ERREUR] "
                      f"{Color.WHITE.value}Token limit reached !!"
                      f"{Color.RESET.value}")

                resultats_finaux.append(
                    {"prompt": line.prompt, "name": "fn_none"})
                print("\n-----------------\n")
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
            if len(send_prompt) > size_start_prompt:
                print(f"{Color.WHITE.value}{result}"
                      f"{Color.RESET.value}", end="", flush=True)

            if state == 1:
                full_text: Any = llm.decode(send_prompt)
                name_func: str = full_text.split('"name": "')[-1]
                if '"' in name_func:
                    nom_genere: str = name_func.replace('"', '').strip()

                    if nom_genere == "fn_none":
                        print(
                            "\n[INFO] Le modèle a déterminé qu'aucune "
                            "fonction ne correspond.")
                        aide_json = '\n}'
                        aide_tokens: Any = llm.encode(aide_json)[0].tolist()
                        send_prompt.extend(aide_tokens)
                        print(aide_json, end="", flush=True)

                        resultats_finaux.append(
                            {"prompt": line.prompt, "name": "fn_none"})
                        break  # Sort du while pour passer au prompt suivant dans la boucle for

                    state = 2

                    for func in data_function:
                        if func.name == nom_genere:
                            chosen_func_obj = func
                            break

                    if chosen_func_obj is None:
                        print(f"{Color.RED.value}\n\n[ERREUR] "
                              f"{Color.WHITE.value}This function "
                              f"{Color.GREEN.value}'{nom_genere}' "
                              f"{Color.WHITE.value} was not found. "
                              f"{Color.RESET.value}")
                        resultats_finaux.append(
                            {"prompt": line.prompt, "name": "fn_none"})
                        break

                    aide_json = ',\n  "parameters": {\n    '
                    aide_tokens = llm.encode(aide_json)[0].tolist()
                    send_prompt.extend(aide_tokens)
                    print(aide_json, end="", flush=True)

            elif state == 2:
                texte_actuel: Any = llm.decode(send_prompt)
                nb_ouvertes: Any = texte_actuel.count('{')
                nb_fermees: Any = texte_actuel.count('}')

                if nb_ouvertes > 0 and nb_ouvertes == nb_fermees:
                    json_str = "{\n" + texte_actuel.split("JSON:\n{")[-1]

                    import re
                    json_str = re.sub(r':\s*r"', ': "', json_str)

                    try:
                        parsed_data = json.loads(json_str)
                        processed_data = post_process_types(
                            parsed_data, chosen_func_obj)

                        objet_json = {"prompt": line.prompt}
                        objet_json.update(processed_data)
                    except json.JSONDecodeError as error:
                        print(f"\n[ERROR] Failed to parse JSON: {error}")
                        objet_json = {"prompt": line.prompt, "name": "fn_none"}

                    resultats_finaux.append(objet_json)

                    print("\n\n-----------------\n\n")
                    break

    output(output_filename, resultats_finaux)
