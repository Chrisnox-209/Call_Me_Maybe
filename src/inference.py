"""Inference engine logic and token constrained generation."""

from typing import Any
import json
import numpy as np
from numpy.typing import NDArray
from llm_sdk import Small_LLM_Model  # type: ignore
from .parse import ParsngFunctions, ParsingPompt
from functools import lru_cache
from .utils import (
    charge_vocab, reverse_vocab, lst_name_fonction, output, Color
)


_ENCODE_CACHE: dict[str, list[int]] = {}

@lru_cache(maxsize=1)
def get_cached_vocab(model_name: str) -> dict[int, str]:
    """
    Caches the vocabulary based on the model name to avoid reloading it.
    """
    llm_temp = Small_LLM_Model(model_name=model_name)
    vocab = reverse_vocab(charge_vocab(llm_temp))
    return dict(vocab)


@lru_cache(maxsize=128)
def _build_prompt_func_cached(functions_signature: tuple) -> str:
    """
    Internal cached function that processes hashable primitive tuples.
    """
    prompt_func = (
        "You are an expert AI API router. You must map the user's task "
        "to the correct function.\n\n"
        "AVAILABLE FUNCTIONS:\n"
    )

    for name, description, params_tuple, returns_type in functions_signature:
        prompt_func += f"- name: {name}\n"
        prompt_func += f"  description: {description}\n"
        prompt_func += "  parameters:\n"
        for param_name, param_type in params_tuple:
            prompt_func += f"    - {param_name}: {param_type}\n"
        prompt_func += f"  returns: {returns_type}\n\n"

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
    )
    return prompt_func


def build_prompt_func(data_function: list[ParsngFunctions]) -> str:
    """
    Construct the base prompt string for function calling by converting 
    objects into a hashable structure for the cache.
    """
    signature = tuple(
        (
            f.name,
            f.description,
            tuple((p_name, p_obj.type)
                  for p_name, p_obj in f.parameters.items()),
            f.returns.type
        )
        for f in data_function
    )
    return _build_prompt_func_cached(signature)


def post_process_types(
    parsed_data: dict[str, Any], chosen_func_obj: Any
) -> dict[str, Any]:
    """
    Convert parsed JSON string values into their correct types 
    (like int or float).
    """
    if not chosen_func_obj or not hasattr(chosen_func_obj, 'parameters'):
        return parsed_data

    if "parameters" in parsed_data and isinstance(parsed_data["parameters"],
                                                  dict):
        for param_name, param_val in parsed_data["parameters"].items():
            if param_name in chosen_func_obj.parameters:
                expected_type = chosen_func_obj.parameters[param_name].type
                try:
                    if expected_type == "number" or expected_type == "float":
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
              vocab: dict[int, str], allowed_names: list[str], llm: Any,
              generated_tokens: list[int]) -> None:
    """
    Force the AI to only generate a valid function name from our list.
    """
    logits[:] = -float("inf")

    allowed_chars = ('abcdefghijklmnopqrstuvwxyz'
                     'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"')

    current_text = llm.decode(generated_tokens)
    current_name = current_text.split('"name": "')[-1]

    for token_id, token_text in vocab.items():
        clean_text = token_text.replace(' ', '').replace(
            'Ġ', '').replace('Ċ', '').replace('<0x00>', '')

        if clean_text == "":
            continue

        is_valid_char = True
        for char in clean_text:
            if char not in allowed_chars:
                is_valid_char = False
                break

        if not is_valid_char:
            continue

        future_string = current_name + clean_text
        is_valid_string = False

        for name in allowed_names:
            if name.startswith(future_string):
                is_valid_string = True
                break

            if name == current_name and clean_text == '"':
                is_valid_string = True
                break

            if future_string == name + '"':
                is_valid_string = True
                break

        if is_valid_string:
            logits[int(token_id)] = logits_origin[int(token_id)]


def cached_encode(llm: Any, text: str, use_cache: bool) -> list[int]:
    """
    Encodes text, using a dictionary cache if use_cache is True.
    """
    if not use_cache:
        return llm.encode(text)[0].tolist()

    if text not in _ENCODE_CACHE:
        _ENCODE_CACHE[text] = llm.encode(text)[0].tolist()
    return _ENCODE_CACHE[text]


def get_authorized_chars_dynamic(expected_type: str,
                                 is_writing_value: bool) -> str:
    """
    Return a simple list of characters the AI is allowed to type right now.
    """
    if expected_type == "key":
        return ('abcdefghijklmnopqrstuvwxyz'
                'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" ,:\n\t}')

    if is_writing_value is False or expected_type is None:
        return ('abcdefghijklmnopqrstuvwxyz'
                'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"{},. :\n\t-\\')

    if expected_type == "number" or expected_type == "float":
        return '0123456789.-, \n}'

    if expected_type == "int":
        return '0123456789-, \n}'

    if expected_type == "string":
        return ('abcdefghijklmnopqrstuvwxyz'
                'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ -.,!?\'"\n}\\[]+*^$|()')

    return ('abcdefghijklmnopqrstuvwxyz'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"{},. :\n\t-\\')


def should_continue_writing(expected_type: str, text_after_colon: str) -> bool:
    """
    Check if the AI is still writing the value, or if it is finished.
    """
    if expected_type in ("number", "float", "int"):
        for char in text_after_colon:
            if char in (",", "}"):
                return False
    else:
        quote_count = 0
        index = 0
        length = len(text_after_colon)

        while index < length:
            char = text_after_colon[index]
            if char == '\\':
                index += 2
                continue
            if char == '"':
                quote_count += 1
            index += 1

        if quote_count >= 2:
            return False

    return True


def step_parameters(logits: NDArray[Any], logits_origin: NDArray[Any],
                    vocab: dict[int, str], llm: Any,
                    generated_tokens: list[int],
                    chosen_function: Any) -> None:
    """
    Force the AI to only generate valid parameters for the chosen function.
    """
    logits[:] = -float("inf")
    current_text = llm.decode(generated_tokens)

    last_key_found = None
    key_position = -1
    expected_type = None
    is_writing_value = True

    if chosen_function and hasattr(chosen_function, 'parameters'):
        for key in chosen_function.parameters.keys():
            idx = current_text.rfind(f'"{key}"')
            if idx > key_position:
                key_position = idx
                last_key_found = key

    if last_key_found is not None:
        text_from_key = current_text[key_position:]
        idx_colon = text_from_key.find(":")

        if idx_colon != -1:
            text_after_colon = text_from_key[idx_colon + 1:]
            expected_type = chosen_function.parameters[last_key_found].type
            is_writing_value = should_continue_writing(
                expected_type, text_after_colon)
        else:
            expected_type = "key"
    else:
        expected_type = None

    allowed_chars = get_authorized_chars_dynamic(
        expected_type, is_writing_value)

    for token_id, token_text in vocab.items():
        clean_text = token_text.replace(' ', '').replace(
            'Ġ', '').replace('Ċ', '\n').replace('<0x00>', '')

        if clean_text == "":
            logits[int(token_id)] = logits_origin[int(token_id)]
            continue

        is_valid = True
        for char in clean_text:
            if char not in allowed_chars:
                is_valid = False
                break

            if is_valid:
                logits[int(token_id)] = logits_origin[int(token_id)]


def run_inference(
    data_prompt: list[ParsingPompt],
    data_function: list[ParsngFunctions],
    output_filename: str,
    model_name: str,
    cache: bool
) -> None:
    """
    The main loop that runs the AI model token by token for each test.
    Optimized to cache static components and tokenization overhead.
    """
    llm = Small_LLM_Model(model_name=model_name)
    vocab = get_cached_vocab(model_name)

    function_prompt = build_prompt_func(data_function)
    function_tokens = cached_encode(llm, function_prompt, cache)
    allowed_names = lst_name_fonction(data_function)

    helper_json_none = '\n}'
    helper_tokens_none = cached_encode(llm, helper_json_none, cache)

    helper_json_params = ',\n  "parameters": {\n    '
    helper_tokens_params = cached_encode(llm, helper_json_params, cache)

    final_results = []

    for item in data_prompt:
        starter = f'Task: {item.prompt}\nJSON:\n{{\n  "name": "'
        generated_tokens = function_tokens + cached_encode(llm, starter, cache)

        state = 1
        chosen_function_object = None

        token_count = 0
        max_tokens = 150

        print(f"\n{Color.GREEN.value}\n\n[PROMPT] "
              f"{Color.BLUE.value}{item.prompt}"
              f"{Color.RST.value}\n"
              f"{Color.WHITE.value}{{\n  \"name\": \"", end="", flush=True)

        size_start_prompt = len(generated_tokens)

        while True:
            token_count += 1
            if token_count > max_tokens:
                print(
                    f"{Color.RED.value}\n\n[ERROR] Token limit reached !!"
                    f"{Color.RST.value}")
                final_results.append(
                    {"prompt": item.prompt, "name": "fn_none"})
                print("\n-----------------\n")
                break

            logits = np.array(llm.get_logits_from_input_ids(generated_tokens))
            logits_origin = logits.copy()

            if state == 1:
                step_name(logits, logits_origin, vocab,
                          allowed_names, llm, generated_tokens)
            elif state == 2:
                step_parameters(logits, logits_origin, vocab,
                                llm, generated_tokens, chosen_function_object)

            next_token = int(np.argmax(logits))
            generated_tokens.append(next_token)

            result_text = llm.decode([next_token])
            if len(generated_tokens) > size_start_prompt:
                print(
                    f"{Color.WHITE.value}{result_text}{Color.RST.value}", end="", flush=True)

            if state == 1:
                full_text = llm.decode(generated_tokens)
                name_generated = full_text.split('"name": "')[-1]

                if '"' in name_generated:
                    clean_name = name_generated.replace('"', '').strip()

                    if clean_name == "fn_none":
                        generated_tokens.extend(helper_tokens_none)
                        print(
                            f"{Color.WHITE.value}{helper_json_none}"
                            f"{Color.RST.value}", end="", flush=True)

                        final_results.append(
                            {"prompt": item.prompt, "name": "fn_none"})
                        print("\n-----------------\n")
                        break

                    state = 2

                    for func in data_function:
                        if func.name == clean_name:
                            chosen_function_object = func
                            break

                    if chosen_function_object is None:
                        print(
                            f"{Color.RED.value}\n\n[ERROR] Function '"
                            f"{clean_name}' not found.{Color.RST.value}")
                        final_results.append(
                            {"prompt": item.prompt, "name": "fn_none"})
                        print("\n-----------------\n")
                        break

                    generated_tokens.extend(helper_tokens_params)
                    print(
                        f"{Color.WHITE.value}{helper_json_params}"
                        f"{Color.RST.value}", end="", flush=True)

            elif state == 2:
                current_text = llm.decode(generated_tokens)
                open_brackets = current_text.count('{')
                closed_brackets = current_text.count('}')

                if open_brackets > 0 and open_brackets == closed_brackets:
                    json_str = "{\n" + current_text.split("JSON:\n{")[-1]

                    # Nettoyages rapides de chaînes
                    json_str = json_str.replace(': r"', ': "').replace(
                        ':r"', ':"').replace(':  r"', ': "')
                    json_str = json_str.replace(": r'", ': "').replace(
                        ":r'", ':"').replace(":  r'", ': "')
                    json_str = json_str.replace("',", '",').replace(
                        "'\n", '"\n').replace("' \n", '" \n').replace("'}",
                                                                      '"}')

                    json_str = (json_str.replace("\\d", "\\\\d")
                                .replace("\\w", "\\\\w")
                                .replace("\\s", "\\\\s")
                                .replace("\\b", "\\\\b")
                                .replace("\\W", "\\\\W")
                                .replace("\\D", "\\\\D"))

                    try:
                        parsed_data = json.loads(json_str)
                        processed_data = post_process_types(
                            parsed_data, chosen_function_object)

                        json_object = {"prompt": item.prompt}
                        json_object.update(processed_data)
                    except json.JSONDecodeError as error:
                        print(
                            f"\n\n{Color.RED.value}[ERROR] "
                            "Failed to parse JSON: "
                            f"{error}{Color.RST.value}")
                        json_object = {
                            "prompt": item.prompt, "name": "fn_none"}

                    final_results.append(json_object)
                    print("\n-----------------\n")
                    break

    output(output_filename, final_results)