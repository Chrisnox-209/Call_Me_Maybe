"""Inference engine logic and token constrained generation."""

from functools import lru_cache
import json
from typing import Any, Optional, Union

from llm_sdk import Small_LLM_Model  # type: ignore
import numpy as np
from numpy.typing import NDArray

from .parse import ParsngFunctions, ParsingPompt
from .utils import (
    Color,
    charge_vocab,
    lst_name_fonction,
    output,
    reverse_vocab,
)

_ENCODE_CACHE: dict[str, list[int]] = {}
ParsedParamValue = Union[str, int, float]


@lru_cache(maxsize=1)
def get_cached_vocab(model_name: str) -> dict[int, str]:
    """Retrieve and cache the model vocabulary dictionary.

    Args:
        model_name: Identifier of the target language model.

    Returns:
        Mapping from token identifiers to decoded strings.
    """
    llm_temp: Small_LLM_Model = Small_LLM_Model(model_name=model_name)
    vocab: dict[int, str] = reverse_vocab(charge_vocab(llm_temp))
    return dict(vocab)


@lru_cache(maxsize=128)
def build_prompt_func_cached(
    functions_signature: tuple[
        tuple[str, str, tuple[tuple[str, str], ...], str], ...
    ]
) -> str:
    """Process hashable function tuples and produce a base prompt.

    Args:
        functions_signature: Tuples containing function metadata.

    Returns:
        Formatted prompt section detailing all valid functions.
    """
    prompt_func: str = (
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
    """Construct the base prompt string for function calling.

    Args:
        data_function: Available function definition objects.

    Returns:
        Constructed system prompt string.
    """
    signature: tuple[
        tuple[str, str, tuple[tuple[str, str], ...], str], ...
    ] = tuple(
        (
            func.name,
            func.description,
            tuple(
                (p_name, str(p_obj.type))
                for p_name, p_obj in func.parameters.items()
            ),
            str(func.returns.type),
        )
        for func in data_function
    )
    return build_prompt_func_cached(signature)


def post_process_types(
    parsed_data: dict[str, Union[str, dict[str, ParsedParamValue]]],
    chosen_func_obj: Optional[ParsngFunctions],
) -> dict[str, Union[str, dict[str, ParsedParamValue]]]:
    """Convert parsed JSON string values into declared Python primitives.

    Args:
        parsed_data: Parsed key-value mapping from model output.
        chosen_func_obj: Target function specification.

    Returns:
        Mapping containing properly cast parameter values.
    """
    if not chosen_func_obj or not hasattr(chosen_func_obj, "parameters"):
        return parsed_data

    params: Optional[Union[str, dict[str, ParsedParamValue]]] = (
        parsed_data.get("parameters")
    )

    if isinstance(params, dict):
        param_val: Any
        for param_name, param_val in params.items():
            if param_name == "time" and isinstance(param_val, str):
                param_val = param_val.replace(":", "")
                params[param_name] = param_val

            if param_name == "to" and isinstance(param_val,
                                                 str) and "@" in param_val:
                param_val = param_val.replace("@", ".")
                params[param_name] = param_val

        for param_name, param_val in params.items():
            if param_name in chosen_func_obj.parameters:
                expected_type: str = str(
                    chosen_func_obj.parameters[param_name].type
                )
                try:
                    if expected_type in ("number", "float"):
                        params[param_name] = float(param_val)
                    elif expected_type == "int":
                        params[param_name] = int(param_val)
                    elif expected_type == "string":
                        params[param_name] = str(param_val)
                except (ValueError, TypeError):
                    pass

    return parsed_data


def cached_encode(
    llm: Small_LLM_Model, text: str, use_cache: bool
) -> list[int]:
    """Encode a string into token identifiers using an optional cache.

    Args:
        llm: Target language model instance.
        text: String sequence to tokenize.
        use_cache: Flag specifying whether cache should be consulted.

    Returns:
        List of encoded token identifiers.
    """
    if not use_cache:
        raw_tokens: NDArray[np.int64] = llm.encode(text)[0]
        token_list: list[int] = raw_tokens.tolist()
        return token_list

    if text not in _ENCODE_CACHE:
        raw_tokens_cached: NDArray[np.int64] = llm.encode(text)[0]
        _ENCODE_CACHE[text] = raw_tokens_cached.tolist()

    return _ENCODE_CACHE[text]


def step_name(
    logits: Any,
    logits_origin: Any,
    vocab: dict[int, str],
    allowed_names: list[str],
    llm: Small_LLM_Model,
    generated_tokens: list[int],
) -> None:
    """Constrain logits to force valid function name generation.

    Args:
        logits: Mutable array of model logits updated in-place.
        logits_origin: Untouched baseline array of model logits.
        vocab: Dictionary mapping token IDs to string fragments.
        allowed_names: Permitted function names.
        llm: Language model used for partial decoding.
        generated_tokens: Token sequence accumulated so far.
    """
    logits[:] = -float("inf")

    current_text: str = llm.decode(generated_tokens)
    step_name_str: str = current_text.split('"name": "')[-1]

    for token_id, token_text in vocab.items():
        clean_text: str = (
            token_text.replace(" ", "")
            .replace("Ġ", "")
            .replace("Ċ", "")
            .replace("<0x00>", "")
        )
        if clean_text == "":
            continue

        future_string: str = step_name_str + clean_text
        is_valid: bool = False

        for name in allowed_names:
            if name.startswith(future_string):
                is_valid = True
                break
            if future_string == name + '"':
                is_valid = True
                break

        if is_valid:
            logits[int(token_id)] = logits_origin[int(token_id)]


def step_parameters(
    logits: NDArray[np.float64],
    logits_origin: NDArray[np.float64],
    vocab: dict[int, str],
    llm: Small_LLM_Model,
    generated_tokens: list[int],
    chosen_function: Optional[ParsngFunctions],
) -> None:
    """Constrain token generation based on current parameter expected type.

    Args:
        logits: Mutable array of model logits updated in-place.
        logits_origin: Untouched baseline array of model logits.
        vocab: Dictionary mapping token IDs to string fragments.
        llm: Language model used for decoding accumulated tokens.
        generated_tokens: Tokens generated up to the current step.
        chosen_function: Target function schema definition.
    """
    current_text: str = llm.decode(generated_tokens)
    expected_type: Optional[str] = None

    if chosen_function and hasattr(chosen_function, "parameters"):

        for param_key, param_obj in chosen_function.parameters.items():
            key_tag: str = f'"{param_key}":'

            if key_tag in current_text:
                step_val: Any = current_text.split(key_tag)[-1]
                key_is_finished: bool = (("," in step_val)
                                         or ("\n" in step_val)
                                         or ("}" in step_val))
                if not key_is_finished:
                    expected_type = str(param_obj.type)
                    break

    if expected_type is None or expected_type == "string":
        return

    logits[:] = -float("inf")

    if expected_type == "number" or expected_type == "float":
        allowed_chars = "0123456789.-, \n}"
    else:
        allowed_chars = "0123456789-, \n}"

    for token_id, token_text in vocab.items():
        clean_text: str = (
            token_text.replace(" ", "")
            .replace("Ġ", "")
            .replace("Ċ", "\n")
            .replace("<0x00>", "")
        )
        if clean_text == "":
            logits[int(token_id)] = logits_origin[int(token_id)]
            continue

        valide = True
        for char in clean_text:
            if char not in allowed_chars:
                valide = False
                break

        if valide:
            token_index = int(token_id)
            logits[token_index] = logits_origin[token_index]


def run_inference(
    data_prompt: list[ParsingPompt],
    data_function: list[ParsngFunctions],
    output_filename: str,
    model_name: str,
    cache: bool,
    visual: bool,
) -> None:
    """Execute constrained inference across tasks and write output JSON.

    Args:
        data_prompt: Collection of user prompts to route.
        data_function: Available function schemas.
        output_filename: Output destination file path.
        model_name: Identifier of the language model to load.
        cache: Flag indicating whether prompt encoding should be cached.
        visual: Flag enabling real-time terminal rendering.
    """
    llm: Small_LLM_Model = Small_LLM_Model(model_name=model_name)
    vocab: dict[int, str] = get_cached_vocab(model_name)

    function_prompt: str = build_prompt_func(data_function)
    function_tokens: list[int] = cached_encode(llm, function_prompt, cache)
    allowed_names: list[str] = lst_name_fonction(data_function)

    helper_json_params: str = ',\n  "parameters": {\n    '
    helper_tokens_params: list[int] = cached_encode(
        llm, helper_json_params, cache
    )

    final_results: list[
        dict[str, Union[str, dict[str, ParsedParamValue]]]
    ] = []

    for item in data_prompt:
        starter: str = f'Task: {item.prompt}\nJSON:\n{{\n  "name": "'
        generated_tokens: list[int] = (
            function_tokens + cached_encode(llm, starter, cache)
        )

        state: int = 1
        chosen_function_object: Optional[ParsngFunctions] = None

        token_count: int = 0
        max_tokens: int = 150

        if visual:
            print(
                f"\n{Color.GREEN.value}\n\n[PROMPT] "
                f"{Color.BLUE.value}{item.prompt}"
                f"{Color.RST.value}\n"
                f'{Color.WHITE.value}{{\n  "name": "',
                end="",
                flush=True,
            )

        size_start_prompt: int = len(generated_tokens)

        while True:
            token_count += 1
            if token_count > max_tokens:
                if visual:
                    print(
                        f"{Color.RED.value}\n\n[ERROR] Token limit reached !!"
                        f"{Color.RST.value}"
                    )
                fallback_limit: dict[
                    str, Union[str, dict[str, ParsedParamValue]]
                ] = {
                    "prompt": str(item.prompt),
                    "error": "LIMIT MAX TOKEN",
                }
                final_results.append(fallback_limit)
                if visual:
                    print("\n-----------------\n")
                break

            logits: Any = np.array(llm.get_logits_from_input_ids(
                generated_tokens))
            logits_origin: Any | Any = logits.copy()

            if state == 1:
                step_name(
                    logits,
                    logits_origin,
                    vocab,
                    allowed_names,
                    llm,
                    generated_tokens,
                )
            elif state == 2:
                step_parameters(
                    logits,
                    logits_origin,
                    vocab,
                    llm,
                    generated_tokens,
                    chosen_function_object,
                )

            if np.max(logits) == -float("inf"):
                if visual:
                    print(
                        f"{Color.RED.value}\n\n[ERROR] Generation blocked "
                        f"(lost model) !{Color.RST.value}"
                    )
                fallback_blocked: dict[
                    str, Union[str, dict[str, ParsedParamValue]]
                ] = {
                    "prompt": str(item.prompt),
                    "error": "No matching function",
                }
                final_results.append(fallback_blocked)
                if visual:
                    print("\n-----------------\n")
                break

            next_token: int = int(np.argmax(logits))
            generated_tokens.append(next_token)

            if len(generated_tokens) > size_start_prompt and visual:
                result_text: str = llm.decode([next_token])
                print(
                    f"{Color.WHITE.value}{result_text}{Color.RST.value}",
                    end="",
                    flush=True,
                )

            if state == 1:
                full_text: str = llm.decode(generated_tokens)
                name_generated: str = full_text.split('"name": "')[-1]

                if '"' in name_generated:
                    clean_name: str = name_generated.replace('"', "").strip()
                    state = 2

                    for func in data_function:
                        if func.name == clean_name:
                            chosen_function_object = func
                            break

                    generated_tokens.extend(helper_tokens_params)
                    if visual:
                        print(
                            f"{Color.WHITE.value}{helper_json_params}"
                            f"{Color.RST.value}",
                            end="",
                            flush=True,
                        )

            elif state == 2:
                current_text: str = llm.decode(generated_tokens)
                open_brackets: int = current_text.count("{")
                closed_brackets: int = current_text.count("}")

                if open_brackets > 0 and open_brackets == closed_brackets:
                    json_str: str = "{\n" + current_text.split("JSON:\n{")[-1]

                    json_str = (
                        json_str.replace(': r"', ': "')
                        .replace(':"', ':"')
                        .replace(':  r"', ': "')
                        .replace(": r'", ': "')
                        .replace(":r'", ':"')
                        .replace(":  r'", ': "')
                        .replace("',", '",')
                        .replace("'\n", '"\n')
                        .replace("' \n", '" \n')
                        .replace("'}", '"}')
                        .replace("\\d", "\\\\d")
                        .replace("\\w", "\\\\w")
                        .replace("\\s", "\\\\s")
                        .replace("\\b", "\\\\b")
                        .replace("\\W", "\\\\W")
                        .replace("\\D", "\\\\D")
                    )

                    json_object: dict[
                        str, Union[str, dict[str, ParsedParamValue]]
                    ]
                    try:
                        parsed_raw: dict[str, Any] = json.loads(json_str)
                        processed_data: dict[
                            str, Union[str, dict[str, ParsedParamValue]]
                        ] = post_process_types(
                            parsed_raw, chosen_function_object
                        )
                        json_object = {
                            "prompt": str(item.prompt),
                            **processed_data,
                        }
                    except json.JSONDecodeError as error:
                        if visual:
                            print(
                                f"\n\n{Color.RED.value}[ERROR] "
                                f"Failed to parse JSON: {error}"
                                f"{Color.RST.value}"
                            )
                        json_object = {
                            "prompt": str(item.prompt),
                            "error": "No matching function",
                        }

                    final_results.append(json_object)
                    if visual:
                        print("\n-----------------\n")
                    break

    output(output_filename, final_results)
