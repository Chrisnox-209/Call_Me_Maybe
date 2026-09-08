"""Inference engine logic and token constrained generation."""

from functools import lru_cache
import json
from typing import Optional, Union, cast

from llm_sdk import Small_LLM_Model  # type: ignore
import numpy as np
from numpy.typing import NDArray
from parse import TypeDef

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
        model_name: The identifier of the model.

    Returns:
        A dictionary mapping token IDs to string representations.
    """
    llm_temp: Small_LLM_Model = Small_LLM_Model(model_name=model_name)
    vocab_raw: dict[str, int] = charge_vocab(llm_temp)
    vocab_reversed: dict[int, str] = reverse_vocab(vocab_raw)
    return dict(vocab_reversed)


def cached_encode(
    llm: Small_LLM_Model, text: str, use_cache: bool
) -> list[int]:
    """Encode a string into token identifiers using an optional cache.

    Args:
        llm: The language model instance.
        text: The string to tokenize.
        use_cache: Flag indicating whether to store/retrieve from cache.

    Returns:
        The sequence of token identifiers.
    """
    if not use_cache:
        raw_tokens: NDArray[np.int64] = llm.encode(text)[0]
        token_list: list[int] = raw_tokens.tolist()
        return token_list

    if text not in _ENCODE_CACHE:
        raw_tokens = llm.encode(text)[0]
        _ENCODE_CACHE[text] = raw_tokens.tolist()

    return _ENCODE_CACHE[text]


def build_prompt_func(data_function: list[ParsngFunctions]) -> str:
    """Build the base system prompt listing all available functions.

    Args:
        data_function: List of parsed function definition objects.

    Returns:
        The constructed system prompt string.
    """
    prompt: str = (
        "You are an expert AI API router. You must map the user's task "
        "to the correct function.\n\n"
        "AVAILABLE FUNCTIONS:\n"
    )

    for func in data_function:
        prompt += f"- name: {func.name}\n"
        prompt += f"  description: {func.description}\n"
        prompt += "  parameters:\n"

        for param_name, param_obj in func.parameters.items():
            param_type_str: str = str(param_obj.type)
            prompt += f"    - {param_name}: {param_type_str}\n"

        prompt += f"  returns: {func.returns.type}\n\n"

    prompt += (
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
    return prompt


def find_function_by_name(
    data_function: list[ParsngFunctions], target_name: str
) -> Optional[ParsngFunctions]:
    """Locate a function definition matching a given name.

    Args:
        data_function: List of parsed function definitions.
        target_name: The name of the target function.

    Returns:
        The matching function object if found, otherwise None.
    """
    for func in data_function:
        if func.name == target_name:
            return func
    return None


def clean_token_representation(raw_token: str) -> str:
    """Sanitize internal tokenizer representations into normal characters.

    Args:
        raw_token: The string token from the vocabulary.

    Returns:
        The sanitized string.
    """
    token: str = raw_token.replace(" ", " ")
    token = token.replace("Ġ", " ")
    token = token.replace("Ċ", "\n")
    token = token.replace("<0x00>", "")
    return token


def mask_logits_for_function_name(
    logits: NDArray[np.float64],
    original_logits: NDArray[np.float64],
    vocab: dict[int, str],
    allowed_names: list[str],
    current_prefix: str,
) -> None:
    """Mask logits to constrain next tokens to valid function names.

    Args:
        logits: Target logit array to be updated in-place.
        original_logits: Untouched initial logit array.
        vocab: The vocabulary mapping IDs to text fragments.
        allowed_names: List of allowed function name strings.
        current_prefix: Name characters typed so far by the engine.
    """
    logits[:] = -float("inf")

    for token_id, raw_token_text in vocab.items():
        clean_text: str = clean_token_representation(raw_token_text)
        if clean_text == "":
            continue

        future_string: str = current_prefix + clean_text
        is_candidate_valid: bool = False

        for target_name in allowed_names:
            if target_name.startswith(future_string):
                is_candidate_valid = True
                break

            full_target_with_quote: str = target_name + '"'
            if full_target_with_quote.startswith(future_string):
                is_candidate_valid = True
                break

        if is_candidate_valid:
            logits[int(token_id)] = original_logits[int(token_id)]


def mask_logits_for_value(
    logits: NDArray[np.float64],
    original_logits: NDArray[np.float64],
    vocab: dict[int, str],
    expected_type: str,
) -> None:
    """Mask logits to restrict token generation based on the parameter type.

    Args:
        logits: Target logit array to be updated in-place.
        original_logits: Untouched initial logit array.
        vocab: The vocabulary mapping IDs to text fragments.
        expected_type: The expected data type (string, int, float, number).
    """
    logits[:] = -float("inf")

    numeric_chars: str = "0123456789.-,\n\t "
    integer_chars: str = "0123456789-,\n\t "

    for token_id, raw_token_text in vocab.items():
        clean_text: str = clean_token_representation(raw_token_text)
        if clean_text == "":
            continue

        is_allowed: bool = True

        if expected_type in ("number", "float"):
            for char in clean_text:
                if char not in numeric_chars:
                    is_allowed = False
                    break
        elif expected_type == "int":
            for char in clean_text:
                if char not in integer_chars:
                    is_allowed = False
                    break

        if is_allowed:
            logits[int(token_id)] = original_logits[int(token_id)]


def convert_value_to_type(
    raw_value: str, expected_type: str
) -> ParsedParamValue:
    """Cast a raw string value into its declared Python primitive type.

    Args:
        raw_value: The generated parameter text.
        expected_type: The formal type specification.

    Returns:
        The cast value or the stripped original string on failure.
    """
    cleaned_string: str = raw_value.strip()

    if expected_type in ("number", "float"):
        cleaned_num: str = (
            cleaned_string.replace(",", "").replace("\n", "").strip()
        )
        try:
            return float(cleaned_num)
        except ValueError:
            return 0.0

    if expected_type == "int":
        cleaned_int: str = (
            cleaned_string.replace(",", "").replace("\n", "").strip()
        )
        try:
            return int(cleaned_int)
        except ValueError:
            return 0

    if cleaned_string.endswith(","):
        cleaned_string = cleaned_string[:-1].strip()

    if cleaned_string.endswith('"'):
        cleaned_string = cleaned_string[:-1]

    if cleaned_string.startswith('"'):
        cleaned_string = cleaned_string[1:]

    return cleaned_string


def generate_function_name(
    llm: Small_LLM_Model,
    vocab: dict[int, str],
    allowed_names: list[str],
    prompt_tokens: list[int],
    visual: bool,
) -> tuple[str, list[int]]:
    """Constrain generation until a complete function name is produced.

    Args:
        llm: The language model instance.
        vocab: Model vocabulary dictionary.
        allowed_names: List of allowed function names.
        prompt_tokens: The input sequence tokens accumulated so far.
        visual: Flag to display generated tokens to standard output.

    Returns:
        A tuple with the generated function name and updated token sequence.
    """
    accumulated_tokens: list[int] = list(prompt_tokens)
    current_name: str = ""

    while True:
        raw_logits: NDArray[np.float64] = np.array(
            llm.get_logits_from_input_ids(accumulated_tokens),
            dtype=np.float64,
        )
        filtered_logits: NDArray[np.float64] = raw_logits.copy()

        mask_logits_for_function_name(
            filtered_logits,
            raw_logits,
            vocab,
            allowed_names,
            current_name,
        )

        next_token_id: int = int(np.argmax(filtered_logits))
        accumulated_tokens.append(next_token_id)

        token_text: str = clean_token_representation(vocab[next_token_id])

        if visual:
            print(
                f"{Color.WHITE.value}{token_text}{Color.RST.value}",
                end="",
                flush=True,
            )

        if '"' in token_text:
            parts: list[str] = token_text.split('"')
            current_name = current_name + parts[0]
            break

        current_name = current_name + token_text

    return current_name.strip(), accumulated_tokens


def generate_parameter_value(
    llm: Small_LLM_Model,
    vocab: dict[int, str],
    expected_type: str,
    prompt_tokens: list[int],
    visual: bool,
) -> tuple[str, list[int]]:
    """Constrain generation of a single parameter value until its delimiter.

    Args:
        llm: The language model instance.
        vocab: Model vocabulary dictionary.
        expected_type: The declared type of the target parameter.
        prompt_tokens: Current sequence tokens.
        visual: Flag to display generated tokens to standard output.

    Returns:
        A tuple with the value string and the updated token sequence.
    """
    accumulated_tokens: list[int] = list(prompt_tokens)
    value_text: str = ""
    max_value_tokens: int = 60
    step_count: int = 0

    while step_count < max_value_tokens:
        step_count += 1
        raw_logits: NDArray[np.float64] = np.array(
            llm.get_logits_from_input_ids(accumulated_tokens),
            dtype=np.float64,
        )
        filtered_logits: NDArray[np.float64] = raw_logits.copy()

        mask_logits_for_value(
            filtered_logits, raw_logits, vocab, expected_type
        )

        next_token_id: int = int(np.argmax(filtered_logits))
        accumulated_tokens.append(next_token_id)

        token_text: str = clean_token_representation(vocab[next_token_id])

        if visual:
            print(
                f"{Color.WHITE.value}{token_text}{Color.RST.value}",
                end="",
                flush=True,
            )

        if expected_type == "string":
            if '"' in token_text:
                parts: list[str] = token_text.split('"')
                value_text = value_text + parts[0]
                break
            value_text = value_text + token_text
        else:
            if (
                "," in token_text
                or "\n" in token_text
                or "}" in token_text
            ):
                cleaned_part: str = (
                    token_text.replace(",", "")
                    .replace("\n", "")
                    .replace("}", "")
                )
                value_text = value_text + cleaned_part
                break
            value_text = value_text + token_text

    return value_text, accumulated_tokens


def process_single_task(
    item: ParsingPompt,
    llm: Small_LLM_Model,
    vocab: dict[int, str],
    function_tokens: list[int],
    data_function: list[ParsngFunctions],
    allowed_names: list[str],
    cache: bool,
    visual: bool,
) -> dict[str, Union[str, dict[str, ParsedParamValue]]]:
    """Execute the structured generation pipeline for a single prompt.

    Args:
        item: The prompt item to process.
        llm: The language model instance.
        vocab: Vocabulary dictionary.
        function_tokens: Cached token representations of available functions.
        data_function: Function definitions list.
        allowed_names: Valid function names list.
        cache: Cache flag.
        visual: Print output flag.

    Returns:
        A dictionary containing the parsed function and parameter values.
    """
    starter: str = f'Task: {item.prompt}\nJSON:\n{{\n  "name": "'
    task_starter_tokens: list[int] = cached_encode(llm, starter, cache)

    active_tokens: list[int] = []
    for token_id in function_tokens:
        active_tokens.append(token_id)
    for token_id in task_starter_tokens:
        active_tokens.append(token_id)

    if visual:
        print(
            f"\n\n{Color.GREEN.value}[PROMPT] {Color.BLUE.value}{item.prompt}"
            f'{Color.RST.value}\n{Color.WHITE.value}{{\n  "name": "',
            end="",
            flush=True,
        )
    func_name: str
    func_name, active_tokens = generate_function_name(
        llm, vocab, allowed_names, active_tokens, visual
    )

    chosen_func: Optional[ParsngFunctions] = find_function_by_name(
        data_function, func_name
    )
    if chosen_func is None:
        return {"prompt": str(item.prompt), "error": "Function not found"}

    result_payload: dict[str, Union[str, dict[str, ParsedParamValue]]] = {
        "prompt": str(item.prompt),
        "name": func_name,
        "parameters": {},
    }

    params_starter: str = ',\n  "parameters": {\n'
    if visual:
        print(
            f"{Color.WHITE.value}{params_starter}{Color.RST.value}",
            end="",
            flush=True,
        )
    for token in cached_encode(llm, params_starter, cache):
        active_tokens.append(token)

    param_keys: list[str] = list(chosen_func.parameters.keys())
    total_keys: int = len(param_keys)

    for index, param_key in enumerate(param_keys):
        param_obj: TypeDef = chosen_func.parameters[param_key]
        expected_type: str = str(param_obj.type)

        key_prefix: str = f'    "{param_key}": '
        if expected_type == "string":
            key_prefix += '"'

        if visual:
            print(
                f"{Color.WHITE.value}{key_prefix}{Color.RST.value}",
                end="",
                flush=True,
            )
        for token in cached_encode(llm, key_prefix, cache):
            active_tokens.append(token)

        raw_val: str
        raw_val, active_tokens = generate_parameter_value(
            llm, vocab, expected_type, active_tokens, visual
        )

        parsed_val: ParsedParamValue = convert_value_to_type(
            raw_val, expected_type
        )
        param_container: dict[str, str | int | float] = cast(
            dict[str, ParsedParamValue], result_payload["parameters"]
        )
        param_container[param_key] = parsed_val

        closing_value_token: str = ""
        if expected_type == "string":
            closing_value_token = '"'

        separator: str = ",\n"
        if index == total_keys - 1:
            separator = "\n"

        suffix: str = closing_value_token + separator
        if visual:
            print(
                f"{Color.WHITE.value}{suffix}{Color.RST.value}",
                end="",
                flush=True,
            )
        for token in cached_encode(llm, suffix, cache):
            active_tokens.append(token)

    closing_json: str = "  }\n}"
    if visual:
        print(
            f"{Color.WHITE.value}{closing_json}{Color.RST.value}",
            end="",
            flush=True,
        )
        print("\n-----------------\n")

    return result_payload


def run_inference(
    data_prompt: list[ParsingPompt],
    data_function: list[ParsngFunctions],
    output_filename: str,
    model_name: str,
    cache: bool,
    visual: bool,
) -> None:
    """Run constrained inference over a series of tasks and save outputs.

    Args:
        data_prompt: List of user tasks/prompts.
        data_function: Available function prototypes.
        output_filename: File destination for final JSON dump.
        model_name: Name of the model to load.
        cache: Flag to enable token caching.
        visual: Flag to display interactive terminal output.
    """
    llm: Small_LLM_Model = Small_LLM_Model(model_name=model_name)
    vocab: dict[int, str] = get_cached_vocab(model_name)

    function_prompt: str = build_prompt_func(data_function)
    function_tokens: list[int] = cached_encode(llm, function_prompt, cache)
    allowed_names: list[str] = lst_name_fonction(data_function)

    final_results: list[
        dict[str, Union[str, dict[str, ParsedParamValue]]]
    ] = []

    for item in data_prompt:
        try:
            item_result: dict[str, str | dict[
                str, str | int | float]] = process_single_task(
                item,
                llm,
                vocab,
                function_tokens,
                data_function,
                allowed_names,
                cache,
                visual,
            )
            final_results.append(item_result)
        except (ValueError, KeyError, json.JSONDecodeError):
            fallback_error: dict[
                str, Union[str, dict[str, ParsedParamValue]]
            ] = {
                "prompt": str(item.prompt),
                "error": "Failed during generation",
            }
            final_results.append(fallback_error)

    output(output_filename, final_results)
