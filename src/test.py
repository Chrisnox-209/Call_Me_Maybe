"""Integration test module for function calling pipeline."""

import json
import os
from typing import Any

from .__main__ import main as run_main
from .parse import Color

FUNCTIONS_FILE: str = "data/input/test_functions_definition.json"
PROMPTS_FILE: str = "data/input/test_function_calling_tests.json"
OUTPUT_FILE: str = "data/output/test_output.json"


def create_functions_definition() -> None:
    """Generate the JSON file defining available tool specifications."""
    functions: list[dict[str, Any]] = [
        {
            "name": "fn_get_weather",
            "description": "Get the current weather in a city.",
            "parameters": {"city": {"type": "string"}},
            "returns": {"type": "none"},
        },
        {
            "name": "fn_calculate_sum",
            "description": "Calculate the sum of two numbers.",
            "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
            "returns": {"type": "none"},
        },
        {
            "name": "fn_evaluate_math_expression",
            "description": "Evaluate a mathematical expression or formula.",
            "parameters": {
                "formula": {"type": "string"},
            },
            "returns": {"type": "none"},
        },
        {
            "name": "fn_substitute_string_with_regex",
            "description": (
                "Replace substrings matching a regex with a "
                "replacement string."
            ),
            "parameters": {
                "source_string": {"type": "string"},
                "regex": {"type": "string"},
                "replacement": {"type": "string"},
            },
            "returns": {"type": "none"},
        },
        {
            "name": "fn_set_alarm",
            "description": "Set an alarm for a specific time.",
            "parameters": {
                "time": {"type": "string"},
                "label": {"type": "string"},
            },
            "returns": {"type": "none"},
        },
        {
            "name": "fn_send_email",
            "description": "Send an email.",
            "parameters": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "returns": {"type": "none"},
        },
        {
            "name": "fn_play_music",
            "description": "Play a specific song.",
            "parameters": {
                "song": {"type": "string"},
                "artist": {"type": "string"},
            },
            "returns": {"type": "none"},
        },
        {
            "name": "fn_turn_on_lights",
            "description": "Turn on lights in a room.",
            "parameters": {"room": {"type": "string"}},
            "returns": {"type": "none"},
        },
        {
            "name": "fn_book_flight",
            "description": "Book a flight.",
            "parameters": {
                "destination": {"type": "string"},
                "date": {"type": "string"},
            },
            "returns": {"type": "none"},
        },
        {
            "name": "fn_create_calendar_event",
            "description": "Create a calendar event.",
            "parameters": {
                "title": {"type": "string"},
                "date": {"type": "string"},
                "time": {"type": "string"},
            },
            "returns": {"type": "none"},
        },
        {
            "name": "fn_translate_text",
            "description": "Translate text.",
            "parameters": {
                "text": {"type": "string"},
                "target_language": {"type": "string"},
            },
            "returns": {"type": "none"},
        },
    ]
    with open(FUNCTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(functions, f, indent=4)


def create_prompts_tests() -> None:
    """Generate the input JSON dataset containing user test queries."""
    prompts: list[dict[str, str]] = [
        {"prompt": "What is the weather like in Paris?"},
        {"prompt": "Calculate the sum of 5 and 10"},
        {"prompt": "Calculate the sum of -15.5 and 42"},
        {"prompt": "Calculate the sum of -8.25 and -11.75"},
        {"prompt": "Evaluate the formula 'sqrt(16) + 3 * (-2.5)'"},
        {
            "prompt": (
                "Compute the mathematical expression "
                "'(12.5 / -2) ^ 3 + cos(0)'"
            )
        },
        {
            "prompt": (
                "Replace 'apple' with 'orange' in 'I have an apple' using"
                " a regex"
            )
        },
        {"prompt": "Set an alarm for 08:00 AM labeled Wake Up"},
        {
            "prompt": (
                "Send an email to john@example.com with subject Hello and"
                " body How are you?"
            )
        },
        {"prompt": "Play Shape of You by Ed Sheeran"},
        {"prompt": "Turn on the lights in the living room"},
        {"prompt": "Book a flight to Tokyo on 2024-12-01"},
        {
            "prompt": (
                "Create an event called Doctor Appointment on 2024-10-15"
                " at 14:00"
            )
        },
        {"prompt": "Translate 'Hello world' into French"},
    ]
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=4)


def run_test() -> None:
    """Run model inference on prompts and validate predictions."""
    os.makedirs("data/input", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    print("1. Creating test files...")
    create_functions_definition()
    create_prompts_tests()

    print("2. Running the model (this may take a moment)...")
    try:
        run_main(
            PROMPTS_FILE,
            OUTPUT_FILE,
            FUNCTIONS_FILE,
            "Qwen/Qwen3-0.6B",
            True,
            True,
        )
    except Exception as e:
        print(
            f"\n{Color.RED.value}[ERROR]{Color.RST.value} "
            f"The program encountered an error: {e}"
        )
        return

    print("3. Checking the output...")
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            result: Any = json.load(f)
    except FileNotFoundError:
        print(
            f"\n{Color.RED.value}[ERROR]{Color.RST.value} "
            "Failed to write output file: Output file was not generated.\n"
        )
        return

    expected_result: list[dict[str, Any]] = [
        {
            "prompt": "What is the weather like in Paris?",
            "name": "fn_get_weather",
            "parameters": {"city": "Paris"},
        },
        {
            "prompt": "Calculate the sum of 5 and 10",
            "name": "fn_calculate_sum",
            "parameters": {"a": 5.0, "b": 10.0},
        },
        {
            "prompt": "Calculate the sum of -15.5 and 42",
            "name": "fn_calculate_sum",
            "parameters": {"a": -15.5, "b": 42.0},
        },
        {
            "prompt": "Calculate the sum of -8.25 and -11.75",
            "name": "fn_calculate_sum",
            "parameters": {"a": -8.25, "b": -11.75},
        },
        {
            "prompt": "Evaluate the formula 'sqrt(16) + 3 * (-2.5)'",
            "name": "fn_evaluate_math_expression",
            "parameters": {"formula": "sqrt(16) + 3 * (-2.5)"},
        },
        {
            "prompt": (
                "Compute the mathematical expression "
                "'(12.5 / -2) ^ 3 + cos(0)'"
            ),
            "name": "fn_evaluate_math_expression",
            "parameters": {"formula": "(12.5 / -2) ^ 3 + cos(0)"},
        },
        {
            "prompt": (
                "Replace 'apple' with 'orange' in 'I have an apple' using"
                " a regex"
            ),
            "name": "fn_substitute_string_with_regex",
            "parameters": {
                "source_string": "I have an apple",
                "regex": "apple",
                "replacement": "orange",
            },
        },
        {
            "prompt": "Set an alarm for 08:00 AM labeled Wake Up",
            "name": "fn_set_alarm",
            "parameters": {"time": "08:00", "label": "Wakeup"},
        },
        {
            "prompt": (
                "Send an email to john@example.com with subject Hello and"
                " body How are you?"
            ),
            "name": "fn_send_email",
            "parameters": {
                "to": "john@example.com",
                "subject": "Hello",
                "body": "How are you?",
            },
        },
        {
            "prompt": "Play Shape of You by Ed Sheeran",
            "name": "fn_play_music",
            "parameters": {"song": "Shape of You", "artist": "Ed Sheeran"},
        },
        {
            "prompt": "Turn on the lights in the living room",
            "name": "fn_turn_on_lights",
            "parameters": {"room": "living room"},
        },
        {
            "prompt": "Book a flight to Tokyo on 2024-12-01",
            "name": "fn_book_flight",
            "parameters": {
                "destination": "Tokyo",
                "date": "2024-12-01",
            },
        },
        {
            "prompt": (
                "Create an event called Doctor Appointment on 2024-10-15"
                " at 14:00"
            ),
            "name": "fn_create_calendar_event",
            "parameters": {
                "title": "Doctor Appointment",
                "date": "2024-10-15",
                "time": "14:00",
            },
        },
        {
            "prompt": "Translate 'Hello world' into French",
            "name": "fn_translate_text",
            "parameters": {
                "text": "Hello world",
                "target_language": "fr",
            },
        },
    ]

    print("\n--- RESULTS ---")
    if result == expected_result:
        print(
            f"\n{Color.GREEN.value}[SUCCESS]{Color.RST.value} "
            "Results successfully saved to "
            f"'{Color.BLUE.value}{OUTPUT_FILE}{Color.RST.value}'.\n"
        )
    else:
        print(
            f"\n{Color.RED.value}[ERROR]{Color.RST.value} "
            "Failed to match generated JSON with expected result.\n"
        )
        print("\nExpected:")
        print(json.dumps(expected_result, indent=4, ensure_ascii=False))
        print("\nGot:")
        print(json.dumps(result, indent=4, ensure_ascii=False))

    print(
        f"\nFiles were saved in '{FUNCTIONS_FILE}', '{PROMPTS_FILE}' "
        f"and '{OUTPUT_FILE}'."
    )
    print("Run 'make clean' to delete them.")


if __name__ == "__main__":
    run_test()
