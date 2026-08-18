*This project has been created as part of the 42 curriculum by sirz.*

## Description
**Call Me Maybe** is an educational project focused on implementing function calling for Large Language Models (LLMs) from scratch. The goal is to build a reliable mechanism that bridges the gap between natural language requests and structured, machine-executable JSON function calls, even when using small models (such as the 0.6B parameter model used here). 

## Instructions
### Prerequisites
- Python 3.10+
- The `uv` package manager (or any compatible python environment tool).

### Installation
To install the project dependencies:
```bash
make install
```

### Execution
To run the main program:
```bash
make run
```
Or use the script manually to specify input/output files and the model:
```bash
uv run python -m src --model Qwen/Qwen3-1.7B --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

### Bonus Features Implemented
- **Support for multiple LLM models**: You can pass the `--model` argument to switch models. Supported options are `Qwen/Qwen3-0.6B` (default), `Qwen/Qwen3-1.7B`, and `Qwen/Qwen3-0.6B-Base`.
- **Comprehensive test suite**: Created a beginner-friendly `src/test.py` script that automatically generates temporary function definitions and prompts, runs the model, and compares the generated output to an expected result. Runnable via `make test`.

### Other commands
- `make multi`: Interactively select the LLM model to run.
- `make test`: Run the end-to-end integration tests to verify the model's output.
- `make lint` / `make lint-strict`: Run Flake8 and Mypy checks.
- `make clean`: Remove all cache and temporary files.
- `make debug`: Launch the script in the Python debugger (pdb).

## Resources
The project relies exclusively on constrained decoding rather than advanced prompting frameworks. No external AI agents or heuristic magic were used. The `Small_LLM_Model` provided by the `llm_sdk` is used purely as an inference engine. 
* References on Constrained Decoding: [Hugging Face text generation strategies](https://huggingface.co/docs/transformers/main_classes/text_generation)
* Pydantic documentation for strict input schema validation.

## Algorithm explanation
Our constrained decoding algorithm intercepts the model's token generation process token-by-token. Instead of passively accepting the highest probability token, the algorithm evaluates the set of valid next characters (based on the expected JSON schema and the current parsing state).
1. We determine if we are expecting a JSON key, a string value, an integer, etc.
2. We map the valid allowed characters for that specific type.
3. Any token from the LLM's vocabulary that contains unauthorized characters has its logit score set to negative infinity (`-inf`).
4. The token with the highest logit score among the remaining *valid* tokens is selected.

## Design decisions
- **Simplicity first**: The logic is purposely built with straightforward `if/else` conditions and minimal abstractions to remain accessible and easy to understand for beginners.
- **Pydantic Validation**: All input configurations (prompts and function definitions) are robustly parsed using Pydantic models.
- **Dynamic Character Whitelists**: The system dynamically changes the set of allowed characters depending on what part of the JSON object is currently being generated.

## Performance analysis
- **Accuracy**: By explicitly blocking invalid syntax, the model boasts a 100% syntactically valid JSON generation rate.
- **Speed**: Processing takes less than 5 minutes for the entire prompt dataset on standard hardware, as constrained logic occurs entirely in fast Python loops without external API latency.
- **Reliability**: Schema mismatches (e.g. producing text when a float is expected) are practically impossible.

## Challenges faced
1. **Regex strings generation**: Instructing the model to generate correct regex strings (like `\\d+`) required tuning the allowed characters (adding brackets, backslashes, and asterisks) and providing clear one-shot examples in the system prompt.
2. **Handling subwords correctly**: Reconstructing complete words from subword tokens involved meticulous matching with the vocabulary space to avoid dropping leading spaces or special characters.

## Testing strategy
- Extensive testing was done with the `moulinette` script checking every edge case (mathematical additions, string substitutions, greeting generation).
- Output files were strictly verified against the exact parameter types expected by `functions_definition.json`.

## Example usage
If `function_calling_tests.json` contains:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  }
]
```
The output file `function_calling_results.json` will contain exactly:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```
