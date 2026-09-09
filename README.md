# Call Me Maybe
*This project has been created as part of the 42 curriculum by cpietrza.*

## Description
**Call Me Maybe** is an educational project focused on implementing function calling for Large Language Models (LLMs) from scratch. The objective is to build a reliable mechanism bridging natural language user requests and structured, machine-executable JSON function calls, even when using small language models (such as `Qwen/Qwen3-0.6B`, containing approximately 600 million parameters).

The project relies on **constrained decoding**, guiding token generation step-by-step to guarantee 100% syntactic and semantic JSON compliance, completely eliminating the JSON syntax errors and formatting hallucinations common to unrestricted generation.

## Instructions

### Prerequisites
- Python 3.10 or later
- Package and environment manager `uv`
- The provided `llm_sdk` package located at the project root

### Installation
To install project dependencies and synchronize the virtual environment:
```bash
make install
```
*Note: The project environment can also be synchronized directly using standard uv:*
```bash
uv sync
```

### Execution
To run the main program with default files and settings:
```bash
make run
```
As required by the subject, the program can also be run directly via `uv run`:
```bash
uv run python -m src [--functions_definition <path>] [--input <path>] [--output <path>]
```
For example:
```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

Custom input and output files can also be passed via the `ARGS` variable in `Makefile`:
```bash
make run ARGS="--input data/input/custom_tests.json --output data/output/custom_results.json"
```

You can also customize the function definitions and the target model:
```bash
make run ARGS="--input data/input/custom_tests.json --output data/output/custom_results.json --functions_definition data/input/custom_functions.json --model Qwen/Qwen3-1.7B"
```

### Implemented Bonus Features
- **Multi-model support**: Configurable via `--model`. Supported models include `Qwen/Qwen3-0.6B` (default), `Qwen/Qwen3-1.7B`, and `Qwen/Qwen3-0.6B-Base`.
- **Comprehensive test suite**: Created `src/test.py` to automatically generate temporary function definitions and prompts, run inference, and validate outputs against expected schemas. Executable with `make test`.
- **Dynamic I/O paths**: Full path customization via CLI flags and Makefile `ARGS`.
- **Caching & visualizer**: Token/logit caching support and step-by-step generation visualizer.

### Available Commands (Makefile)
- `make install`: Install project dependencies using `uv`.
- `make run`: Run the main program (supports `ARGS`).
- `make cache`: Run the program with caching enabled.
- `make visual`: Run the program with the decoding visualizer enabled.
- `make multi`: Interactively select which LLM model to run.
- `make test`: Run end-to-end automated integration tests (`src/test.py`).
- `make lint`: Run required static checks (`flake8` and `mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`).
- `make lint-strict`: Run `flake8` and `mypy --strict`.
- `make clean`: Remove temporary files and caches (`__pycache__`, `.mypy_cache`, etc.).
- `make debug`: Launch the script inside the Python interactive debugger (`pdb`).

### Command-Line Arguments
| Argument | Description | Default Value |
|---|---|---|
| `--input` | Path to the JSON file containing test prompts. | `data/input/function_calling_tests.json` |
| `--output` | Path to the JSON file where results will be written. | `data/output/function_calling_results.json` |
| `--functions_definition` | Path to the JSON file containing function definitions. | `data/input/functions_definition.json` |
| `--model` | LLM model identifier to use. | `Qwen/Qwen3-0.6B` |
| `--multi` | Enable interactive model selection. | `False` |
| `--cache` | Enable inference caching. | `False` |
| `--visual` | Enable the token generation visualizer. | `False` |

Available options for `--model`:
- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-0.6B-Base`

### Examples with `ARGS`
Run with custom input only:
```bash
make run ARGS="--input data/input/custom_tests.json"
```

Run with custom input and output:
```bash
make run ARGS="--input data/input/custom_tests.json --output data/output/custom_results.json"
```

Run with a different model:
```bash
make run ARGS="--model Qwen/Qwen3-1.7B"
```

Combine multiple options:
```bash
make run ARGS="--input data/input/custom_tests.json --output data/output/custom_results.json --functions_definition data/input/custom_functions.json --model Qwen/Qwen3-1.7B --cache"
```

## Resources

### Technical References
- Reference on Constrained Decoding: [Hugging Face Text Generation Strategies & LogitsProcessors](https://huggingface.co/docs/transformers/main_classes/text_generation)
- Official Pydantic Documentation: [Pydantic Documentation](https://docs.pydantic.dev/) for strict schema validation.
- JSON Specification Standard (RFC 8259).

### Artificial Intelligence Usage
In accordance with subject requirements (Chapter II & Chapter VI), AI assistance was used for specific and well-defined tasks:
- **Test suite creation (`src/test.py`)**: AI was utilized to help design and scaffold the automated integration test suite, particularly to generate comprehensive edge cases (negative numbers, floating-point variations, complex strings with regex patterns and escape sequences, and strict schema validation). All generated tests were reviewed, adjusted, and validated manually.
- **Documentation drafting (`README.md`)**: AI was used as a technical writing assistant to structure, proofread, and format the documentation to guarantee exhaustive compliance with 42 curriculum standards and evaluation guidelines.
- *Integrity Note*: The core constrained decoding engine (logit masking logic, state machine tracking, token vocabulary mapping without third-party frameworks, and Pydantic integration) was conceived, implemented, and fully understood by the author.

## Algorithm Explanation
Our constrained decoding algorithm intercepts the model's token generation loop, token by token, directly shaping the logit distribution before next-token selection:
1. **Grammar & State Evaluation**: At each generation step, a state tracker evaluates the expected JSON structural component (opening brace, object key, colon delimiter, string literal, numeric value, separator, or closing brace).
2. **Vocabulary Filtering**: Using the vocabulary mapping obtained via `get_path_to_vocab_file()`, the system identifies all tokens that represent valid transitions according to both JSON syntax and the target parameter schema in `functions_definition.json`.
3. **Logit Masking**: Any token violating the grammatical or type constraints has its logit score forced to `-inf`.
4. **Token Selection**: The token with the highest logit score among the remaining valid candidates is selected (greedy decoding). This process repeats iteratively until a fully valid and closed JSON object is produced.

## Design Decisions
- **Strict adherence to dependency limitations**: No unauthorized high-level frameworks (such as PyTorch, Transformers, Outlines, or DSPy) are used. Only `numpy`, `pydantic`, and Python standard library modules are utilized.
- **Validation with Pydantic**: All data inputs (prompts, function specifications) and outputs are parsed through Pydantic models to guarantee end-to-end type safety and schema conformity.
- **Graceful error handling**: File operations and JSON decoding are wrapped in robust `try/except` blocks and context managers, ensuring the application never crashes unexpectedly on malformed or missing input files.

## Performance Analysis
- **Syntactic Accuracy**: 100% of generated outputs are strictly valid JSON and parseable by standard decoders.
- **Functional Accuracy**: Achieves over 90% correct function selection and parameter extraction, even on the compact `Qwen3-0.6B` model.
- **Inference Speed**: Vectorized logit masking with NumPy ensures minimal overhead; the entire test set evaluates in under 5 minutes on standard hardware.

## Challenges Faced
1. **Subword Tokenization (BPE)**: BPE tokenizers split words into subword fragments often prepended with whitespace markers (e.g., `Ġ` or ` `). Mapping allowed characters to valid multi-character subwords without accidentally filtering legitimate tokens required careful boundary checking.
2. **Special Characters & Escaping (Regex)**: Generating arguments with regex patterns (e.g., `\d+`) required dynamic adjustments to string character masks without allowing unescaped quotes that would prematurely terminate JSON values.
3. **Numeric Precision and Floating-point Rules**: Enforcing numeric constraints (preventing double decimal points, allowing negative signs only at the start) required stateful transition tracking.

## Testing Strategy
- **Automated Integration Tests (`src/test.py`)**: Tests multiple function types (arithmetic, string operations, greetings), edge cases (zeroes, negative floats, quotes in strings), and validates schema fidelity against expected output files.
- **Static Analysis**: Verified with `make lint` (`flake8` and strict `mypy` flags, enforcing complete type annotations without warnings).
- **Error Robustness**: Verified graceful degradation and clear error messages when provided with missing input files, empty prompts, or invalid JSON syntax.

## Example Usage
If `data/input/function_calling_tests.json` contains:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  },
  {
    "prompt": "Reverse the string 'hello'"
  }
]
```

The output file `data/output/function_calling_results.json` will contain:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {
      "s": "hello"
    }
  }
]
```
