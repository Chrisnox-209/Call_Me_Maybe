MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
--disallow-untyped-defs --check-untyped-defs
MYPY_STRICT = --strict
FLAKE_STRICT = --max-complexity=5
MAIN = src/main.py

.PHONY: install run debug clean lint lint-strict

install:
	@uv sync

run:
	@uv run python $(MAIN) $(ARGS)

clean:
	@rm -Rf .venv
	@rm -Rf __pycache__
	@rm -Rf .mypy_cache
	@rm -Rf uv.lock
	@echo "All code clean"

lint:
	@uv run python3 -m mypy --exclude 'llm_sdk' $(MYPY_FLAGS) .
	@uv run python3 -m flake8 --exclude .venv,llm_sdk .

lint-strict:
	@uv run python3 -m mypy --exclude 'llm_sdk' $(MYPY_FLAGS) $(MYPY_STRICT) .
	@uv run python3 -m flake8 --exclude .venv,llm_sdk $(FLAKE_STRICT) .

debug:
	@uv run python3 -m pdb $(MAIN)