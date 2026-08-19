MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
	--disallow-untyped-defs --check-untyped-defs
MYPY_STRICT = --strict
FLAKE_STRICT = --max-complexity=20
MAIN = src/__main__.py

.PHONY: all install run clean lint lint-strict debug vocab multi test cache

all: run

install:
	@uv sync

run:
	@uv run python -m src $(ARGS)

clean:
	@rm -Rf .venv
	@rm -Rf __pycache__
	@rm -Rf src/__pycache__
	@rm -Rf .mypy_cache
	@rm -Rf uv.lock
	@rm -Rf llm_sdk/.venv
	@rm -Rf data/output
	@rm -f data/input/test_*.json
	@echo "All code clean"

lint:
	@uv run python3 -m mypy --exclude 'llm_sdk' $(MYPY_FLAGS) .
	@uv run python3 -m flake8 --exclude .venv,llm_sdk .

lint-strict:
	@uv run python3 -m mypy --exclude 'llm_sdk' $(MYPY_FLAGS) $(MYPY_STRICT) .
	@uv run python3 -m flake8 --exclude .venv,llm_sdk $(FLAKE_STRICT) .

debug:
	@uv run python3 -m pdb $(MAIN)

vocab:
	@uv run python -m src.vocab $(ARGS)

multi:
	@uv run python -m src --multi $(ARGS)

cache:
	@uv run python -m src --cache $(ARGS)

test:
	@uv run python -m src.test
