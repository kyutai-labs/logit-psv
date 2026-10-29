PYTHON ?= .venv/bin/python
RUFF ?= .venv/bin/ruff

.PHONY: test lint format

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover -s tests

lint:
	$(RUFF) check .
	$(RUFF) format --check .

format:
	$(RUFF) check --fix .
	$(RUFF) format .
