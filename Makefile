PYTHON ?= ../.conda/bin/python
PIP ?= $(PYTHON) -m pip

.PHONY: install-dev lint typecheck test check run-baselines

install-dev:
	$(PIP) install -r requirements.txt -r requirements-dev.txt

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

check: lint typecheck test

run-baselines:
	$(PYTHON) scripts/run_baselines.py
