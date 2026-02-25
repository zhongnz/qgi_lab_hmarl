PYTHON ?= ../.conda/bin/python
PIP ?= $(PYTHON) -m pip

.PHONY: install-dev lint typecheck test check run-baselines clean

install-dev:
	$(PIP) install -r requirements.txt -r requirements-dev.txt

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest tests/ -q

check: lint typecheck test

run-baselines:
	$(PYTHON) scripts/run_baselines.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache runs/
