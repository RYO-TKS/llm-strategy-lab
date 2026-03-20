SHELL := /bin/bash
PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PYTEST := $(BIN)/pytest
RUFF := $(BIN)/ruff

.PHONY: help setup lint test run-sample clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "%-12s %s\n", $$1, $$2}'

setup: ## Create a virtualenv and install dependencies
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]

lint: ## Run Ruff and syntax checks
	PYTHONPATH=packages/core/src $(RUFF) check packages tests
	PYTHONPATH=packages/core/src $(BIN)/python -m compileall packages tests

test: ## Run unit tests
	PYTHONPATH=packages/core/src $(PYTEST)

run-sample: ## Run the scaffold experiment runner
	PYTHONPATH=packages/core/src $(BIN)/python -m llm_strategy_lab.cli --config configs/experiments/sample_research.yaml

clean: ## Remove local caches and build outputs
	rm -rf $(VENV) .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
