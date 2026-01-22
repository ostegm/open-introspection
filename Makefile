.PHONY: lint typecheck check fix test clean

# Run all checks
check: lint typecheck

# Linting with ruff
lint:
	uv run ruff check src/ experiments/

# Type checking with mypy
typecheck:
	uv run mypy src/

# Auto-fix lint issues
fix:
	uv run ruff check --fix src/ experiments/

# Run tests
test:
	uv run pytest

# Run experiments
exp1:
	uv run python experiments/01_setup_sanity_check.py

exp2:
	uv run python experiments/02_concept_extraction.py

exp3:
	uv run python experiments/03_introspection_test.py

# Clean up caches
clean:
	rm -rf __pycache__ .mypy_cache .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
