UV ?= uv
COVERAGE_JSON ?= coverage.json

.PHONY: help quality lint format-check typecheck test architecture crap vulture smoke docs build docker benchmark

help:
	@printf '%s\n' 'Targets: quality lint format-check typecheck test architecture crap vulture smoke docs build docker benchmark'

quality: lint format-check typecheck test architecture crap vulture smoke docs

lint:
	$(UV) run ruff check src tests scripts benchmarks

format-check:
	$(UV) run ruff format --check src tests scripts benchmarks

typecheck:
	$(UV) run ty check src tests scripts benchmarks

test:
	HYPOTHESIS_PROFILE=ci $(UV) run pytest -p no:cacheprovider --cov --cov-report=json:"$(COVERAGE_JSON)" --cov-report=term-missing

architecture:
	$(UV) run python scripts/check_architecture.py

crap:
	$(UV) run python scripts/check_crap.py --coverage-json "$(COVERAGE_JSON)"

vulture:
	$(UV) run vulture src vulture_whitelist.py --min-confidence 60

smoke:
	$(UV) run python scripts/smoke.py

docs:
	$(UV) run mkdocs build --strict

build:
	$(UV) build

docker:
	docker build --tag osm-polygon-eunis:local .

benchmark:
	$(UV) run pytest -p no:cacheprovider benchmarks/
