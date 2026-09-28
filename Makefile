.PHONY: install test lint run show-queue demo

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

run:
	triage run

show-queue:
	triage show-queue

demo: run show-queue
