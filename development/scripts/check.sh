#!/bin/bash
echo "======== Running ruff ========"
uv run ruff check --fix
echo "======== Running mypy ========"
uv run mypy .
echo "======== Running pydoclint ========"
uv run pydoclint .
echo "======== Running pytest ========"
uv run -m pytest -vv .
