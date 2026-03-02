# comp-4360-project

Repository for the Winter 2026 COMP 4360 - Machine Learning project.

## Quickstart

1. Install `uv`:

  ```bash
  # macOS and Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Windows (not recommended)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

2. Set up the environment:

  ```bash
  uv sync
  ```

Afterwards, all Python scripts should be run using `uv run` in place of `python`/`python3`.

## Project Organization

For now, put all source code as in the original SimMIM repository inside `src/`.

Accompanying tests should be put in the `tests/` directory.
