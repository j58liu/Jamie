# j_cli

## Milestone 0 status

A Python 3.12+ foundation for a local-first macOS AI assistant. The implemented
commands are `assistant --version` and `assistant doctor`, using Typer and Rich.
Configuration is validated with Pydantic. No LLM integration, agent loop, tools,
memory, or browser automation is implemented.

## Development

Use a Python 3.12 or newer interpreter. From the repository root:

```sh
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

assistant --version
assistant doctor
assistant --help
```

The editable install exposes the `assistant` command. `python -m assistant` is
also supported. No API keys, model downloads, or macOS permissions are required.

Run the development checks inside the activated environment:

```sh
python -m pytest
ruff check .
ruff format --check .
```

To format code or apply safe lint fixes:

```sh
ruff format .
ruff check --fix .
```

## Configuration and diagnostics

`assistant doctor` checks exactly two things: Python is version 3.12 or newer,
and the project configuration is valid. It displays a Rich table and exits with
code `0` when both pass, or `1` if either fails. It does not perform network
requests, create directories, or write configuration files.

By default, configuration is read from `assistant.toml` in the current working
directory. If that file is absent, built-in defaults are used. The only setting
in milestone 0 is whether diagnostics use color:

```toml
color = true
```

Set `color = false` for uncolored diagnostics. Unknown fields, incorrect types
(including strings instead of booleans), and malformed TOML are rejected. To
validate a specific file:

```sh
assistant doctor --config /path/to/assistant.toml
```

An explicitly requested file must exist and be readable. `assistant --version`
works independently of configuration and reads the installed package version.

## Project layout

- `pyproject.toml`: package metadata, dependencies, CLI entry point, and check settings.
- `src/assistant/`: CLI and Pydantic configuration loading.
- `tests/`: CLI behavior and configuration failure cases.
- `assistant.toml`: minimal project configuration.
- `AGENTS.md`: scope, development practices, and commit approval rules.

## Original project vision

The original notes below describe future ambitions, not milestone 0 capabilities.

### Local CLI built dedicated for my personal 2022 M2 16 GB Macbook Air
This CLI is able to:
- Control Macbook
- Conduct research
- Autonomous Coding
- Etc...

Currently supporting:
- Qwen 3.5 9B (distilled from Qwen 3.8 9B)
- OpenAI
- Claude
- Gemini
