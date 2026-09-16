# Jamie

A local-first macOS assistant using Python 3.12+, Typer, Rich, and Pydantic.
Jamie sends text prompts to a local llama.cpp server using HTTPX.

## Development

Use a Python 3.12 or newer interpreter. From the repository root:

```sh
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

jamie --version
jamie doctor
jamie --help
```

The editable install exposes the `jamie` command. The Python package remains
`assistant`, so `python -m assistant` is also supported.

Run the development checks inside the activated environment:

```sh
python -m pytest
ruff check .
ruff format --check .
jamie --version
jamie doctor
```

The automated tests use HTTPX mock transports and do not require a model or a
running server.

## Ask the local model

The selected model is
[empero-ai/Qwen3.8-9B-Distill](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill).
For llama.cpp, use the publisher's
[GGUF release](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill-GGUF),
specifically `Qwen3.8-9B-Q4_K_M.gguf` (about 5.78 GB). The linked source model
contains Transformers weights; the GGUF release is the format llama.cpp loads.

Install llama.cpp on macOS with Homebrew:

```sh
brew install llama.cpp
```

In a separate terminal, start the server. The first run downloads the selected
GGUF into llama.cpp's model cache:

```sh
llama-server \
  -hf empero-ai/Qwen3.8-9B-Distill-GGUF:Q4_K_M --no-mmproj \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 8192 --parallel 1 \
  --jinja --reasoning-format deepseek \
  --temp 0.6 --top-p 0.95 --top-k 20
```

Use a recent llama.cpp build with support for this model's Qwen3.5 architecture.
The 8,192-token context and single request slot provide a starting configuration
for a 16 GB Mac; memory use also depends on other running applications.
`--reasoning-format deepseek` separates reasoning from the final answer in the
HTTP response, so Jamie prints the answer text.

Keep the server running. Then, in the activated project environment:

```sh
jamie ask "hello"
jamie ask "Explain Python generators briefly." --temperature 0.6 --max-tokens 1024
jamie ask "Explain your answer to this puzzle: ..." --max-tokens 4096
```

Each invocation sends one user message in a non-streaming HTTP POST to
`/v1/chat/completions` and prints the answer as plain text. The server chooses
its loaded model; Jamie does not hardcode a model name or download model files.
No OpenAI SDK or API key is used. See the
[llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
for server setup and chat template options.

The defaults are temperature `0.6` and a maximum of `1024` generated tokens.
The transport allows 5 seconds to connect and 120 seconds of read inactivity.
Requests go directly to the server without using environment proxy settings,
following redirects, or automatically retrying.

Connection failures, timeouts, HTTP errors, and invalid responses produce an
error on stderr and exit code `1`. An empty prompt or invalid generation option
exits with code `2`. If generation reaches the token limit, the answer is printed
with a notice on stderr; use a larger `--max-tokens` value if needed. Reasoning
models may consume that budget before producing answer text, which Jamie reports
as an error instead of printing a blank answer.

To manually verify a running model:

```sh
curl --noproxy '*' http://127.0.0.1:8080/health
jamie ask "Reply with one short greeting."
```

## Configuration and diagnostics

`jamie doctor` checks exactly two things: Python is version 3.12 or newer,
and the project configuration is valid. It displays a Rich table and exits with
code `0` when both pass, or `1` if either fails. It does not perform network
requests, create directories, or write configuration files.

Configuration is read from `assistant.toml` in the current working directory.
If that file is absent, built-in defaults are used:

```toml
color = true
```

Set `color = false` for uncolored output. Unknown fields, incorrect types
(including strings instead of booleans), and malformed TOML are rejected.
Both commands accept an explicit configuration file:

```sh
jamie doctor --config /path/to/assistant.toml
jamie ask "hello" --config /path/to/assistant.toml
```

An explicitly requested file must exist and be readable. `jamie --version`
works independently of configuration and reads the installed package version.

## Project layout

- `pyproject.toml`: package metadata, dependencies, CLI entry point, and check settings.
- `src/assistant/cli.py`: Jamie commands.
- `src/assistant/config.py`: local Pydantic configuration loading.
- `src/assistant/llm/base.py`: `Message`, `LLMResponse`, `LLMError`, and `LLMBackend`.
- `src/assistant/llm/llamacpp.py`: llama.cpp HTTP transport.
- `tests/`: CLI, configuration, and HTTP transport behavior and failure cases.
- `assistant.toml`: minimal project configuration.
