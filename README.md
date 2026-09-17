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
The transport allows 5 seconds to connect and 600 seconds of read inactivity.
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

## Model evaluations

With the local server running, establish a baseline with:

```sh
jamie eval model --runs 5
```

This runs 20 fixed cases five times each: eight structured-output cases, six
planning cases, and six text instruction-following cases. The planning cases
check small dependency-ordering tasks with a single expected order. They do not
measure general planning ability or execute any actions.

Each round visits cases in case-ID order. Every request contains only the case's
unchanged user prompt, with a fresh conversation, temperature `0.6`, and a
`1024`-token limit. There are no additional instructions, JSON constraints,
retries, warm-up prompts, or prompt adjustments. Server-side sampling settings
and caching remain as configured on the running llama.cpp server.

The summary shows passes and failures, JSON validity, instruction following,
median latency, output tokens, token-count coverage, and total duration:

- **JSON validity:** complete response text must parse as JSON. Surrounding JSON
  whitespace is accepted; Markdown fences, explanatory text, duplicate keys,
  and non-finite numbers fail. This metric applies to the 14 JSON cases, so
  five runs produce 70 JSON attempts. Request failures count as failed parses.
- **Instruction following:** JSON responses must contain the expected fields,
  with matching types and values. Extra object fields are allowed; arrays must
  have the exact expected length and order. Text cases match exactly after
  removing surrounding whitespace. JSON parse failures and missing responses
  fail this check. It is measured across all cases.
- **Success:** a response must pass its expectation checks and complete without
  a request error, empty answer, or token-limit truncation. Parse success alone
  does not count as a pass.
- **Latency:** elapsed time for each complete backend call, including connection
  time and failures. The displayed median includes all completed attempts; it
  is not time to first token. Total duration includes evaluation overhead.
- **Output tokens:** the server's `usage.completion_tokens`, including reasoning
  tokens when counted by the server. Missing usage is recorded as `null`, not
  estimated or treated as zero. The summary shows the sum of available counts
  and how many attempts reported usage.

Reports are saved under `evals/results/` with unique timestamped names. Each
JSON report includes the exact case definitions and their SHA-256 hash,
settings (including the read timeout), timestamps, model IDs, unmodified answer
text, per-run metrics, failure reasons, and summary. Results are checkpointed after every completed
attempt. Ctrl-C saves an interrupted report; unfinished requests are not counted
as completed attempts. Earlier reports are preserved.

Exit codes are `0` when all attempts pass, `1` for failed evaluations or input /
report errors, `2` for invalid CLI options, and `130` for interruption. Model
failures do not stop the remaining cases.

Use a different case file or output directory when needed:

```sh
jamie eval model --cases evals/cases/planning.json --runs 5 --output-dir evals/results
python evals/runner.py --runs 5
```

Each case file contains a non-empty JSON array. IDs must be unique across all
loaded files. Cases default to JSON expectations:

```json
{
  "id": "json-001",
  "prompt": "Return a JSON object with action=finish and message=hello.",
  "expected": {"action": "finish"}
}
```

For exact text checks, use `"format": "text"` and a string `"expected"` value.
The CLI also accepts `--temperature`, `--max-tokens`, and `--config`. Keep the
cases and settings unchanged when comparing results to this baseline. Model
outputs are stochastic, so repeated runs need not produce identical results.

The [saved baseline](evals/results/baseline.md) contains all 100 attempts with
Qwen3.8-9B-Distill Q4_K_M. It used the earlier 120-second read timeout; the
current default is 600 seconds. The original results are preserved, with the
timeout and server settings recorded in a companion environment file.

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
Commands accept an explicit configuration file:

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
- `src/assistant/evaluation.py`: evaluation loading, scoring, metrics, and report saving.
- `src/assistant/eval_cli.py`: evaluation command and terminal summaries.
- `evals/runner.py`: checkout entry point for the evaluation command.
- `evals/cases/`: fixed baseline case files, also bundled in installed wheels.
- `evals/results/`: saved evaluation reports.
- `tests/`: CLI, configuration, transport, and evaluation behavior and failure cases.
- `assistant.toml`: minimal project configuration.
