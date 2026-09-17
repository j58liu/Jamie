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

## Action protocol

Model responses intended for action handling have exactly two forms. A tool
action names a tool and supplies a JSON object of arguments:

```json
{
  "action": "tool",
  "tool": "read_file",
  "args": {
    "path": "README.md"
  }
}
```

A finish action contains the final user-facing message:

```json
{
  "action": "finish",
  "message": "The README describes..."
}
```

`parse_action` validates these responses with Pydantic. It accepts direct JSON,
JSON enclosed by one outer `json` Markdown fence, and either form following one
leading `<think>...</think>` block. It rejects malformed JSON, unknown actions,
missing or incorrectly typed fields, duplicate JSON keys, extra top-level
fields, surrounding prose, and incomplete fences or think blocks. It does not
search for embedded JSON or repair model output.

`ACTION_PROTOCOL_INSTRUCTIONS` gives Qwen the two response shapes as a fixed
system instruction. Protocol evaluations use that instruction unchanged.

## Agent loop

`run_agent` starts a fresh conversation with the action protocol instructions
and a user prompt. A finish action returns its message. A tool action is passed
to a `ToolExecutor`, and its typed `ToolResult` is returned to the model as a
JSON observation before the next action. When the executor supplies tool
instructions, the loop includes the tool descriptions and JSON argument schemas
in the system message.

The loop applies these limits from its first version:

- At most 12 valid model actions are processed.
- An invalid response receives a correction asking for exactly one valid
  action. Two additional parse attempts are allowed for each step.
- Three consecutive failures of the same tool name and arguments stop the run.
  A successful call or a different failed action resets this count.
- The complete run has a 300-second timeout covering model and tool calls.

All limits can be reduced or extended through `run_agent` arguments except the
three-failure threshold, which is fixed. Tool implementations should enforce
their own operation timeout as well: once Jamie's total deadline expires, the
loop stops waiting for the active synchronous call but cannot forcibly terminate
that Python function.

## Read-only tools

`create_read_only_registry(root)` creates a registry containing exactly these
tools:

- `read_file` reads a bounded line range from one UTF-8 file.
- `list_directory` lists and sorts immediate directory entries.
- `find_files` recursively finds files using a glob pattern.
- `search_files` recursively searches UTF-8 files for literal text and reports
  matching paths and line numbers.

Each tool publishes its name, description, Pydantic input schema, and `READ`
permission value. The registry validates model arguments before execution
and converts expected validation and filesystem errors into failed tool
observations. Paths must be relative to the configured workspace root. Resolved
paths and symlink targets cannot leave that root. File matching skips symlinks,
and every tool applies a configurable result limit with a fixed upper bound.

The registry can be passed directly to the agent loop:

```python
from pathlib import Path

from assistant.agent import run_agent
from assistant.tools import create_read_only_registry

tools = create_read_only_registry(Path.cwd())
answer = run_agent(backend, tools, "What does README.md contain?")
```

## Permission system

Tool permissions come from a fixed application policy:

| Permission | Policy decision |
| --- | --- |
| `READ` | Allow |
| `SAFE_WRITE` | Confirm |
| `SENSITIVE` | Confirm |
| `DESTRUCTIVE` | Confirm |
| `EXTERNAL` | Confirm |

The default policy assigns the four registered filesystem tools to `READ`. It
also reserves classifications for future names such as `write_file`, `shell`,
`delete_file`, and `send_email`, although those tools are not registered or
implemented. A tool cannot register without a central policy entry, and its
declared permission must match that entry.

`ToolRegistry.execute` validates arguments, asks `PermissionPolicy` for an
allow-or-confirm decision, and calls an `ApprovalProvider` for confirmation
when required. The default provider denies every confirmation request. A user
interface can inject `CallbackApprovalProvider`; its request is bound to the
exact tool name, permission, and normalized arguments. Approval-like fields in
model arguments do not bypass validation or policy.

Audit recording starts before approval or execution. The default in-memory log
records the request and its completion, denial, rejection, or error. An optional
`JsonlAuditLog` flushes each event to disk. Records contain a SHA-256 digest of
the arguments instead of their raw values. If the initial audit write fails,
execution does not begin.

The model only produces an action. The policy and approval provider determine
whether Jamie may execute it.

## Model evaluations

With the local server running, establish a baseline with:

```sh
jamie eval model --runs 5
```

This runs 30 fixed cases five times each: eight structured-output cases, six
planning cases, six text instruction-following cases, and ten action-protocol
cases. The planning cases check small dependency-ordering tasks with a single
expected order. The action-protocol cases validate tool and finish responses;
they do not execute any actions.

Each round visits cases in case-ID order. Every request starts a fresh
conversation with the case's unchanged user prompt, temperature `0.6`, and a
`1024`-token limit. Protocol cases also receive the fixed protocol system
instruction described above. Other cases receive no additional instructions.
There are no JSON constraints, retries, warm-up prompts, or prompt adjustments.
Server-side sampling settings and caching remain as configured on the running
llama.cpp server.

The summary shows passes and failures, JSON validity, protocol parse success,
instruction following, median latency, output tokens, token-count coverage,
and total duration:

- **JSON validity:** complete response text must parse as JSON. Surrounding JSON
  whitespace is accepted; Markdown fences, explanatory text, duplicate keys,
  and non-finite numbers fail. This metric applies to the 14 JSON cases, so
  five runs produce 70 JSON attempts. Request failures count as failed parses.
- **Protocol parse success:** responses must parse as a `ToolAction` or
  `FinishAction`. The limited fence and think-block handling described above is
  allowed. This metric applies to the ten protocol cases, so five runs produce
  50 protocol attempts. Request failures count as failed parses.
- **Instruction following:** JSON responses must contain the expected fields,
  with matching types and values. Extra object fields are allowed; arrays must
  have the exact expected length and order. Protocol actions must exactly match
  the expected action, tool, arguments, or message. Text cases match exactly
  after removing surrounding whitespace. Parse failures and missing responses
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
jamie eval model --cases evals/cases/action_protocol.json --runs 5
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
For action checks, use `"format": "protocol"` and a complete tool or finish
action as the expectation. The CLI also accepts `--temperature`, `--max-tokens`,
and `--config`. Keep the cases and settings unchanged when comparing results.
Model outputs are stochastic, so repeated runs need not produce identical
results.

## Agent evaluations

With the local server running, evaluate the complete model, action loop, tools,
observations, and final answers with:

```sh
jamie eval agent --runs 5
```

The four fixed cases create isolated temporary workspaces and exercise
`read_file`, `list_directory`, `find_files`, and `search_files`. Each attempt
passes only when the expected tool is called, its successful result contains
the expected workspace content, and the final answer contains the expected
text. A model answer that happens to be correct without the required tool call
fails.

The summary reports passes, expected tool calls, correct tool outputs, answer
checks, median end-to-end latency, output tokens, and total duration. Reports
are checkpointed under `evals/results/` after each attempt and include the case
fixtures, tool arguments and observations, final answers, errors, timestamps,
model ID, raw model outputs, token usage, and a hash of the case definitions.
Temporary workspaces are removed after each attempt.

The [saved agent baseline](evals/results/agent-baseline.md) contains five runs
of each fixed case against Qwen3.8-9B-Distill Q4_K_M. All 20 attempts passed
the tool-call, tool-output, and final-answer checks.

Use a custom case file, run count, result directory, or per-run timeout when
needed:

```sh
jamie eval agent --cases evals/agent_cases/read_only_tools.json --runs 1
jamie eval agent --runs 5 --timeout 300 --output-dir evals/results
```

Agent evaluation cases define their workspace files and checks explicitly:

```json
{
  "id": "agent-read-001",
  "prompt": "What does README.md contain? Read the file before answering.",
  "files": {"README.md": "hello world\n"},
  "expected_tool": "read_file",
  "expected_args": {"path": "README.md"},
  "tool_output_contains": ["hello world"],
  "answer_contains": ["hello world"]
}
```

The [saved baseline](evals/results/baseline.md) contains all 100 attempts with
Qwen3.8-9B-Distill Q4_K_M. It used the earlier 120-second read timeout; the
current default is 600 seconds. The original results are preserved, with the
timeout and server settings recorded in a companion environment file. That run
predates the ten action-protocol cases and contains the original 20 cases.

The [action protocol baseline](evals/results/action-protocol-baseline.md)
contains 50 attempts with the ten protocol cases. It reached 100% protocol
parse success and 90% exact instruction following, with its server settings and
five exact-message failures recorded alongside the raw report.

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
- `src/assistant/agent/protocol.py`: validated tool and finish action parsing.
- `src/assistant/agent/loop.py`: bounded model actions and tool observations.
- `src/assistant/tools/`: the read-only tool contracts, implementations, and registry.
- `src/assistant/security/`: permission policy, trusted approvals, and audit records.
- `src/assistant/evaluation.py`: evaluation loading, scoring, metrics, and report saving.
- `src/assistant/agent_evaluation.py`: complete agent evaluation and scoring.
- `src/assistant/eval_cli.py`: evaluation command and terminal summaries.
- `evals/runner.py`: checkout entry point for the evaluation command.
- `evals/cases/`: fixed baseline case files, also bundled in installed wheels.
- `evals/agent_cases/`: isolated workspace and tool-use cases, also bundled in wheels.
- `evals/results/`: saved evaluation reports.
- `tests/`: CLI, configuration, transport, and evaluation behavior and failure cases.
- `assistant.toml`: minimal project configuration.
