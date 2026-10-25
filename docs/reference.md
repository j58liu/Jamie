# Jamie reference

The complete command, protocol, security, and evaluation reference. For an overview,
see the [README](../README.md). Design notes: [connectors](design/connectors.md),
[tasks](design/tasks.md), [personal memory](design/memory.md), and
[planning](design/planning.md).

## Workflow evaluations

```sh
jamie eval scenarios --runs 3 [--category planning|cross_app|routine|memory|safety]
jamie eval personal-memory
```

Scenario cases live in `evals/planning_cases/`, `evals/cross_app_cases/`,
`evals/personal_memory_cases/`, and `evals/safety_cases/`. Each case defines one or
more turns (each turn is a new session that shares memory), the approval policy,
fixture emails, events, reminders, and seeded memories, and injected connector
failures. It also defines checks: expected calls and arguments, required and
forbidden tools, answer text, drafts, sends, created events and reminders, memory
state, and whether writes were verified. The report scores routine accuracy,
workflow completion, approval violations, and duplicate external writes. The
deterministic memory cases in `personal_memory_cases/deterministic/` run without a
model.

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
[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B). For llama.cpp, use
[Unsloth's GGUF release](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF),
specifically `Qwen3.5-2B-Q4_K_M.gguf` (about 1.28 GB). The linked source model
contains Transformers weights; the GGUF release is the format llama.cpp loads.

Install llama.cpp on macOS with Homebrew:

```sh
brew install llama.cpp
```

The first setup needs internet access to install llama.cpp and download the
model. In a separate terminal, run the following command once to download the
selected GGUF into llama.cpp's model cache and start the server:

```sh
llama-server \
  -hf unsloth/Qwen3.5-2B-GGUF:Q4_K_M --no-mmproj \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 32768 --parallel 1 \
  --jinja --reasoning-format deepseek \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --temp 0.6 --top-p 0.95 --top-k 20
```

After that download finishes, stop the server with Ctrl+C. For subsequent
starts, use this launcher from the repository root:

```sh
./scripts/start-local-model.sh
```

The launcher uses the cached model with llama.cpp's `--offline` option, binds
only to `127.0.0.1:8080`, enables all GPU layers and Flash Attention, and prints
generation timings in the server terminal. Saved prompt caching stays enabled
with a 512 MiB RAM limit to reduce memory pressure on a 16 GB Mac; this limit is
separate from the active 32,768-token context. The launcher does not download
files. It requires a recent llama.cpp build that supports `--offline` and
`--cache-ram`. If you already have a GGUF file elsewhere, pass its path instead:

```sh
./scripts/start-local-model.sh "/path/to/Qwen3.5-2B-Q4_K_M.gguf"
```

With the model installed, local conversations and local file tools work without
internet access. Web research, web search, and fetching web pages still require
internet access. The cached GGUF is stored on disk and loaded into unified
memory for generation; moving it to an external drive does not speed up
generation. Close memory-heavy applications if macOS reports high memory
pressure. Compare the server's generation timing when measuring model speed;
Jamie's total turn duration also includes prompt processing and any tool work.

Use a recent llama.cpp build with support for the Qwen3.5 architecture.
The 32,768-token context and single request slot provide a starting configuration
for a 16 GB Mac; memory use also depends on other running applications.
The chat template setting disables thinking by default so short requests return
an answer without spending the token budget on reasoning. When thinking is
enabled, `--reasoning-format deepseek` separates reasoning from the final answer
in the HTTP response, so Jamie prints the answer text.

Keep the server running. Then, in the activated project environment:

```sh
jamie ask "hello"
jamie ask "Explain Python generators briefly." --temperature 0.6 --max-tokens 1024
jamie ask "Explain your answer to this puzzle: ..." --max-tokens 4096
```

Each invocation sends one user message to `/v1/chat/completions` and streams
the visible answer as plain text. Reasoning stays hidden. Cancelled or incomplete
answers are not saved as completed exchanges. The server chooses
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

## Interactive terminal

With the local server running, start a conversation with `jamie`. The terminal
shows a short welcome, activity while a turn runs, Markdown answers, tool
durations, and a status line with model, context, thinking mode, directory, and
Git branch when available. It keeps normal terminal scrollback.

Requests in the interactive terminal use Jamie's agent tools. For example,
`open Safari` can request the `open_application` tool. Jamie shows the action
and asks for approval when the Python permission policy requires it. Declining
ends the turn; Jamie does not retry or look for another way. Asking Jamie to
research a topic runs the `research` tool described under
[Web research](#web-research); its cited report is the answer.

Use `/mode chat` for streamed conversation without tools, and `/mode task` for
actions. `/scope files`, `/scope read`, `/scope web`, `/scope macos`,
`/scope personal`, or `/scope all` selects the tools supplied to the model in task mode. The status
line shows the active mode and scope. These settings keep the same Python
permission checks. One-shot tasks accept the same choice with
`jamie agent "Read README.md" --scope read`.

`/read <path>`, `/ls [path]`, `/find <pattern>`, and `/search <text>` display
workspace results immediately without calling the model. They use the same
validated, audited file tools. Use a normal task prompt when you want the model
to interpret the file contents. `/tokens 256` changes the session's maximum
answer length; `/tokens` shows the current value.

Before inference, the runtime measures the complete request, including tool
instructions, recalled memory, the current prompt, and tool observations. It
reserves space for the requested answer. When older history no longer fits,
automatic compaction selects original conversation excerpts by validated IDs;
it preserves recent exchanges when space permits and keeps the current task's
observations. An oversized current request produces an error before inference.
Compaction failures leave the conversation unchanged. `/context` reports the
last complete request and its answer reserve after a turn; before a turn it
reports the conversation's base cost.

`/compact` asks the model to summarize the conversation and continues from that
summary, which frees context for long sessions. It runs like a turn: Ctrl+C
cancels it and leaves the conversation unchanged. The saved conversation keeps
every exchange, so `/resume` restores the full history rather than the summary.

Enter sends a prompt. Press Escape then Enter to insert a line break. Tab
completes slash commands and paths (including directories after `/cwd`); a dim
preview describes a matching slash command as you type. Use `/help` to list
commands, Ctrl+C to cancel a running turn, and Ctrl+D or `/quit` to leave.

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

During agent turns, Jamie also sends the two shapes as a JSON schema
(`response_format`), with `tool` limited to the registered tool names.
llama.cpp turns the schema into a sampling grammar, so the model cannot write
prose around an action; without it, Qwen3.5-2B usually starts with a sentence
before the JSON. With thinking on, the constraint applies after the reasoning.
The parser still validates every reply, and model calls made inside tools, such
as the research workflow's, are not constrained.

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
- If the same tool call (name and arguments) gives the same result a second
  time in a turn, even with other calls in between, the model is told it
  already tried it. A third time stops the run.
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
- `find_files` recursively finds files and folders (shown with a trailing `/`)
  using a glob pattern.
- `search_files` recursively searches UTF-8 files for literal text and reports
  matching paths and line numbers.

`find_files` and `search_files` skip hidden folders such as `.git` and `.venv`,
which would otherwise fill the results; hidden files are still found. A missing
path error suggests `find_files`, so the model can search for the name instead
of guessing another path.

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
| `SEND` | Confirm every time (never for the whole session) |

The default policy assigns filesystem reads and cached page extraction to
`READ`; the five reversible editing tools, application opening, Finder reveal,
and clipboard writing to `SAFE_WRITE`; clipboard reading, Shortcuts, and
AppleScript to `SENSITIVE`; and web search and page retrieval to `EXTERNAL`.
Connector reads are `READ`; Gmail drafts and calendar changes are `EXTERNAL`;
sending a Gmail draft is `SEND`; reminder changes are `SAFE_WRITE`.
It reserves classifications for unimplemented names such as `shell` and
`delete_file`. A tool cannot register without a central
policy entry, and its declared permission must match that entry.

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

## Editing, diffs, and undo

`create_workspace_registry(root, approvals=...)` adds five approved editing
tools to the four read tools:

- `write_file` creates or replaces a UTF-8 text file.
- `edit_file` replaces exact text, rejecting ambiguous matches unless
  `replace_all` is explicitly set.
- `move_file` moves a file or directory to an unused workspace path.
- `create_directory` creates one directory, with optional parent creation.
- `trash_file` removes a file or directory while retaining its turn snapshot.

All five tools are classified as `SAFE_WRITE`, so the default approval provider
denies them. A trusted user interface must approve the exact normalized
arguments before snapshots or filesystem changes occur. Paths stay within the
workspace, editing through symbolic links is rejected, and no shell command is
used.

The registry groups changes made by one `run_agent` call into one turn. Each
affected path is copied to temporary snapshot storage before its first change
in that turn. Every successful editing result includes the cumulative unified
diff, for example:

```diff
-    return a - b
+    return a + b
```

The harness handles two slash commands directly, without sending them to the
model:

- `/diff` returns the active turn's diff, or the most recently completed turn's
  diff.
- `/undo` restores every snapshot from the most recently completed changed
  turn. Multiple edits to one path restore the content from before its first
  edit in that turn.

After a successful edit, Jamie tracks a relevant verification check. A changed
file must be read with `read_file` or matched with `search_files`. Directory
creation, moves, and trash operations require a relevant `list_directory` call.
If the model attempts to finish while checks remain, the harness rejects that
finish action and tells the model exactly which read operations are still
needed. A later edit reinstates the corresponding check.

Snapshots last for the lifetime of the workspace registry and are stored
outside the workspace. Undo and diff commands are audited; entering `/undo` is
the user's explicit approval for that restoration.

Interactive sessions retain a separate registry for each resolved working directory.
`/diff` and `/undo` apply to the current directory's latest changed turn, including
after later chat turns. `/cwd` switches registries; returning to a directory restores
its undo history. `/new` starts a conversation without discarding workspace changes.
Temporary snapshots are removed when the terminal session exits.

## macOS tools

`create_macos_registry(root, approvals=...)` adds six macOS tools to the
workspace registry:

- `open_application` runs `open -g -j -a` with an application name, so the app
  launches hidden and stays behind the app you are using. An app that is already
  running is not hidden. Jamie cannot see or control an app's windows.
- `reveal_in_finder` runs `open -R` for an existing workspace path.
- `clipboard_read` reads text with `pbpaste`.
- `clipboard_write` replaces clipboard text with `pbcopy`.
- `run_shortcut` runs a named shortcut with `shortcuts run`.
- `run_applescript` runs supplied AppleScript with `osascript -e`.

The tools pass fixed argument lists directly to the macOS commands without a
command shell. Finder paths must remain inside the configured workspace.
Application opening, Finder reveal, and clipboard writing require
`SAFE_WRITE` approval. Clipboard reading, Shortcuts, and AppleScript require
`SENSITIVE` approval because they can expose data or perform broader
automation. The terminal shows the exact tool, permission, and arguments and
defaults to denying the request.

Use the agent command for model-directed tools:

```sh
jamie "open Safari"
```

Jamie sends the task to Qwen, validates its action, applies the permission
policy, and asks for approval before running the macOS command. `jamie ask`
continues to send a prompt directly to the model without tools. The explicit
form `jamie agent "open Safari"` is also available. No screenshot or
screen-capture tool is included.

## Web research

Jamie provides these web tools without browser automation:

- `web_search` queries DuckDuckGo's HTML search endpoint and returns bounded
  titles, URLs, and snippets.
- `fetch_url` retrieves one public HTTP or HTTPS text page, keeps it in an
  in-memory session cache, and returns it as Markdown: headings, lists, links
  (made absolute), code blocks, and table rows. Scripts, styles, and the page
  title markup are dropped, and a page's `<main>` or `<article>` is preferred
  when it has enough text. The model receives at most 24,000 characters, about
  6k tokens of the 32k context, with `truncated` set when a page is longer.
- `extract_page` returns the same Markdown for a page already in the cache.
- `research` runs the complete research sequence below as one action and
  returns its cited report, which ends the turn. It is available where Jamie
  has a model backend: the interactive terminal and `jamie agent`.

Search, retrieval, and research are `EXTERNAL` operations. `web_search` runs
without asking: it sends only the model-written query to DuckDuckGo, which can
include text from the conversation or from files Jamie read. `fetch_url` and
`research` require approval, and approving `research` covers the searches and
page reads inside it. Extraction reads only the session cache and is classified
as `READ`.
Retrieval does not use environment proxies or automatic redirects. It validates
every redirect, rejects credentials and private, loopback, link-local, or
otherwise non-public addresses, restricts content types, and limits downloaded
and extracted content. A fetched page must exist in the same session before it
can be extracted.

The higher-level command performs the complete sequence directly:

```sh
jamie research "What evidence supports this claim?"
```

Qwen first generates up to three searches. Jamie runs independent searches and
page retrievals with up to three workers, retaining search order for source IDs
and retrieving up to five distinct pages. It then cross-checks the evidence and
produces structured answer paragraphs. Python renders their source IDs as links.
A separate model review checks that the cited excerpts support each paragraph;
Python requires exact supporting quotes from those excerpts and checks numeric
claims against the quotes. Invalid citations or evidence receive one correction
attempt. These checks reduce unsupported claims; the model's support judgment
is still fallible. A successful answer cites at least two sources. Use
`--max-sources` to select a limit from two through eight.

Research results record search, retrieval, model, and total durations. The
fixture evaluation can compare serial and concurrent retrieval with a fixed
simulated network delay:

```sh
jamie eval research --runs 3 --workers 1 --fixture-delay 0.5
jamie eval research --runs 3 --workers 3 --fixture-delay 0.5
```

## Everyday performance benchmarks

```sh
jamie eval benchmark --runs 3 --scope files
jamie eval benchmark --runs 3 --scope all
python scripts/benchmark-local-models.py --runs 3
```

The everyday suite covers file discovery, document questions, verified edits,
remembered decisions, failed-file recovery, and declined approval. Reports include
task success, median and p95 time, time to visible text, model calls, protocol
retries, approval prompts, input/output usage, fixed fixtures and their hashes,
model file/hash, server build/settings, generation options, and hardware. Failed
trials remain in the results. Timing covers warm-server use; each trial has a fresh
temporary workspace and SQLite database. The model comparison uses cached files
offline, starts one test server at a time on port 8081, and stops only that server.
Run GPU benchmarks serially. See [the implementation record](design/performance-improvements.md)
for recorded comparisons and limitations.

## Persistent memory

Jamie stores completed local conversations in SQLite. On macOS the default
database is:

```text
~/Library/Application Support/Jamie/memory.sqlite3
```

The schema contains `sessions`, `conversations`, `messages`, `tool_calls`,
`memories`, and `sources`. Research sessions also retain their
retrieved source records. Tool arguments use the same bounded redaction as
audit records, and tool output is stored only as a bounded preview. The database
file is created with user-only permissions.

Completed exchanges are added to an SQLite FTS5 index. Before an `ask` or agent
request, Jamie searches that index with the current prompt, retrieves up to five
matching memories, and supplies bounded excerpts to Qwen in a separate system
section. The section labels recalled text as untrusted background so an old
conversation cannot override current instructions. For example, after a
conversation that selected SQLite:

```sh
jamie "What was the database library we discussed?"
```

Jamie can retrieve the earlier exchange and answer from that context. Memory is
text-only and local; no embedding model or vector database is used.

Set a different database path in `assistant.toml` when needed:

```toml
memory_path = "state/memory.sqlite3"
```

The `JAMIE_MEMORY_PATH` environment variable provides a process-level override.
Configuration loading and `jamie doctor` validate the path setting without
creating the database.

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
JSON report includes the exact case definitions,
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

The [saved baseline](../evals/results/baseline.md) contains all 100 attempts with
Qwen3.8-9B-Distill Q4_K_M. It used the earlier 120-second read timeout; the
current default is 600 seconds. The original results are preserved, with the
timeout and server settings recorded in a companion environment file. That run
predates the ten action-protocol cases and contains the original 20 cases.

The [action protocol baseline](../evals/results/action-protocol-baseline.md)
contains 50 attempts with the ten protocol cases. It reached 100% protocol
parse success and 90% exact instruction following, with its server settings and
five exact-message failures recorded alongside the raw report.

## Agent evaluations

With the local server running, evaluate the complete model, action loop, tools,
observations, and final answers with:

```sh
jamie eval agent --runs 5
```

The read-only cases create isolated temporary workspaces and exercise
`read_file`, `list_directory`, `find_files`, and `search_files`. The personal
workflow case (`evals/agent_cases/personal_workflow.json`) runs "find Alex's email,
check my calendar, and prepare a reply" against in-memory Gmail and Calendar, and
also checks required and forbidden tools. Each attempt
passes only when the expected tool is called, its successful result contains
the expected workspace content, and the final answer contains the expected
text. A model answer that happens to be correct without the required tool call
fails.

The summary reports passes, expected tool calls, correct tool outputs, answer
checks, median end-to-end latency, output tokens, and total duration. Reports
are checkpointed under `evals/results/` after each attempt and include the case
fixtures, tool arguments and observations, final answers, errors, timestamps,
model ID, raw model outputs, token usage, and the case definitions.
Temporary workspaces are removed after each attempt.

The [saved agent baseline](../evals/results/agent-baseline.md) contains five runs
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

## Research evaluations

With the local server running, evaluate the complete research sequence against
fixed search results and page fixtures:

```sh
jamie eval research --runs 5
```

The fixtures keep retrieval repeatable while Qwen still generates searches,
cross-checks the supplied pages, and writes the cited answer. Each attempt
reports the number of cited sources and checks that citations exist, resolve to
successfully extracted pages, contain no URLs outside the search results, cover
multiple source hosts, and support the expected claims. Claim support requires
the expected answer text, a citation on that answer line, and the expected
evidence in the cited page fixture.

Reports are checkpointed under `evals/results/` and contain the exact fixtures,
generated searches, extracted source URLs, cross-check notes, raw
model outputs, answers, cited URLs, individual checks, failures, model
information, token usage, and timing. Custom cases can be selected with
`--cases`:

```sh
jamie eval research --cases evals/research_cases/supported_claims.json --runs 1
```

## Memory evaluations

Run the fixed FTS5 retrieval cases without contacting the model:

```sh
jamie eval memory
```

Each case seeds an isolated temporary database, performs one natural-language
query, and checks whether the expected memory appears within its configured
top-k results. The report includes recall@k, mean reciprocal rank, retrieved
text, and case definitions. This provides a baseline for deciding
whether text retrieval is sufficient before considering embeddings.

The [saved FTS5 baseline](../evals/results/memory-fts-baseline.md) passed all five
cases with 100% recall@k and a mean reciprocal rank of 1.000. Embeddings remain
deferred until broader evaluations show a retrieval gap.

## Configuration and diagnostics

`jamie doctor` checks Python 3.12+, the configuration, the inference backend and
model files, the application data directory, macOS tools, each configured
connector, persistent tasks, and the background worker. It exits with code `1`
when a check fails. Apart from asking the local llama.cpp server for its
properties, it performs no network requests and creates no files.

Configuration is read from `assistant.toml` in the current working directory.
If that file is absent, built-in defaults are used:

```toml
color = true
```

Set `color = false` for uncolored output. `memory_path` and `tasks_path` are
optional and default to Jamie's macOS application-support directory
(`JAMIE_MEMORY_PATH` and `JAMIE_TASKS_PATH` override them). `timezone`, `[google]`,
and `[reminders]` configure the connectors. Unknown fields, incorrect types
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
- `src/assistant/runtime/application.py`: reusable async direct-chat lifecycle.
- `src/assistant/llm/base.py`: `Message`, `LLMResponse`, `LLMError`, and `LLMBackend`.
- `src/assistant/llm/llamacpp.py`: llama.cpp HTTP transport.
- `src/assistant/agent/protocol.py`: validated tool and finish action parsing.
- `src/assistant/agent/loop.py`: bounded model actions and tool observations.
- `src/assistant/agent/events.py`: typed turn, tool, approval, and answer events.
- `src/assistant/ui/renderer.py`: one-shot answer rendering from runtime events.
- `src/assistant/tools/`: filesystem, editing, macOS, web, and connector tools plus the registry.
- `src/assistant/connectors/`: Gmail, Google Calendar, and Apple Reminders connectors.
- `src/assistant/tasks/`: persistent tasks, scheduler, deferred approvals, recovery, worker.
- `src/assistant/tasks_cli.py`: `jamie tasks` and `jamie worker` commands.
- `native/reminders-helper/`: the EventKit helper for Apple Reminders.
- `src/assistant/security/`: permission policy, trusted approvals, and audit records.
- `src/assistant/research.py`: bounded search, retrieval, cross-checking, and cited synthesis.
- `src/assistant/memory.py`: SQLite schema, FTS5 retrieval, and conversation storage.
- `src/assistant/evaluation.py`: evaluation loading, scoring, metrics, and report saving.
- `src/assistant/agent_evaluation.py`: complete agent evaluation and scoring.
- `src/assistant/research_evaluation.py`: citation, source, and claim-support evaluation.
- `src/assistant/memory_evaluation.py`: deterministic FTS5 retrieval evaluation.
- `src/assistant/eval_cli.py`: evaluation command and terminal summaries.
- `evals/runner.py`: checkout entry point for the evaluation command.
- `evals/cases/`: fixed baseline case files, also bundled in installed wheels.
- `evals/agent_cases/`: isolated workspace and tool-use cases, also bundled in wheels.
- `evals/research_cases/`: fixed research sources and expected claims, also bundled in wheels.
- `evals/memory_cases/`: fixed memory retrieval cases, also bundled in wheels.
- `evals/results/`: saved evaluation reports.
- `tests/`: CLI, configuration, transport, and evaluation behavior and failure cases.
- `assistant.toml`: minimal project configuration.
