# Performance and usability results

Measured October 19, 2026 on this Apple M2 Mac (8 CPU cores, 16 GiB unified memory),
using llama.cpp build b10964-b29c606e2. Reports record model paths and SHA-256 hashes,
actual server settings/build, generation options, hardware, fixtures, task hashes,
code revision and working-tree differences. Models were already cached; no model
was downloaded. Independent GPU workloads were run serially for the comparisons below.

## Everyday tasks and tool context

The [initial baseline](before/benchmark-20261019T145625412210Z-c90d01ea.json) passed 18/18 trials across six tasks, with a
2.06-second median and
3.61-second p95. It used workspace tools and
preceded visible streaming, complete request budgeting, and selectable tool scopes.
It is a starting reference, not a controlled measurement of one isolated change.

With the same everyday cases and 512-token generation limit, [all tools](full-tools/benchmark-20261019T150922461666Z-65e9caec.json)
passed 13/18 trials; [file tools](file-tools/benchmark-20261019T151109831385Z-f9e8e581.json) passed 17/18. Median initial input fell
from 1,858 to 1,060 tokens (43%). Median task latency was 1.95 versus 1.74 seconds.
Local `/read`, `/ls`, `/find`, and `/search` also avoid model calls entirely; their
interaction tests verify both displayed results and the ordinary policy/audit path.

## Visible streaming

| Presentation | Passes | Median time to visible text | Median total task time |
| --- | ---: | ---: | ---: |
| [Buffered](buffered/benchmark-20261019T153055752115Z-b87f197c.json) | 3/3 | 5.80 s | 5.81 s |
| [Streamed](streamed/benchmark-20261019T153114710724Z-5933a20f.json) | 3/3 | 0.11 s | 6.57 s |

Both used the same generator-explanation prompt and 512-token limit. Streaming
improves when reading can begin; these trials do not show faster generation.
Answers varied between stochastic trials. Reasoning and action JSON are excluded
from visible deltas. Cancellation/incomplete-response tests verify that partial
answers are not saved and later turns recover.

Reproduce with `jamie eval benchmark --cases evals/streaming_cases --runs 3
--max-tokens 512 --no-stream`, then repeat with `--stream` (on one shell line).

## Long conversations

The [final long-session evaluation](context-final/benchmark-20261019T152959118162Z-11922b98.json) passed 3/3 trials, each with 12 turns,
a 4,096-token context and a 256-token answer reserve/generation limit. Every trial
exercised one automatic compaction and retained Heron, SQLite and October 3.
All actual requests were measured and budgeted before inference. Unit tests also
cover chunked summary requests, preserved current tool observations, recalled
memory, nested research requests, oversized prompts, and failure without history loss.

## Research retrieval and evidence

| Retrieval workers | Passes and supported fixture claims | Median search + retrieval | Median total task time |
| --- | ---: | ---: | ---: |
| [1](research-workers-1/research-20261019T152651531336Z-16825064.json) | 6/6 | 2.27 s | 9.62 s |
| [3](research-workers-3/research-20261019T152750288069Z-53e1ccff.json) | 6/6 | 1.01 s | 8.76 s |

The two fixed research cases were repeated three times per configuration, using
0.5 seconds of simulated latency per search/page request. Retrieval became about
55% faster and total task time about 9% shorter in these fixtures. These are
controlled network simulations with a real local model, not public-web guarantees.
Both configurations use the same synthesis, exact-quote/numeric checks and separate
model-assisted support review. Source IDs keep search order even when pages finish
out of order. Tests reject invented quotes, unsupported/unchecked paragraphs, and
unsupported numeric values. Semantic support judgments remain fallible.

## Inference settings

Every row below uses the same six task definitions, three runs per task,
file tools, thinking off, GPU layers enabled and Flash Attention on. The
[comparison manifest](inference/comparison.json) links full reports. IDs encode
model size, context and maximum answer tokens; `no-cache` sets saved prompt cache
RAM to zero. Other rows use 512 MiB of saved prompt cache RAM.

| Configuration | Passes | Median task time | p95 task time |
| --- | ---: | ---: | ---: |
| [2b-32k-1024](inference/2b-32k-1024/benchmark-20261019T152025692400Z-7ced4d58.json) | 16/18 | 1.82 s | 3.92 s |
| [2b-8k-1024](inference/2b-8k-1024/benchmark-20261019T152107847658Z-a2ddd87b.json) | 15/18 | 1.74 s | 3.46 s |
| [2b-4k-1024](inference/2b-4k-1024/benchmark-20261019T152146652359Z-e7f365ac.json) | 15/18 | 1.93 s | 3.66 s |
| [2b-8k-256](inference/2b-8k-256/benchmark-20261019T152228072710Z-af0e7099.json) | 17/18 | 2.09 s | 3.63 s |
| [2b-8k-512](inference/2b-8k-512/benchmark-20261019T152310815050Z-dd53de99.json) | 18/18 | 2.12 s | 4.39 s |
| [4b-8k-512](inference/4b-8k-512/benchmark-20261019T152358907813Z-57f0cace.json) | 18/18 | 4.24 s | 6.42 s |
| [2b-iq4-xs-8k-512](inference/2b-iq4-xs-8k-512/benchmark-20261019T152516987409Z-fc9ec29c.json) | 11/18 | 2.18 s | 5.72 s |
| [2b-8k-512-no-cache](inference/2b-8k-512-no-cache/benchmark-20261019T152603480134Z-66e54256.json) | 14/18 | 2.02 s | 2.90 s |

Qwen3.5-2B Q4_K_M remains the default. Its 8k/512 file-task configuration passed
18/18 at 2.12 seconds, versus 4.24 seconds for the 4B configuration's 18/18. IQ4_XS
lost task accuracy. The samples do not establish that a smaller context or a lower
token cap alone accelerates short answers. Retain the default 32k window for large
page/tool inputs. To try the measured file-task configuration, start the cached
model with `JAMIE_CONTEXT_SIZE=8192 ./scripts/start-local-model.sh`, then use
`/scope files` and `/tokens 512`. `JAMIE_MODEL_REPO` and `JAMIE_CACHE_RAM_MIB` expose
validated offline launcher choices. Start the model in a separate terminal as usual.

## Limits and excluded attempts

These are small stochastic samples on one Mac. Latency includes failures and
workspace/database setup; the server stays warm within each configuration, and
server/model loading is excluded. p95 uses nearest rank. A pass checks fixture
outcomes and failure handling, not general assistant intelligence. Reports retain
failures and their model output; no task expectations were weakened to improve scores.

`context/` retains the failed free-form summary experiment (2/3), which led to
validated excerpt selection. `context-excerpts/` retains the first extractive check;
those early context trials used a conservative reserve different from their output
cap, corrected in `context-final/`. `research-check/` retains citation-format
failures before structured synthesis; `research-structured/` records its first check.
`interrupted/` contains overlapping GPU attempts that were stopped and excluded.
The owned test server was stopped after each configuration; the user's server on
port 8080 was preserved.
