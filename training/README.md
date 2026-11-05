# Fine-tuning Jamie's local model

Jamie runs Qwen3.5-4B. Fine-tuning does not turn it into a frontier model, but it does
make it much more reliable at the things Jamie asks of it: picking the right tool,
writing valid tool calls, reading results correctly, finishing with a grounded answer,
routing messages, turning plain words into schedules, and saying "unknown" instead of
inventing answers on job applications.

## What it trains on

`python -m assistant.training.dataset` builds the data on your Mac in under a minute.
It creates small random "worlds" (a home folder with documents, an inbox, a calendar,
reminders, background tasks, an applicant profile) and runs scripted everyday requests
through **Jamie's real tool registry**: the same system prompt, tool list, tool output,
and action format Jamie uses at run time. Only the model's own replies are scripted,
and a trajectory is dropped if any tool call fails unexpectedly.

| Part | Examples |
| --- | --- |
| Tool use | finding and reading files anywhere on the Mac, recent files, copying and organizing, reading email, checking the calendar, drafting replies, reminders, web lookups, browsing and booking, Jamie's own tasks and schedules, the job profile |
| Safety | ignoring instructions inside emails, recovering from a wrong path |
| Routing | which tools a message needs, and now vs background vs schedule (one letter) |
| Scheduling | plain words to a schedule form |
| Job answers | answering only from the profile and résumé, otherwise `UNKNOWN` |

None of it reuses the evaluation fixtures in `evals/`, so `jamie eval` stays a fair
before-and-after test.

## Run it

On a Mac with Apple silicon and 16 GB of memory:

```sh
brew install llama.cpp          # once
source .venv/bin/activate
training/train.sh --quick       # 60 steps to check the whole pipeline (~30 min, mostly downloads)
training/train.sh               # the real run (about 2-3 hours on an M2)
```

The script:

1. builds the data set,
2. trains a LoRA adapter on the 4-bit model with MLX (QLoRA; settings in `lora.yaml`),
3. merges the adapter into the original full-precision weights (`merge_lora.py`),
4. converts to GGUF with llama.cpp and quantizes to Q4_K_M,
5. installs it as `~/Library/Application Support/Jamie/models/jamie-4b.gguf`.

`scripts/start-local-model.sh` (which Jamie.app uses) starts that file automatically
when it exists. Delete it to go back to the original model; the previous fine-tune is
kept next to it as `jamie-4b.previous.gguf`.

## Measure it

Run the same evaluations before and after:

```sh
jamie eval scenarios --runs 3
jamie eval agent --runs 5
```

## Settings

| Variable | Default | |
| --- | --- | --- |
| `JAMIE_TRAIN_WORLDS` | 20 | more worlds, more examples, longer training |
| `JAMIE_TRAIN_MODEL` | `mlx-community/Qwen3.5-4B-4bit` | model trained (4-bit) |
| `JAMIE_BASE_MODEL` | `Qwen/Qwen3.5-4B` | original weights the adapter is merged into |
| `JAMIE_TRAIN_DIR` | `training/work` | where data, adapter, and models are written (~25 GB) |

To train for more steps, edit `iters` (and the cosine schedule's step count) in
`lora.yaml`. If training runs out of memory, lower `max_seq_length` to 4096; the few
examples longer than that (every tool listed at once) will be cut.
