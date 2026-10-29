# Persistent tasks

Jamie keeps track of unfinished work after the process exits. Tasks are stored apart from conversation memory, run on schedules or when new email arrives, pause for approval without blocking anything, and recover after a crash without repeating side effects.

## Lifecycle

```
scheduled ─▶ ready ─▶ running ─▶ completed
               ▲        │ ├────▶ failed      (after 3 attempts, or an expired approval)
               │        │ └────▶ cancelled
               │        ▼
               └── waiting_approval ──▶ cancelled (rejected)
```

`tasks/models.py` lists the allowed transitions; any other transition raises `InvalidTransition`. Writers state the state they expect, so a stale update fails rather than overwriting.

## Storage

`tasks.sqlite3` lives next to the memory database by default. Override it with `tasks_path` in `assistant.toml` or with `JAMIE_TASKS_PATH`. The file is created with user-only permissions and uses WAL mode with versioned migrations.

| Table | Holds |
| --- | --- |
| `tasks` | Instruction, state, priority, revision, lease, attempts, result |
| `task_steps` | One row per tool call: tool, status, operation key, and the argument *shape* (ids kept, content reduced to lengths) |
| `task_payloads` | Full arguments, results, and approval displays; deleted when the task finishes |
| `task_approvals` | Fingerprint, kind, status, task revision, and expiry |
| `schedules` | Kind, spec, IANA timezone, next run, missed-run policy, and event cursor |

## Execution

`TaskEngine.run_next()` does the following:
1. Claims a ready task with `BEGIN IMMEDIATE` and leases it to one worker. A heartbeat thread extends the lease while the task runs.
2. Rebuilds the conversation from the stored steps.
3. If the user approved an action, runs exactly that stored action first.
4. Runs the agent loop. Before each tool call it records a `started` step with a stable operation key (`task:revision:index`), then records the outcome afterwards.
5. Ends in one of three ways:
   - The task finishes and is marked `completed`, and the user is notified.
   - An action needs confirmation: the task becomes `waiting_approval`, and the user is notified.
   - An error happens: the task retries with backoff, and fails after three attempts.

The local model is contacted only when a task is ready. Checking schedules and approvals uses SQLite only.

## Approvals

An approval is bound to the tool, its normalized arguments, the account, the task, and the task revision, all through a SHA-256 fingerprint. Any change produces a different fingerprint, so an old approval no longer matches:
- a different recipient,
- different content,
- a different action,
- editing the task (`jamie tasks edit` increments the revision).

Approvals expire after 24 hours and can be used only once.

```
jamie tasks approvals          # see exactly what will run
jamie tasks approve <id> [--run]
jamie tasks reject <id>
```

The REPL offers `/tasks`, `/approvals`, `/approve`, and `/reject`, and shows waiting approvals at startup.

## Recovery and idempotency

A step left `started` means Jamie stopped between recording an operation and recording its result. Jamie never assumes an operation ran exactly once.

| Operation | What recovery does |
| --- | --- |
| Reads | Runs them again |
| `calendar_create_event` | The operation key becomes the Google event id. A retry returns the existing event (HTTP 409, or a lookup by id). |
| `gmail_create_draft` | The operation key becomes the draft's Message-ID. Recovery searches drafts by `rfc822msgid`. |
| `reminders_create` | Looks for a reminder with the same title |
| Everything else, including sending email | Marks the step ambiguous and asks the user whether to run it again |

## Scheduling

```
jamie tasks add "Summarize my inbox"                                  # now
jamie tasks add "Call the dentist" --at "tomorrow 9am"                # once
jamie tasks add "Morning brief" --every daily@08:00                   # daily
jamie tasks add "Weekly review" --every weekly:sun@18:00 --missed ask # weekly
jamie tasks add "Check build status" --every interval:2h              # interval
jamie tasks add "File new invoices" --on-email "subject:invoice"      # event
```

- Times are computed in the schedule's timezone, so 08:00 stays 08:00 across daylight-saving changes.
- Intervals use elapsed time.
- When the Mac was asleep for more than 15 minutes past a run, the missed-run policy applies:
  - `run_late` runs once now; several missed runs collapse into one.
  - `skip` waits for the next run.
  - `ask` creates the task and waits for approval.
- Email triggers poll Gmail with a cheap search. The first poll records a baseline, and later polls create a task only for new message ids.

## Background worker

```
jamie worker install     # launchd agent: one pass every 5 minutes and at login
jamie worker run --once  # one pass by hand
jamie worker uninstall
```

The launch agent (`~/Library/LaunchAgents/com.jamie.worker.plist`) runs `python -m assistant worker run --once` from the folder where you installed it, with that folder's configuration. If the local model is not running when a task is ready, the task waits, and Jamie sends a single notification.
