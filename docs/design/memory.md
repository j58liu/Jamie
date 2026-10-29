# Personal memory

Jamie keeps two kinds of memory in `memory.sqlite3`:

| Kind | Holds | How it is found |
| --- | --- | --- |
| Conversation memory | Earlier user and assistant messages | FTS5 search over past exchanges |
| Personal memory | Typed facts about the user, with evidence | FTS5 plus filters, recency, and authority |

Commitments, such as "I promised to reply to Sam", are recognized but not stored as facts. Ask Jamie to create a reminder, or use `/schedule`, to act on one.

## Records

| Type | Example |
| --- | --- |
| `profile` | Working hours are 9am to 5pm |
| `preference` | Prefers morning meetings |
| `relationship` | Alex Chen is a project collaborator (also linked to the entity "Alex Chen") |
| `project` | Project Jamie has a release milestone |
| `knowledge` | A fact from a document, with a source reference |

Every record stores:
- the subject, key, value, and scope,
- its source and confidence,
- verbatim evidence and a reference to where it came from,
- when it was created and updated, and when it expires (optional),
- its status and the record it superseded, so corrections stay traceable,
- a mention count (repeats are merged, not duplicated).

## Authority

Authority comes from how a fact was learned, never from how confident the model sounds:

| Source | Authority |
| --- | --- |
| `user_correction` | 4 |
| `explicit_user_statement` | 3 |
| `verified_record` | 2 |
| `inference` | 1 (confidence capped at 0.5) |

When a new value conflicts with an active record in the same slot (same type, subject, key, and scope):
- If its authority is equal or higher, it supersedes the old record.
- Otherwise it is stored as `disputed` and is not used as fact. `jamie memory conflicts` lists these.

## Extraction

After a turn, Jamie may learn from **the user's own message only**. It never reads tool output, emails, or its own replies for this. Two steps keep it cheap and safe:
1. A keyword filter ("I prefer", "remember", "my manager", "from now on", and similar) decides whether to spend a model call at all.
2. The model classifies candidates as an explicit statement, inference, third-party statement, suggestion, temporary context, or commitment. Python then rejects:
   - evidence that is not the user's verbatim words,
   - text inside quotation marks or `>` quotes,
   - reported speech ("Alex said…"),
   - suggestions and temporary context.

Extraction is controlled by `[memory] extraction = "cue" | "off"` in `assistant.toml`.

## Retrieval

`recall()` uses FTS5 relevance, optionally narrowed by person, project, type, minimum source, or time. It then:
- weights by recency (90-day half-life) and authority,
- drops duplicates,
- leaves out expired, superseded, disputed, and forgotten records.

Only matching records are returned, so unrelated private facts stay out of the prompt. The prompt section is limited by `[memory] personal_context_tokens` (default 400), and every line names its type, source, date, and record id.

Embeddings are deliberately absent. Add them only if the evaluations show keyword search missing important queries.

## Commands

```
jamie memory list [--type preference] [--all]
jamie memory search "meetings" [--person Alex] [--stated-only]
jamie memory add preference meeting_time morning
jamie memory correct 12 afternoon      # wins over everything, keeps history
jamie memory forget 12                 # erases the value and evidence
jamie memory show 12                   # evidence and correction history
jamie memory conflicts
jamie memory people
```
