# Personal application connectors

Jamie can read Gmail, check Google Calendar, and manage Apple Reminders. Each provider sits behind a connector, so the agent loop never calls a provider API directly.

| Layer | Responsibility | Where |
| --- | --- | --- |
| Connector | Makes provider API calls and returns small typed models | `src/assistant/connectors/` |
| Connector tool | Exposes one typed operation to Jamie | `src/assistant/tools/integrations.py` |
| Registry | Validates tool arguments | `src/assistant/tools/registry.py` |
| Security policy | Decides whether the action may run | `src/assistant/security/policy.py` |
| Agent | Chooses the action | `src/assistant/agent/loop.py` |

Each connector reports a `ConnectorInfo`: its connection status, capabilities, and supported operations. Connector failures are typed (`NotConnectedError`, `AuthExpiredError`, `PermissionScopeError`, `RateLimitedError`, `NotFoundError`, `ProviderError`, `UnavailableError`). They are all `ToolError` subclasses, so the model gets a short, actionable message.

Replacing Gmail with another mail provider means writing a new connector with the same methods. The tools, the policy, and the agent stay the same.

## Tools and permissions

| Tool | Permission | Asks first? |
| --- | --- | --- |
| `connectors_status` | READ | no |
| `gmail_search_messages`, `gmail_get_message`, `gmail_get_thread`, `gmail_get_draft` | READ | no |
| `gmail_create_draft` | EXTERNAL | yes |
| `gmail_send_draft` | SEND | yes, **every time** (never "allow for session") |
| `calendar_list_events`, `calendar_check_availability`, `calendar_get_event` | READ | no |
| `calendar_create_event`, `calendar_update_event` | EXTERNAL | yes |
| `reminders_list` | READ | no |
| `reminders_create`, `reminders_complete` | SAFE_WRITE | yes |

Use `--scope personal` (or `/scope personal` in the REPL). The model then sees only the connector tools, which keeps the prompt small for the 2B model. This scope also raises the step limit to 16, because a reply workflow chains five or six calls.

Results are kept small:
- A search returns `id`, `thread_id`, `sender`, `subject`, `snippet`, and `received_at`.
- A message body has HTML and quoted history removed and is capped at 4,000 characters.
- A thread shows the full body of the latest message only.
- An availability check returns busy blocks and free slots, never event titles.

## Time handling

- Every datetime a tool accepts must include a timezone offset; naive values are rejected.
- The system prompt includes the current local time and zone, so the model can resolve "next Friday" before calling a tool. `assistant.connectors.timeutil.resolve_relative` provides the same resolution in code.
- Durations are added in elapsed time, so meetings across a DST change keep their length.
- All-day events report their inclusive last day. Google stores the end date as exclusive.

## Setting up Google (Gmail and Calendar)

1. In Google Cloud Console, create a project and enable the Gmail API and the Google Calendar API.
2. Configure the OAuth consent screen and add your account as a test user.
3. Create an OAuth client of type **Desktop app** and download its JSON file.
4. Add this to `assistant.toml`:

   ```toml
   timezone = "America/Toronto"   # optional; defaults to the system zone

   [google]
   client_secrets_path = "~/.config/jamie/google-client.json"
   ```
5. Connect with the narrowest access level that covers what you need:

   ```
   jamie connect google                          # read Gmail, read events, free/busy
   jamie connect google --scopes compose         # + create and send drafts
   jamie connect google --scopes calendar-write  # + create and update events
   ```

Sign-in uses the loopback redirect with PKCE. Only the refresh token and the granted scopes are stored, in the macOS login Keychain under `com.jamie.assistant.google`. Access tokens stay in memory. Run `jamie disconnect google` to revoke and delete the stored credentials.

Gmail read and compose scopes are **restricted** scopes. While your project is in testing mode, only listed test users can connect. Distributing Jamie publicly would require Google's verification and a security assessment.

## Setting up Apple Reminders

Reminders uses a small Swift helper built on EventKit (`native/reminders-helper`). Jamie runs it with a fixed command (`status`, `list`, `create`, or `complete`) and exchanges JSON over stdin and stdout. No AppleScript is generated.

```
scripts/build-reminders-helper.sh
```

```toml
[reminders]
enabled = true
# helper_path = "~/Library/Application Support/Jamie/bin/reminders-helper"  # the default
```

The first reminders request triggers the macOS Reminders permission prompt. If you deny it, re-enable access under System Settings > Privacy & Security > Reminders.

## Checking status

```
jamie connectors
jamie doctor
```

## Evaluating the reply workflow

`evals/agent_cases/personal_workflow.json` runs the target workflow ("find Alex's email, check my calendar, prepare a reply") against in-memory Gmail and Calendar fakes (`assistant.connectors.fakes`), with every approval granted. It checks the following:
- Jamie searched mail and checked availability.
- Jamie saved a draft.
- Jamie never called `gmail_send_draft`.
