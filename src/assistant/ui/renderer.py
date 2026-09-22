"""Render completed answers for one-shot terminal commands."""

from rich.console import Console

from assistant.agent.events import AgentEvent, AssistantFinished


class OneShotRenderer:
    """Print the final answer once; command handlers retain error exit handling."""

    def __init__(self, console: Console, errors: Console) -> None:
        self._console = console
        self._errors = errors

    def handle(self, event: AgentEvent) -> None:
        if not isinstance(event, AssistantFinished):
            return
        self._console.print(event.text, markup=False, highlight=False)
        if event.finish_reason == "length":
            self._errors.print(
                "The response reached the token limit; use --max-tokens to allow a longer answer.",
                markup=False,
                highlight=False,
            )
