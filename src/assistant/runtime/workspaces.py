"""Keep tools and reversible changes alive for each workspace in a session."""

from collections.abc import Callable
from pathlib import Path

from assistant.agent.loop import ToolExecutor


class WorkspaceTools:
    """A workspace switch selects its own registry and retains earlier undo history."""

    def __init__(self, factory: Callable[[], ToolExecutor]) -> None:
        self._factory = factory
        self._registries: dict[Path, ToolExecutor] = {}

    def current(self) -> ToolExecutor:
        root = Path.cwd().resolve()
        if root not in self._registries:
            self._registries[root] = self._factory()
        return self._registries[root]

    def close(self) -> None:
        for registry in self._registries.values():
            close = getattr(registry, "close", None)
            if close is not None:
                close()
        self._registries.clear()
