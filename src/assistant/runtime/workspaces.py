"""Keep tools and reversible changes alive for each workspace in a session."""

from collections.abc import Callable
from pathlib import Path

from assistant.agent.loop import ToolExecutor
from assistant.tools.scopes import ToolScope, select_tools


class WorkspaceTools:
    """A workspace switch selects its own registry and retains earlier undo history."""

    def __init__(self, factory: Callable[[], ToolExecutor]) -> None:
        self._factory = factory
        self._registries: dict[Path, ToolExecutor] = {}

    def current(self, scope: ToolScope | str = ToolScope.ALL) -> ToolExecutor:
        root = Path.cwd().resolve()
        if root not in self._registries:
            self._registries[root] = self._factory()
        return select_tools(self._registries[root], scope)

    def close(self) -> None:
        for registry in self._registries.values():
            close = getattr(registry, "close", None)
            if close is not None:
                close()
        self._registries.clear()
