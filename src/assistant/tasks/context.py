"""The stable operation key of the step being executed, for provider idempotency."""

from contextvars import ContextVar

current_operation_key: ContextVar[str | None] = ContextVar("current_operation_key", default=None)
