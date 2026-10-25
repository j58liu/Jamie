"""Read and validate local project configuration without creating files."""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator


class GoogleConfig(BaseModel):
    """A Google Cloud "Desktop app" OAuth client used for Gmail and Calendar."""

    model_config = ConfigDict(extra="forbid", strict=True)

    client_secrets_path: str | None = None
    client_id: str | None = None
    client_secret: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.client_secrets_path or self.client_id)


class RemindersConfig(BaseModel):
    """Apple Reminders through the native EventKit helper."""

    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool = False
    helper_path: str | None = None


class ProjectConfig(BaseModel):
    """Local settings; unknown keys and incorrect types are rejected."""

    model_config = ConfigDict(extra="forbid", strict=True)

    color: bool = True
    memory_path: str | None = None
    timezone: str | None = None
    google: GoogleConfig = GoogleConfig()
    reminders: RemindersConfig = RemindersConfig()

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str | None) -> str | None:
        if value is None:
            return value
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Unknown timezone {value!r}.") from exc
        return value

    @field_validator("memory_path")
    @classmethod
    def validate_memory_path(cls, value: str | None) -> str | None:
        if value is not None and (not value.strip() or "\x00" in value):
            raise ValueError("Memory path must not be blank or contain null characters.")
        return value

    def resolved_memory_path(self) -> Path:
        if self.memory_path is not None:
            return Path(self.memory_path).expanduser()
        from assistant.memory import default_memory_path

        return default_memory_path()


class ConfigurationError(ValueError):
    """A configuration file could not be read or validated."""


def load_config(path: Path | None = None) -> ProjectConfig:
    """Load assistant.toml from the working directory, or use built-in defaults.

    An explicitly supplied path must exist. A missing default file is optional.
    """
    config_path = path if path is not None else Path("assistant.toml")
    try:
        with config_path.open("rb") as config_file:
            data = tomllib.load(config_file)
    except FileNotFoundError as exc:
        if path is None:
            return ProjectConfig()
        raise ConfigurationError(f"Configuration file not found: {config_path}") from exc
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"Cannot read configuration {config_path}: {exc}") from exc

    try:
        return ProjectConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid configuration in {config_path}: {exc}") from exc
