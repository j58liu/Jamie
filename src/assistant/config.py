"""Read and validate local project configuration without creating files."""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError


class ProjectConfig(BaseModel):
    """Local settings; unknown keys and incorrect types are rejected."""

    model_config = ConfigDict(extra="forbid", strict=True)

    color: bool = True


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
