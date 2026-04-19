"""
migrations/env.py — Alembic environment configuration.

Uses the SYNCHRONOUS SQLite driver for migrations.
The async sqlite+aiosqlite driver is only for runtime (src/db.py).

Key design decisions:
- Imports all SQLModel models so SQLModel.metadata is fully populated
  before Alembic uses target_metadata.
- Strips the async driver prefix if DATABASE_URL env var is set.
- sys.path is extended so `from src.db import ...` resolves from project root.
"""
from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlmodel import SQLModel

# ---------------------------------------------------------------------------
# Make the project root importable from the migrations/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import ALL models so SQLModel.metadata is fully populated.
# If a model is not imported here, its table will be invisible to Alembic.
from src.db import AgentLog, EmailAction, Event, PendingAction  # noqa: F401 E402

# ---------------------------------------------------------------------------
# Alembic Config object
# ---------------------------------------------------------------------------
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# This is the metadata object Alembic inspects for autogenerate support.
target_metadata = SQLModel.metadata


def get_url() -> str:
    """
    Return synchronous SQLite URL for migrations.

    Priority: DATABASE_URL env var → alembic.ini sqlalchemy.url
    Strips the async 'sqlite+aiosqlite' prefix to 'sqlite'.
    """
    url = os.getenv("DATABASE_URL") or config.get_main_option(
        "sqlalchemy.url", "sqlite:///./agent_memory.db"
    )
    # Strip async driver prefix — alembic requires the synchronous driver
    return url.replace("sqlite+aiosqlite", "sqlite")


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (emit SQL to stdout)."""
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB connection."""
    cfg = dict(config.get_section(config.config_ini_section) or {})
    cfg["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Required for SQLite: render AS NULL for nullable columns in comparisons
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
