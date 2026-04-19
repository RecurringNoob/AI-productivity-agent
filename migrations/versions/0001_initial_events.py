"""
Migration 0001 — Create initial events table (v1 schema)

This represents the v1 schema: a minimal events table with only
id, title, start_time, and created_at. Columns added in v2 and v3
are applied in subsequent migrations.

Revision ID: 0001
Revises: (none — initial migration)
Create Date: 2026-04-14 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the events table with v1 schema (id, title, start_time, created_at)."""
    op.create_table(
        "event",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("start_time", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Drop the events table."""
    op.drop_table("event")
