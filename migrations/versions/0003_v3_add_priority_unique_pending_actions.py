"""
Migration 0003 — v3 (Hardened): Add priority + UNIQUE(external_id) to events;
                                 Create pending_actions table

This migration represents the v3.0 hardening improvements:
- priority column on events enables proactive reschedule suggestions
- UNIQUE index on external_id prevents duplicate contest rows at the DB level
  (semantic deduplication in AggregatingContestProvider is the first line of
  defence; this constraint catches anything that slips through)
- pending_actions table supports the Human-in-the-Loop (HITL) confirmation flow

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-14 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Add priority column to events (backfill existing rows to 0)
    # ------------------------------------------------------------------
    with op.batch_alter_table("event") as batch_op:
        batch_op.add_column(
            sa.Column(
                "priority",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )
        # Unique index on external_id — the DB-level deduplication guard
        # Design Document §6.1: "Even if semantic deduplication misses a duplicate,
        # the DB write will fail cleanly with an IntegrityError"
        batch_op.create_index(
            "uq_event_external_id",
            ["external_id"],
            unique=True,
        )

    # ------------------------------------------------------------------
    # Create pending_actions table (new in v3.0)
    # Stores HITL actions awaiting user confirm/undo
    # ------------------------------------------------------------------
    op.create_table(
        "pendingaction",
        sa.Column("id", sa.String(), nullable=False),           # UUID string
        sa.Column("action_type", sa.String(), nullable=False),  # "schedule_event" | "delete_email"
        sa.Column("payload", sa.String(), nullable=False),      # JSON-serialised parameters
        sa.Column("description", sa.String(), nullable=False),  # Human-readable for UI
        sa.Column(
            "status",
            sa.String(),
            nullable=False,
            server_default="REQUIRES_REVIEW",                   # → CONFIRMED | CANCELLED | EXPIRED
        ),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("pendingaction")

    with op.batch_alter_table("event") as batch_op:
        batch_op.drop_index("uq_event_external_id")
        batch_op.drop_column("priority")
