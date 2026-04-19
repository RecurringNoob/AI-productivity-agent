"""
Migration 0002 — v2: Add end_time, source, external_id to events;
                      Create email_actions and agent_logs tables

This migration represents the v2.0 improvements:
- Events now track end_time and source (contest | meeting | manual)
- external_id enables external contest IDs to be stored
- email_actions logs all email triage decisions
- agent_logs records every agent interaction for observability

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-14 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Extend the events table with v2 columns
    # Note: SQLite only allows nullable columns (or columns with a
    # server_default) to be added via ALTER TABLE — non-nullable additions
    # require batch migrations (render_as_batch=True in env.py).
    # ------------------------------------------------------------------
    with op.batch_alter_table("event") as batch_op:
        batch_op.add_column(
            sa.Column("end_time", sa.DateTime(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("source", sa.String(), nullable=True, server_default="")
        )
        batch_op.add_column(
            sa.Column("external_id", sa.String(), nullable=True)
        )

    # ------------------------------------------------------------------
    # Create email_actions table
    # Tracks every email triage decision made by the email agent
    # ------------------------------------------------------------------
    op.create_table(
        "emailaction",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email_id", sa.String(), nullable=False),
        sa.Column("sender", sa.String(), nullable=False),
        sa.Column("subject", sa.String(), nullable=False),
        sa.Column("intent", sa.String(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # ------------------------------------------------------------------
    # Create agent_logs table
    # Audit log for all agent interactions (used by GET /agent/history)
    # ------------------------------------------------------------------
    op.create_table(
        "agentlog",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("user_input", sa.String(), nullable=False),
        sa.Column("agent", sa.String(), nullable=False),
        sa.Column("outcome", sa.String(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("agentlog")
    op.drop_table("emailaction")

    with op.batch_alter_table("event") as batch_op:
        batch_op.drop_column("external_id")
        batch_op.drop_column("source")
        batch_op.drop_column("end_time")
