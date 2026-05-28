"""Initial meeting tables

Revision ID: 0001
Revises:
Create Date: 2026-05-27 00:00:00.000000

"""
from typing import Sequence, Union
import uuid
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "meetings",
        sa.Column("id",                UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("call_id",           sa.String(256),     nullable=False),
        sa.Column("title",             sa.String(512),     nullable=False, server_default="Untitled Meeting"),
        sa.Column("date",              sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_minutes",  sa.Integer,         nullable=True),
        sa.Column("participant_count", sa.Integer,         nullable=True),
        sa.Column("summary",           sa.Text,            nullable=True),
        sa.Column("decisions",         JSONB,              nullable=True),
        sa.Column("open_questions",    JSONB,              nullable=True),
        sa.Column("transcript_path",   sa.String(1024),    nullable=True),
        sa.Column("audio_path",        sa.String(1024),    nullable=True),
        sa.Column("status",            sa.String(32),      nullable=False, server_default="processing"),
        sa.Column("created_at",        sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_meetings_call_id", "meetings", ["call_id"], unique=True)

    op.create_table(
        "action_items",
        sa.Column("id",              UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("meeting_id",      UUID(as_uuid=True), sa.ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False),
        sa.Column("task",            sa.Text,            nullable=False),
        sa.Column("owner",           sa.String(256),     nullable=True),
        sa.Column("deadline",        sa.Date,            nullable=True),
        sa.Column("priority",        sa.String(32),      nullable=True, server_default="medium"),
        sa.Column("status",          sa.String(32),      nullable=False, server_default="open"),
        sa.Column("reminder_job_id", sa.String(256),     nullable=True),
        sa.Column("jira_ticket_id",  sa.String(128),     nullable=True),
        sa.Column("created_at",      sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_action_items_meeting_id", "action_items", ["meeting_id"])


def downgrade() -> None:
    op.drop_table("action_items")
    op.drop_table("meetings")
