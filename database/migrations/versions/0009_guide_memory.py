"""tourflow ai guide conversation memory

Revision ID: 0009_guide_memory
Revises: 0008_traveler_auth_ownership
Create Date: 2026-09-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0009_guide_memory"
down_revision: Union[str, None] = "0008_traveler_auth_ownership"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "guide_messages",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("trip_id", sa.String(length=36), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["trip_id"], ["trips.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_guide_messages_user_trip_created", "guide_messages", ["user_id", "trip_id", "created_at"], unique=False)
    op.create_index(op.f("ix_guide_messages_trip_id"), "guide_messages", ["trip_id"], unique=False)
    op.create_index(op.f("ix_guide_messages_user_id"), "guide_messages", ["user_id"], unique=False)
    op.create_table(
        "guide_conversation_summaries",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("trip_id", sa.String(length=36), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("message_count", sa.Integer(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["trip_id"], ["trips.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("trip_id"),
    )
    op.create_index(op.f("ix_guide_conversation_summaries_trip_id"), "guide_conversation_summaries", ["trip_id"], unique=False)
    op.create_index(op.f("ix_guide_conversation_summaries_user_id"), "guide_conversation_summaries", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_guide_conversation_summaries_user_id"), table_name="guide_conversation_summaries")
    op.drop_index(op.f("ix_guide_conversation_summaries_trip_id"), table_name="guide_conversation_summaries")
    op.drop_table("guide_conversation_summaries")
    op.drop_index(op.f("ix_guide_messages_user_id"), table_name="guide_messages")
    op.drop_index(op.f("ix_guide_messages_trip_id"), table_name="guide_messages")
    op.drop_index("ix_guide_messages_user_trip_created", table_name="guide_messages")
    op.drop_table("guide_messages")
