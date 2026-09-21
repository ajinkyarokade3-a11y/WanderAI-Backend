"""tourflow ai guide conversation memory

Revision ID: 0011_guide_memory
Revises: 0010_notification_indexes
Create Date: 2026-09-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0011_guide_memory"
down_revision: Union[str, None] = "0010_notification_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(bind, name: str) -> bool:
    return sa.inspect(bind).has_table(name)


def _index_exists(bind, table: str, name: str) -> bool:
    insp = sa.inspect(bind)
    return any(idx["name"] == name for idx in insp.get_indexes(table))


def _unique_exists(bind, table: str, column: str) -> bool:
    insp = sa.inspect(bind)
    return any(
        uq.get("column_names") == [column]
        for uq in insp.get_unique_constraints(table)
    )


def upgrade() -> None:
    bind = op.get_bind()
    # The application startup (Base.metadata.create_all) may already have
    # created these tables without recording the migration. Never drop or
    # recreate them; only add what is actually missing.
    if not _table_exists(bind, "guide_messages"):
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
    if not _table_exists(bind, "guide_conversation_summaries"):
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
    elif not _unique_exists(bind, "guide_conversation_summaries", "trip_id"):
        # Model declares trip_id unique; converge a pre-existing table
        # without touching its data (fails loudly instead if dup rows exist).
        op.create_unique_constraint(
            "uq_guide_conversation_summaries_trip_id",
            "guide_conversation_summaries",
            ["trip_id"],
        )
    for index, table, columns in (
        ("ix_guide_messages_user_trip_created", "guide_messages", ["user_id", "trip_id", "created_at"]),
        ("ix_guide_messages_trip_id", "guide_messages", ["trip_id"]),
        ("ix_guide_messages_user_id", "guide_messages", ["user_id"]),
        ("ix_guide_conversation_summaries_trip_id", "guide_conversation_summaries", ["trip_id"]),
        ("ix_guide_conversation_summaries_user_id", "guide_conversation_summaries", ["user_id"]),
    ):
        if not _index_exists(bind, table, index):
            op.create_index(index, table, columns, unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    for index, table in (
        ("ix_guide_conversation_summaries_user_id", "guide_conversation_summaries"),
        ("ix_guide_conversation_summaries_trip_id", "guide_conversation_summaries"),
        ("ix_guide_messages_user_id", "guide_messages"),
        ("ix_guide_messages_trip_id", "guide_messages"),
        ("ix_guide_messages_user_trip_created", "guide_messages"),
    ):
        if _table_exists(bind, table) and _index_exists(bind, table, index):
            op.drop_index(index, table_name=table)
    if _table_exists(bind, "guide_conversation_summaries"):
        op.drop_table("guide_conversation_summaries")
    if _table_exists(bind, "guide_messages"):
        op.drop_table("guide_messages")
