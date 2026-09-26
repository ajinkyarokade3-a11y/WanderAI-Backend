"""traveler-operator bidirectional chat (trip-scoped, approval-gated)

Revision ID: 0016_traveler_operator_chat
Revises: 0015_replace_broken_seed_photo
Create Date: 2026-09-26 00:00:00.000000

New isolated table ``traveler_operator_chat_messages``: the single shared
traveler<->operator conversation per trip. Deliberately separate from
``trip_messages`` (internal operator notes, traveler-invisible) — no
existing rows are touched.

Eligibility is enforced in service code, not the schema: traveler-confirmed
(Trip.status confirmed|ongoing + confirmed_at) AND operator-accepted
(TripApproval.approved + assignment_started). Downgrade drops the table.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0016_traveler_operator_chat"
down_revision: Union[str, None] = "0015_replace_broken_seed_photo"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "traveler_operator_chat_messages",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("trip_id", sa.String(length=36), nullable=False),
        sa.Column("sender_type", sa.String(length=20), nullable=False),
        sa.Column("sender_id", sa.String(length=36), nullable=True),
        sa.Column("sender_name", sa.String(length=255), nullable=True),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("read_by", sa.String(length=20), nullable=False,
                  server_default="none"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["trip_id"], ["trips.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["sender_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_traveler_operator_chat_trip_id",
                    "traveler_operator_chat_messages", ["trip_id"], unique=False)
    op.create_index("ix_traveler_operator_chat_created_at",
                    "traveler_operator_chat_messages", ["created_at"], unique=False)
    op.create_index("ix_traveler_operator_chat_sender",
                    "traveler_operator_chat_messages",
                    ["trip_id", "sender_type"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_traveler_operator_chat_sender",
                  table_name="traveler_operator_chat_messages")
    op.drop_index("ix_traveler_operator_chat_created_at",
                  table_name="traveler_operator_chat_messages")
    op.drop_index("ix_traveler_operator_chat_trip_id",
                  table_name="traveler_operator_chat_messages")
    op.drop_table("traveler_operator_chat_messages")
