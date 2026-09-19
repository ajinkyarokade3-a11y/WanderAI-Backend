"""notification indexes for traveler API

Revision ID: 0010_notification_indexes
Revises: 0009_activity_capacity_repair
Create Date: 2026-09-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0010_notification_indexes"
down_revision: Union[str, None] = "0009_activity_capacity_repair"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # is_read and created_at indexes were added to model; trip_id and user_id already indexed
    # Use batch mode / check existing for portability
    try:
        op.create_index("ix_notifications_is_read", "notifications", ["is_read"])
    except Exception:
        pass
    try:
        op.create_index("ix_notifications_created_at", "notifications", ["created_at"])
    except Exception:
        pass


def downgrade() -> None:
    try:
        op.drop_index("ix_notifications_is_read", table_name="notifications")
    except Exception:
        pass
    try:
        op.drop_index("ix_notifications_created_at", table_name="notifications")
    except Exception:
        pass
