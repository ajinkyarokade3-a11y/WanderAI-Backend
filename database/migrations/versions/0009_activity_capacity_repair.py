"""repair missing activities.capacity column

Revision ID: 0009_activity_capacity_repair
Revises: 0008_traveler_auth_ownership
Create Date: 2026-09-16 00:00:00.000000

0004_activity_assignments was supposed to add activities.capacity, but
production databases stamped at 0008 still lack the column (the
create_table half of 0004 applied while the add_column did not persist).
Add it back idempotently so both existing and fresh databases converge
with backend.models.models.Activity.capacity (Integer, nullable).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0009_activity_capacity_repair"
down_revision: Union[str, None] = "0008_traveler_auth_ownership"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_capacity(bind) -> bool:
    insp = sa.inspect(bind)
    return any(col["name"] == "capacity" for col in insp.get_columns("activities"))


def upgrade() -> None:
    bind = op.get_bind()
    if not _has_capacity(bind):
        op.add_column("activities", sa.Column("capacity", sa.Integer(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    if _has_capacity(bind):
        op.drop_column("activities", "capacity")
