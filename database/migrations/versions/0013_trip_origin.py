"""trip origin (starting city)

Revision ID: 0013_trip_origin
Revises: 0012_avatar
Create Date: 2026-09-22 00:00:00.000000

Stores the traveler's starting city on the trip row so requirements,
transport matching, and restores all read the same persisted value.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0013_trip_origin"
down_revision: Union[str, None] = "0012_avatar"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("trips", sa.Column("origin", sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column("trips", "origin")
