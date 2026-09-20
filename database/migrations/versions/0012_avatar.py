"""traveler profile photo (DB-backed avatar)

Revision ID: 0012_avatar
Revises: 0011_guide_memory
Create Date: 2026-09-20 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0012_avatar"
down_revision: Union[str, None] = "0011_guide_memory"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("avatar_image", sa.LargeBinary(), nullable=True))
    op.add_column("users", sa.Column("avatar_mime", sa.String(length=50), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "avatar_mime")
    op.drop_column("users", "avatar_image")
