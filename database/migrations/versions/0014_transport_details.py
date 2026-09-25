"""transport operator/schedule details

Revision ID: 0014_transport_details
Revises: 0013_trip_origin
Create Date: 2026-09-25 00:00:00.000000

Adds nullable operator/schedule columns to transport_options so the
itinerary and the transport switcher can show real details (service
number, operator, departure/arrival, stops, class, availability, booking
URL) where providers supply them. All nullable: unknown stays NULL and
the UI hides it instead of inventing data.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0014_transport_details"
down_revision: Union[str, None] = "0013_trip_origin"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("transport_options", sa.Column("service_number", sa.String(length=100), nullable=True))
    op.add_column("transport_options", sa.Column("operator_name", sa.String(length=255), nullable=True))
    op.add_column("transport_options", sa.Column("departure_time", sa.String(length=50), nullable=True))
    op.add_column("transport_options", sa.Column("arrival_time", sa.String(length=50), nullable=True))
    op.add_column("transport_options", sa.Column("stops", sa.JSON(), nullable=True))
    op.add_column("transport_options", sa.Column("travel_class", sa.String(length=100), nullable=True))
    op.add_column("transport_options", sa.Column("availability_status", sa.String(length=100), nullable=True))
    op.add_column("transport_options", sa.Column("booking_url", sa.String(length=1024), nullable=True))


def downgrade() -> None:
    op.drop_column("transport_options", "booking_url")
    op.drop_column("transport_options", "availability_status")
    op.drop_column("transport_options", "travel_class")
    op.drop_column("transport_options", "stops")
    op.drop_column("transport_options", "arrival_time")
    op.drop_column("transport_options", "departure_time")
    op.drop_column("transport_options", "operator_name")
    op.drop_column("transport_options", "service_number")
