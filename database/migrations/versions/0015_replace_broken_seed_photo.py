"""replace broken Udaipur seed photo with verified provider photos

Revision ID: 0015_replace_broken_seed_photo
Revises: 0014_transport_details
Create Date: 2026-09-25 00:00:00.000000

The Udaipur seed catalog (and the destination hero) used the Unsplash photo
``photo-1568495286058-9c3e0b8b0e0e``, which returns HTTP 404 — every hotel,
activity, hero, and already-generated itinerary item referencing it rendered
as a blank photo card. The seed source now carries verified provider photos
(City Palace museum, Incredible India CDN, TripAdvisor CDN; each GET-verified
200 image/jpeg); this migration repairs rows in existing databases:

- the three catalog rows + destination hero are pointed at the same
  verified, place-relevant photos as the fixed seed (by stable PK);
- any itinerary item whose stored ``meta_data.ui.image_url`` still holds the
  dead URL has that key removed, so reads fall back to the fixed linked
  catalog row or the destination-level photo instead of a 404.

All replacements are real, place-relevant provider photos — nothing is
invented. Downgrade is a no-op (restoring a 404 is never desired).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0015_replace_broken_seed_photo"
down_revision: Union[str, None] = "0014_transport_details"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

BROKEN_PHOTO_ID = "photo-1568495286058-9c3e0b8b0e0e"

LAKE_PICHOLA_PHOTO = (
    "https://dynamic-media-cdn.tripadvisor.com/media/photo-o/0c/77/15/9a/photo0jpg.jpg"
    "?w=900&h=500&s=1"
)
CITY_PALACE_PHOTO = "https://citypalacemuseum.org/img/about.jpg"
BAGORE_HAVELI_PHOTO = (
    "https://s7ap1.scene7.com/is/image/incredibleindia/"
    "bagore-ki-haveli-udaipur-rajasthan-1-new-attr-hero?qlt=82&ts=1742173637457"
)
UDAIPUR_HERO_PHOTO = (
    "https://s7ap1.scene7.com/is/image/incredibleindia/"
    "city-palace-udaipur-rajasthan-1-musthead-hero?qlt=82&ts=1742174471611"
)


def _images_replace(images, new_url):
    """Swap dead entries for the verified photo; keep any healthy ones."""
    if not isinstance(images, list) or not images:
        return [new_url]
    fixed = [u for u in images
             if not (isinstance(u, str) and BROKEN_PHOTO_ID in u)]
    if new_url not in fixed:
        fixed.append(new_url)
    return fixed


def upgrade() -> None:
    bind = op.get_bind()
    meta = sa.MetaData()
    hotel = sa.Table("hotels", meta, autoload_with=bind)
    activity = sa.Table("activities", meta, autoload_with=bind)
    destination = sa.Table("destinations", meta, autoload_with=bind)
    item = sa.Table("itinerary_items", meta, autoload_with=bind)

    # Fixed catalog rows (stable seed PKs; same photos as the fixed seed).
    row = bind.execute(
        sa.select(hotel.c.images).where(hotel.c.id == "htl-uda-001")).first()
    if row is not None:
        bind.execute(hotel.update().where(hotel.c.id == "htl-uda-001").values(
            images=_images_replace(row[0], LAKE_PICHOLA_PHOTO)))
    for activity_id, photo in (("act-uda-001", CITY_PALACE_PHOTO),
                               ("act-uda-006", BAGORE_HAVELI_PHOTO)):
        row = bind.execute(
            sa.select(activity.c.images).where(activity.c.id == activity_id)).first()
        if row is not None:
            bind.execute(activity.update().where(activity.c.id == activity_id).values(
                images=_images_replace(row[0], photo)))
    row = bind.execute(sa.select(destination.c.hero_image_url).where(
        destination.c.id == "dest-udaipur-006")).first()
    if row is not None and isinstance(row[0], str) and BROKEN_PHOTO_ID in row[0]:
        bind.execute(destination.update().where(
            destination.c.id == "dest-udaipur-006").values(
                hero_image_url=UDAIPUR_HERO_PHOTO))

    # Already-generated itinerary items: drop the dead stored URL so reads
    # resolve the fixed catalog row (or the destination photo) instead.
    # CAST(... AS TEXT) LIKE keeps this portable across SQLite/Postgres.
    candidates = bind.execute(sa.select(item.c.id, item.c.meta_data).where(
        sa.cast(item.c.meta_data, sa.Text).like(f"%{BROKEN_PHOTO_ID}%"))).all()
    for item_id, meta_data in candidates:
        if not isinstance(meta_data, dict):
            continue
        ui = meta_data.get("ui")
        if not isinstance(ui, dict):
            continue
        url = ui.get("image_url")
        if not (isinstance(url, str) and BROKEN_PHOTO_ID in url):
            continue
        ui = {k: v for k, v in ui.items() if k != "image_url"}
        meta_data = {**meta_data, "ui": ui}
        bind.execute(item.update().where(item.c.id == item_id).values(
            meta_data=meta_data))


def downgrade() -> None:
    # No-op by design: restoring a 404 photo URL helps nobody.
    pass
