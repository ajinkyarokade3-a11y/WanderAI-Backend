"""Live provider inventory fill (SerpApi hotels, OSM/Wikimedia places)."""
from backend.live_fill.service import ensure_live_destination, fill_destination_inventory

__all__ = ["ensure_live_destination", "fill_destination_inventory"]
