"""Pydantic schemas for the Digital Twin trip simulation."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    """Request body for POST /api/trips/{trip_id}/simulate.

    An optional scenario override allows the client to simulate hypothetical
    weather conditions (e.g., 'severe', 'heavy_rain') without changing the
    real weather context.
    """

    scenario: Optional[str] = Field(
        default=None,
        description="Optional weather scenario override (e.g., 'severe', 'heavy_rain', 'high_wind', 'unavailable').",
    )


class SimulatedAffectedItem(BaseModel):
    """One itinerary item identified as affected by weather."""

    item_id: str
    day_number: int
    order_index: int
    item_type: str
    title: str
    reason: str
    severity: str
    impact_type: str = "none"
    recommended_action: str = "no_change"
    estimated_delay_minutes: Optional[int] = None


class SimulatedDependency(BaseModel):
    """A dependency chain between itinerary items."""

    source_item_id: str
    source_title: str
    target_item_id: str
    target_title: str
    dependency_type: str
    description: str


class SimulatedConflict(BaseModel):
    """A schedule conflict detected during simulation."""

    item_id: str
    item_title: str
    conflict_type: str
    description: str
    severity: str


class SimulationResponse(BaseModel):
    """Structured simulation result returned by the endpoint."""

    trip_id: str
    weather_context: Dict[str, Any]
    scenario: Optional[str] = None
    affected_items: List[SimulatedAffectedItem] = Field(default_factory=list)
    all_items: List[SimulatedAffectedItem] = Field(default_factory=list)
    dependencies: List[SimulatedDependency] = Field(default_factory=list)
    conflicts: List[SimulatedConflict] = Field(default_factory=list)
    replanning_required: bool = False
    simulation_timestamp: str
