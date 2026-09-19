from typing import List, Optional, Any, Dict
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Request
from sqlalchemy.orm import Session
from backend.database.connection import get_db
from backend.models.models import (
    Destination, Hotel, Activity, TransportOption, Trip, TripPreference,
    User, ItineraryItem, Booking, Alert, Notification, ChangeHistory, Review
)
from backend.schemas.schemas import (
    DestinationRead, HotelRead, ActivityRead, TransportRead,
    TripCreate, TripRead, TripUpdate, TripPreferenceRead, TripPreferenceUpdate,
    AIChatRequest, AIChatResponse, AIExtractPreferencesRequest, AIRecommendRequest,
    AIGenerateItineraryRequest, AIReplanRequest, ResearchContext, ResearchResult,
    AccommodationContext, AccommodationResult, TransportationContext, TransportationResult,
    ExperienceContext, ExperienceResult, ItineraryContext, ItineraryResult,
    TripManagementContext, TripManagementResult, BookingRecommendationContext,
    BookingRecommendationResult, AssistantChatContext, AssistantChatResult,
    SerpApiHotelResult, HotelSearchResponse, SelectHotelRequest,
    LivePlace, PlacesLiveResponse, PlaceImageResponse, SerpApiRestaurantResult, RestaurantSearchResponse,
    VehicleCreate, VehicleRead, DriverCreate, DriverRead,
    AccommodationAssignRequest, AccommodationReplaceRequest, RoomAllocationRequest,
    AccommodationIssueRequest, AccommodationAssignmentRead, PropertyRead, PropertyTripsResponse,
    TransportAssignRequest, TransportReplaceRequest, JourneyTimingRequest, TransportStatusRequest,
    TransportAssignmentRead, NotifyTravelerRequest, NotifyTravelerResponse,
    ActivityAssignRequest, ActivityReplaceRequest, ActivityAllocationRequest,
    ActivityIssueRequest, ActivityAssignmentRead, ActivityVendorsResponse,
    OpsVendorCreate, VendorAssignmentsResponse, OpsVendorRead, VendorVerifyRequest,
    ActivityInventoryRead, TripConfirmRequest, TripConfirmResponse,
    TripApprovalRequest, TripApprovalRead, TripPipelineResponse,
    TripFinalizeRequest, TripFinalizeResponse,
    TripMessageCreate, TripMessageRead, TripMessageOverviewEntry,
    TravelerSignupRequest, TravelerLoginRequest, TravelerRead, TravelerAuthResponse,
    TravelerTripSaveRequest, TravelerTripSaveResponse, TravelerTripSummary,
    TravelerProfileResponse, TravelerProfileUpdate, TravelerPreferencesRead, TravelerPreferencesUpdate,
    TravelerNotificationRead, NotificationListResponse, UnreadCountResponse, MarkAllReadResponse,
    TravelerBookingRead, TravelerBookingListResponse, BookingCancelRequest, BookingStatusResponse,
    AddRestaurantRequest, RestaurantItemRead,
)
from backend.ai.gemini_service import gemini_service
from backend.research.service import DestinationResearchService, ResearchExecutionError
from backend.accommodation.service import AccommodationRecommendationService, AccommodationExecutionError
from backend.transportation.service import TransportationRecommendationService, TransportationExecutionError
from backend.experience.service import ExperienceRecommendationService, ExperienceExecutionError
from backend.itinerary.service import ItineraryExecutionError, ItineraryRecommendationService, ItineraryValidationError
from backend.trip.service import TripManagementExecutionError, TripManagementService, TripManagementValidationError
from backend.booking.service import (
    BookingRecommendationExecutionError, BookingRecommendationService,
    BookingRecommendationValidationError,
)
from backend.assistant.service import AssistantExecutionError, AssistantService, AssistantValidationError
from backend.database.config import settings
from backend.hotels.service import SerpApiError, search_serpapi_hotels
from backend.recommendation.engine import RecommendationEngine
from backend.dynamic_destination.service import DynamicDestinationDiscoveryError, DynamicDestinationDiscoveryService
from backend.itinerary.generator import ItineraryGenerationError, ItineraryGenerator
from backend.replanning.engine import ReplanningEngine
import os
import secrets

router = APIRouter()


def _trip_or_404(db: Session, trip_id: str) -> Trip:
    trip = db.query(Trip).filter(Trip.id == trip_id).first()
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    return trip


def _resolve_catalog_destination(db: Session, trip_in: TripCreate) -> Destination:
    destination = _find_catalog_destination(db, trip_in)
    if not destination:
        raise HTTPException(status_code=422, detail="A valid catalog destination is required")
    return destination


def _find_catalog_destination(db: Session, trip_in: TripCreate) -> Optional[Destination]:
    destination_id = trip_in.destination_id
    destination_name = trip_in.destination_name
    if isinstance(trip_in.destination, str):
        destination_name = destination_name or trip_in.destination
    elif isinstance(trip_in.destination, dict):
        destination_id = destination_id or trip_in.destination.get("id")
        destination_name = destination_name or trip_in.destination.get("name")

    destination = None
    if destination_id:
        destination = db.query(Destination).filter(
            Destination.id == destination_id,
            Destination.inventory_source == "catalog",
        ).first()
    if not destination and destination_name:
        clean_name = destination_name.strip()
        destination = db.query(Destination).filter(
            ((Destination.name.ilike(clean_name)) | (Destination.slug.ilike(clean_name))),
            Destination.inventory_source == "catalog",
        ).first()
    return destination


def _requested_destination_name(trip_in: TripCreate) -> Optional[str]:
    if trip_in.destination_name:
        return trip_in.destination_name.strip()
    if isinstance(trip_in.destination, str):
        return trip_in.destination.strip()
    if isinstance(trip_in.destination, dict) and trip_in.destination.get("name"):
        return str(trip_in.destination["name"]).strip()
    return None


def _resolve_or_discover_destination(db: Session, trip_in: TripCreate) -> Destination:
    destination = _find_catalog_destination(db, trip_in)
    if destination:
        return destination
    if trip_in.destination_id and not _requested_destination_name(trip_in):
        raise HTTPException(status_code=422, detail="A valid catalog destination is required")
    destination_name = _requested_destination_name(trip_in)
    if not destination_name:
        raise HTTPException(status_code=422, detail="A valid destination name is required")
    try:
        return DynamicDestinationDiscoveryService(db, gemini_service).discover_and_persist(
            trip_in, destination_name, secrets.token_hex(16)
        )
    except DynamicDestinationDiscoveryError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=f"Destination research could not be verified: {exc}") from exc
    except RuntimeError as exc:
        # Gemini research failed (dead key, quota, outage): fall back to
        # live providers (Nominatim geocode + SerpApi hotels + OSM places)
        # so any real place still builds a trip. Only when that fails too
        # is the original unavailability reported.
        db.rollback()
        from backend.live_fill.service import ensure_live_destination
        live = ensure_live_destination(db, destination_name)
        if live is None:
            raise HTTPException(status_code=503, detail=f"Destination research is unavailable: {exc}") from exc
        return live


def _validate_generation_inventory(db: Session, destination: Destination, currency: str, traveler_count: int, duration_days: int) -> None:
    currency = (currency or "INR").upper()
    hotel_filters = [
        Hotel.destination_id == destination.id,
        Hotel.currency == currency,
        Hotel.is_active == True,
    ]
    transport_filters = [
        TransportOption.destination_id == destination.id,
        TransportOption.currency == currency,
        TransportOption.capacity >= max(1, traveler_count),
        TransportOption.is_active == True,
    ]
    activity_filters = [
        Activity.destination_id == destination.id,
        Activity.currency == currency,
        Activity.is_active == True,
    ]
    if destination.inventory_source == "discovered":
        scoped_filters = [
            ("discovery_session_id", destination.discovery_session_id),
            ("verification_status", "verified_candidate"),
            ("inventory_source", "discovered"),
        ]
        for field, value in scoped_filters:
            hotel_filters.append(getattr(Hotel, field) == value)
            transport_filters.append(getattr(TransportOption, field) == value)
            activity_filters.append(getattr(Activity, field) == value)
    hotel_count = db.query(Hotel).filter(*hotel_filters).count()
    transport_count = db.query(TransportOption).filter(*transport_filters).count()
    activity_count = db.query(Activity).filter(*activity_filters).count()
    required_activities = max(0, max(1, duration_days) - 1)
    missing = []
    if hotel_count == 0:
        missing.append("hotels")
    # Transport has no live provider and is legitimately traveler-arranged:
    # its absence no longer blocks creation; the itinerary simply carries
    # no pre-booked transfer items.
    if activity_count < required_activities:
        missing.append(f"at least {required_activities} distinct activities")
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Catalog inventory for {destination.name} is incomplete: missing {', '.join(missing)}",
        )


def _record_change(db: Session, trip: Trip, action: str, field: str, value: str, reason: str, by: str = "user") -> None:
    db.add(ChangeHistory(trip_id=trip.id, changed_by=by, action=action,
                         field_changed=field, new_value=value, reason=reason))


def _hotel_option(hotel: Hotel, trip: Trip, badge: str) -> Dict[str, Any]:
    """Build a frontend AccommodationOption-shaped dict from a catalog Hotel row.

    Pricing follows the traveler-facing rules (rooms = ceil(travelers / 2),
    nights = max(1, duration_days - 1)). The catalog stores no review counts,
    so review_count is 0 rather than an invented value.
    """
    travelers = trip.traveler_count or 2
    nights = max(1, (trip.duration_days or 2) - 1)
    rooms = max(1, -(-travelers // 2))
    price_per_night = float(hotel.price_per_night or 0) * rooms
    images = hotel.images or []
    dest_name = trip.destination.name if trip.destination else ""
    return {"id": hotel.id, "name": hotel.name, "rating": hotel.rating, "review_count": 0,
            "category": hotel.category, "location": hotel.address or dest_name,
            "room_type": f"{rooms}x {hotel.category.title()} Room ({travelers} Guests)",
            "price_per_night": price_per_night, "total_price": price_per_night * nights,
            "nights": nights, "amenities": hotel.amenities or [],
            "why_it_matches": f"Verified catalog {hotel.category} stay in {dest_name} rated {hotel.rating}.",
            "hero_image": images[0] if images else None, "images": images, "badge": badge}


def _trip_dict(trip: Trip, db: Session) -> Dict[str, Any]:
    """Serialize the persisted trip plus UI-derived selection fields.

    Hotel selection fields are derived from canonical data only: the trip's
    hotel itinerary items joined to active catalog Hotel rows. No selection
    is fabricated when the trip has no usable hotel item (selected is None).
    """
    itinerary = []
    for item in trip.itinerary:
        row = {"id": item.id, "trip_id": item.trip_id, "day_number": item.day_number,
               "order_index": item.order_index, "item_type": item.item_type, "title": item.title,
               "description": item.description, "start_time": item.start_time, "end_time": item.end_time,
               "cost": item.cost, "status": item.status, "hotel_id": item.hotel_id,
               "activity_id": item.activity_id, "transport_id": item.transport_id,
               "location": item.location, "meta_data": item.meta_data or {}}
        row.update((item.meta_data or {}).get("ui", {}))
        # Image fallback: resolve from the linked catalog row so every
        # hotel/activity stop renders a photo even when the stored item
        # predates image metadata (transport has no catalog photos).
        if not row.get("image_url"):
            catalog_images = None
            try:
                if item.hotel is not None:
                    catalog_images = item.hotel.images
                elif item.activity is not None:
                    catalog_images = item.activity.images
            except Exception:
                catalog_images = None
            if catalog_images:
                row["image_url"] = catalog_images[0]
        itinerary.append(row)
    bookings = [{"id": b.id, "trip_id": b.trip_id, "vendor_id": b.vendor_id,
                 "booking_reference": b.booking_reference, "item_type": b.item_type, "item_id": b.item_id,
                 "amount": b.amount, "currency": b.currency, "status": b.status,
                 "payment_status": b.payment_status, "booking_date": b.booking_date.isoformat()} for b in trip.bookings]
    hotel_items = sorted(
        [i for i in trip.itinerary if i.item_type == "hotel" and i.hotel is not None and i.hotel.is_active],
        key=lambda i: (i.day_number, i.order_index),
    )
    selected_accommodation = _hotel_option(hotel_items[0].hotel, trip, "best_match") if hotel_items else None
    selected_id = hotel_items[0].hotel.id if hotel_items else None
    alternatives: List[Dict[str, Any]] = []
    if trip.destination_id:
        query = db.query(Hotel).filter(Hotel.destination_id == trip.destination_id, Hotel.is_active == True)
        if selected_id:
            query = query.filter(Hotel.id != selected_id)
        candidates = query.order_by(Hotel.rating.desc(), Hotel.price_per_night.asc()).limit(4).all()
        cheapest_id = min(candidates, key=lambda h: h.price_per_night).id if candidates else None
        top_rated_id = candidates[0].id if candidates else None
        for hotel in candidates:
            if hotel.id == cheapest_id:
                badge = "cheapest"
            elif hotel.id == top_rated_id:
                badge = "best_rated"
            elif hotel.category == "luxury":
                badge = "luxury"
            else:
                badge = "best_match"
            alternatives.append(_hotel_option(hotel, trip, badge))
    daily_accommodations = [{"day_number": item.day_number, "hotel": _hotel_option(item.hotel, trip, "best_match")}
                            for item in hotel_items]
    transport_cost = sum(float(i.cost or 0) for i in trip.itinerary if i.item_type == "transport")
    accommodation_cost = sum(float(i.cost or 0) for i in trip.itinerary if i.item_type == "hotel")
    activities_cost = sum(float(i.cost or 0) for i in trip.itinerary if i.item_type == "activity")
    total_cost = transport_cost + accommodation_cost + activities_cost
    target_budget = float(trip.total_budget or 0)
    cost_breakdown = {"transport": transport_cost, "accommodation": accommodation_cost,
                      "activities": activities_cost, "food_and_other": 0.0, "total": total_cost,
                      "target_budget": target_budget, "remaining_budget": target_budget - total_cost,
                      "is_under_budget": total_cost <= target_budget}
    return {"id": trip.id, "user_id": trip.user_id, "destination_id": trip.destination_id,
            "title": trip.title, "status": trip.status, "start_date": trip.start_date.isoformat() if trip.start_date else None,
            "end_date": trip.end_date.isoformat() if trip.end_date else None, "duration_days": trip.duration_days,
            "total_budget": trip.total_budget, "currency": trip.currency, "traveler_count": trip.traveler_count,
            "pace": trip.pace, "discovery_session_id": trip.discovery_session_id,
            "confirmed_at": trip.confirmed_at.isoformat() if trip.confirmed_at else None,
            "confirmed_by": trip.confirmed_by,
            "created_at": trip.created_at.isoformat(), "updated_at": trip.updated_at.isoformat(),
            "destination": {"id": trip.destination.id, "name": trip.destination.name, "slug": trip.destination.slug,
                            "country": trip.destination.country, "state_region": trip.destination.state_region,
                            "description": trip.destination.description, "hero_image_url": trip.destination.hero_image_url,
                            "best_time_to_visit": trip.destination.best_time_to_visit, "tags": trip.destination.tags or [],
                            "latitude": trip.destination.latitude, "longitude": trip.destination.longitude,
                            "source_url": trip.destination.source_url, "evidence": trip.destination.evidence or [],
                            "inventory_source": trip.destination.inventory_source,
                            "verification_status": trip.destination.verification_status,
                            "discovery_session_id": trip.destination.discovery_session_id,
                            "is_featured": trip.destination.is_featured, "created_at": trip.destination.created_at.isoformat()} if trip.destination else None,
            "itinerary": itinerary, "bookings": bookings,
            "total_cost": total_cost, "cost_breakdown": cost_breakdown,
            "selected_accommodation": selected_accommodation,
            "accommodation_alternatives": alternatives,
            "daily_accommodations": daily_accommodations,
            "preferences": {"id": trip.preferences.id, "trip_id": trip.preferences.trip_id,
                            "budget_tier": trip.preferences.budget_tier, "interests": trip.preferences.interests or [],
                            "travel_companions": trip.preferences.travel_companions,
                            "accommodation_types": trip.preferences.accommodation_types or [],
                            "transport_preferences": trip.preferences.transport_preferences or [],
                            "dietary_requirements": trip.preferences.dietary_requirements or [],
                            "special_requests": trip.preferences.special_requests,
                            "created_at": trip.preferences.created_at.isoformat(),
                            "updated_at": trip.preferences.updated_at.isoformat()} if trip.preferences else None,
            "alerts": [{"id": a.id, "trip_id": a.trip_id, "alert_type": a.alert_type, "severity": a.severity,
                        "title": a.title, "description": a.description, "is_resolved": a.is_resolved,
                        "created_at": a.created_at.isoformat()} for a in trip.alerts],
            "notifications": [{"id": n.id, "trip_id": n.trip_id, "user_id": n.user_id, "title": n.title,
                               "message": n.message, "type": n.type, "is_read": n.is_read,
                               "created_at": n.created_at.isoformat()} for n in trip.notifications],
            "change_history": [{"id": h.id, "trip_id": h.trip_id, "changed_by": h.changed_by, "action": h.action,
                                "field_changed": h.field_changed, "old_value": h.old_value, "new_value": h.new_value,
                                "reason": h.reason, "timestamp": h.timestamp.isoformat()} for h in trip.change_history]}

# ----------------------------------------------------
# Health Check API
# ----------------------------------------------------
@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Health check verifying API, DB connectivity, and Gemini AI status."""
    try:
        dest_count = db.query(Destination).count()
        trip_count = db.query(Trip).count()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
        dest_count = 0
        trip_count = 0

    return {
        "status": "healthy",
        "service": "TourFlow AI API",
        "version": "1.0.0",
        "database": db_status,
        "counts": {
            "destinations": dest_count,
            "trips": trip_count
        },
        "ai_engine": {
            "gemini_available": gemini_service.is_available(),
            "model": "gemini-3.6-flash"
        }
    }

# ----------------------------------------------------
# Frontend sync metadata
# ----------------------------------------------------
@router.get("/sync/version")
def sync_version(db: Session = Depends(get_db)):
    """Return a lightweight version marker used by the operator frontend."""
    trips = db.query(Trip).all()
    latest = max((t.updated_at or t.created_at for t in trips), default=None)
    version = int(latest.timestamp()) if latest else 0
    return {
        "version": version,
        "timestamp": latest.isoformat() if latest else datetime.utcnow().isoformat() + "Z",
        "trips_count": len(trips),
    }

# ----------------------------------------------------
# Destinations API
# ----------------------------------------------------
@router.get("/destinations/search")
def search_destinations(
    q: Optional[str] = Query(default=None, max_length=255),
    featured_only: bool = Query(default=False),
    category: Optional[str] = Query(default=None, max_length=100),
    tag: Optional[str] = Query(default=None, max_length=100),
    country: Optional[str] = Query(default=None, max_length=100),
    state_region: Optional[str] = Query(default=None, max_length=100),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Search catalog destinations (never fabricated). Filters: text query, featured, category/tag, country, state_region. Pagination limit/offset."""
    query = db.query(Destination)
    # featured filter
    if featured_only:
        query = query.filter(Destination.is_featured == True)
    if country:
        query = query.filter(Destination.country.ilike(country.strip()))
    if state_region:
        query = query.filter(Destination.state_region.ilike(state_region.strip()))
    # text search across name, state_region, country, description, tags
    if q and q.strip():
        term = f"%{q.strip()}%"
        # For tags JSON, ilike on casted text works for sqlite/postgres as fallback; also handle python filtering later but include SQL quick filter
        query = query.filter(
            (Destination.name.ilike(term)) |
            (Destination.state_region.ilike(term)) |
            (Destination.country.ilike(term)) |
            (Destination.description.ilike(term))
        )
        # also consider tags: fetch separately if needed; we will post-filter to include tags matches that were excluded by SQL
        # To avoid missing tags-only matches, union with tags-matched ids via python scan
        tag_filter = q.strip().lower()
        all_dests = db.query(Destination).all()
        tag_matched_ids = {d.id for d in all_dests if any(tag_filter in (t or "").lower() for t in (d.tags or []))}
        if tag_matched_ids:
            # Include tag-matched that may have been excluded
            base_ids = {d.id for d in query.all()}
            combined_ids = base_ids | tag_matched_ids
            # Reapply other filters to tag-matched set
            query = db.query(Destination).filter(Destination.id.in_(list(combined_ids)))
            if featured_only:
                query = query.filter(Destination.is_featured == True)
            if country:
                query = query.filter(Destination.country.ilike(country.strip()))
            if state_region:
                query = query.filter(Destination.state_region.ilike(state_region.strip()))
    # category/tag filter (both aliases)
    cat = (category or tag)
    if cat and cat.strip():
        cat_low = cat.strip().lower()
        # DB-level filter not reliable for JSON, so filter in python after total count? Instead load and filter
        all_filtered = query.all()
        matched = [d for d in all_filtered if any(cat_low == (t or "").lower() or cat_low in (t or "").lower() for t in (d.tags or []))]
        total = len(matched)
        matched_sorted = sorted(matched, key=lambda d: d.name.lower())
        results = matched_sorted[offset: offset+limit]
        return {"results": results, "total": total, "limit": limit, "offset": offset}
    total = query.count()
    results = query.order_by(Destination.name.asc()).offset(offset).limit(limit).all()
    return {"results": results, "total": total, "limit": limit, "offset": offset}


@router.get("/destinations/featured", response_model=List[DestinationRead])
def get_featured_destinations(db: Session = Depends(get_db)):
    """Return curated featured destinations."""
    return db.query(Destination).filter(Destination.is_featured == True).order_by(Destination.name.asc()).all()


@router.get("/destinations/categories")
def get_destination_categories(db: Session = Depends(get_db)):
    """Return available tags/categories from actual DB data."""
    rows = db.query(Destination).all()
    tags_set = set()
    for r in rows:
        for t in (r.tags or []):
            if isinstance(t, str) and t.strip():
                tags_set.add(t.strip())
    return {"categories": sorted(tags_set, key=lambda s: s.lower()), "total": len(tags_set)}


@router.get("/destinations/nearby")
def get_nearby_destinations(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(default=500, ge=1, le=20000),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Return destinations within radius_km of given coordinates, sorted by distance."""
    import math
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2-lat1)
        dlon = math.radians(lon2-lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
        return 2*R*math.asin(math.sqrt(a))
    cands = db.query(Destination).filter(Destination.latitude.isnot(None), Destination.longitude.isnot(None)).all()
    scored = []
    for d in cands:
        try:
            dist = haversine(latitude, longitude, float(d.latitude), float(d.longitude))
        except:
            continue
        if dist <= radius_km:
            scored.append((dist, d))
    scored.sort(key=lambda x: x[0])
    limited = scored[:limit]
    return {
        "latitude": latitude,
        "longitude": longitude,
        "radius_km": radius_km,
        "results": [
            {**DestinationRead.model_validate(doc).model_dump(), "distance_km": round(dist, 2)}
            for dist, doc in limited
        ],
        "total": len(scored),
        "limit": limit,
    }


@router.get("/destinations", response_model=List[DestinationRead])
def get_destinations(
    featured_only: bool = False,
    db: Session = Depends(get_db)
):
    """Retrieve list of all destinations or filtered by featured status."""
    query = db.query(Destination)
    if featured_only:
        query = query.filter(Destination.is_featured == True)
    return query.order_by(Destination.name.asc()).all()

@router.get("/destinations/{id}", response_model=DestinationRead)
def get_destination_by_id(id: str, db: Session = Depends(get_db)):
    """Retrieve destination details by ID or Slug."""
    destination = db.query(Destination).filter(
        (Destination.id == id) | (Destination.slug == id)
    ).first()
    if not destination:
        raise HTTPException(status_code=404, detail="Destination not found")
    return destination

# ----------------------------------------------------
# Hotels API
# ----------------------------------------------------
@router.get("/hotels", response_model=List[HotelRead])
def get_hotels(
    destination_id: Optional[str] = None,
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Retrieve hotels with optional destination and category filters."""
    query = db.query(Hotel).filter(Hotel.is_active == True)
    if destination_id:
        query = query.filter(Hotel.destination_id == destination_id)
    if category:
        query = query.filter(Hotel.category == category)
    return query.order_by(Hotel.rating.desc()).all()

@router.get("/hotels/search", response_model=HotelSearchResponse)
def search_hotels_live(
    destination: str = Query(min_length=1, max_length=255),
    check_in_date: str = Query(min_length=8, max_length=10),
    check_out_date: str = Query(min_length=8, max_length=10),
    adults: int = Query(default=2, ge=1, le=16),
    children: int = Query(default=0, ge=0, le=10),
    currency: str = Query(default="INR", min_length=3, max_length=10),
    gl: str = Query(default="in", min_length=2, max_length=5),
    hl: str = Query(default="en", min_length=2, max_length=10),
    min_price: Optional[float] = Query(default=None, ge=0),
    max_price: Optional[float] = Query(default=None, ge=0),
    min_rating: Optional[float] = Query(default=None, ge=0, le=5),
):
    """Live hotel search via SerpApi Google Hotels (backend key, normalized)."""
    from backend.hotels.service import validate_search_dates
    api_key = (settings.SERPAPI_API_KEY or "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="Hotel search provider is not configured")
    try:
        validate_search_dates(check_in_date, check_out_date)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        results = search_serpapi_hotels(
            api_key, settings.SERPAPI_BASE_URL, destination=destination,
            check_in_date=check_in_date.strip(), check_out_date=check_out_date.strip(),
            adults=adults, children=children, currency=currency, gl=gl, hl=hl,
            min_price=min_price, max_price=max_price, min_rating=min_rating,
            timeout_s=settings.SERPAPI_TIMEOUT_S, max_results=settings.SERPAPI_MAX_RESULTS,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SerpApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"destination": destination.strip(), "check_in_date": check_in_date.strip(),
            "check_out_date": check_out_date.strip(), "currency": currency.upper(),
            "results": [SerpApiHotelResult(**item) for item in results], "source": "serpapi"}

@router.get("/places/live", response_model=PlacesLiveResponse)
def places_live(destination: str = Query(min_length=1, max_length=255),
                limit: int = Query(default=12, ge=1, le=30),
                db: Session = Depends(get_db)):
    """Live places/attractions with provider-backed images (keyless providers).

    Catalog coordinates are preferred; unknown destinations are geocoded.
    Always 200 (possibly empty) -- never fabricated.
    """
    from backend.places.service import get_live_places
    dest = db.query(Destination).filter(
        (Destination.name.ilike(destination.strip())) | (Destination.slug.ilike(destination.strip()))
    ).first()
    result = get_live_places(
        destination.strip(),
        dest.latitude if dest else None, dest.longitude if dest else None, limit,
        settings.NOMINATIM_API_URL, settings.OVERPASS_API_URL, settings.COMMONS_API_URL,
        settings.PLACES_TIMEOUT_S, settings.PLACES_RADIUS_M,
    )
    return {"destination": result["destination"], "latitude": result["latitude"],
            "longitude": result["longitude"],
            "places": [LivePlace(**place) for place in result["places"]],
            "source": result["source"]}


@router.get("/places/image", response_model=PlaceImageResponse)
def place_image(
    location: str = Query(min_length=1, max_length=255),
    destination: Optional[str] = Query(default=None, max_length=255),
    count: int = Query(default=1, ge=1, le=6),
):
    """Real photos for one location via SerpApi Google Images (backend key).

    Returns up to ``count`` distinct relevance-ranked photos from a single
    provider call. Always 200 with an empty list when nothing real is found --
    never fabricated, and a provider failure never breaks trip generation.
    503 when the key is unconfigured; 422 on blank location.
    """
    from backend.images.service import get_real_images_for_location
    api_key = (settings.SERPAPI_API_KEY or "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="Image search provider is not configured")
    query = location.strip()
    if not query:
        raise HTTPException(status_code=422, detail="location must not be blank")
    images = get_real_images_for_location(
        query, (destination or "").strip() or None,
        api_key, settings.SERPAPI_BASE_URL, settings.SERPAPI_TIMEOUT_S, count,
    )
    return {"location": query, "image_url": images[0] if images else None,
            "images": images, "source": "serpapi_images" if images else "none"}


@router.get("/restaurants/search", response_model=RestaurantSearchResponse)
def search_restaurants_live(
    destination: str = Query(min_length=1, max_length=255),
    meal_type: Optional[str] = Query(default=None, max_length=20),
    cuisine: Optional[str] = Query(default=None, max_length=100),
    latitude: Optional[float] = Query(default=None, ge=-90, le=90),
    longitude: Optional[float] = Query(default=None, ge=-180, le=180),
    min_rating: Optional[float] = Query(default=None, ge=0, le=5),
    max_results: int = Query(default=8, ge=1, le=20),
):
    """Live restaurant search via SerpApi Google Maps (backend key, normalized).

    Returns real local-business candidates only. Empty list when the provider
    has no results; 503 when the key is unconfigured; 502 on provider failure.
    """
    from backend.restaurants.service import SerpApiRestaurantError, search_serpapi_restaurants
    api_key = (settings.SERPAPI_API_KEY or "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="Restaurant search provider is not configured")
    meal = (meal_type or "").strip().lower() or None
    if meal is not None and meal not in {"breakfast", "brunch", "lunch", "dinner"}:
        raise HTTPException(status_code=422, detail="meal_type must be breakfast, brunch, lunch, or dinner")
    try:
        results = search_serpapi_restaurants(
            api_key, settings.SERPAPI_BASE_URL, destination=destination.strip(),
            meal_type=meal, cuisine=(cuisine or "").strip() or None,
            latitude=latitude, longitude=longitude, min_rating=min_rating,
            timeout_s=settings.SERPAPI_TIMEOUT_S, max_results=max_results,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SerpApiRestaurantError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"destination": destination.strip(), "meal_type": meal, "cuisine": (cuisine or "").strip() or None,
            "results": [SerpApiRestaurantResult(**item) for item in results], "source": "serpapi"}


@router.get("/weather", response_model=Dict[str, Any])
def get_weather(
    destination: str = Query(min_length=1, max_length=255),
    latitude: Optional[float] = Query(default=None, ge=-90, le=90),
    longitude: Optional[float] = Query(default=None, ge=-180, le=180),
    date: Optional[str] = Query(default=None, max_length=10),
    days: int = Query(default=5, ge=1, le=16),
    db: Session = Depends(get_db),
):
    """Live weather/forecast via server-side provider. Never fabricated."""
    from backend.weather.service import fetch_weather, resolve_coordinates, WeatherNotConfigured, WeatherProviderError, validate_date_str
    if not destination.strip():
        raise HTTPException(status_code=422, detail="destination is required")
    try:
        date_str = validate_date_str(date)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if not settings.WEATHER_BASE_URL.strip():
        raise HTTPException(status_code=503, detail="Weather provider is not configured")
    lat, lon, disp = resolve_coordinates(destination.strip(), latitude, longitude, db, settings.NOMINATIM_API_URL, settings.PLACES_TIMEOUT_S)
    if lat is None or lon is None:
        raise HTTPException(status_code=422, detail="Could not resolve destination coordinates")
    try:
        data = fetch_weather(lat, lon, settings.WEATHER_BASE_URL, settings.WEATHER_TIMEOUT_S, days=days, date_str=date_str, api_key=settings.WEATHER_API_KEY)
    except WeatherNotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e))
    except WeatherProviderError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {
        "destination": disp,
        "latitude": lat,
        "longitude": lon,
        "current": data["current"],
        "forecast": data["forecast"],
        "source": data["source"],
        "retrieved_at": data["retrieved_at"],
    }


# ----------------------------------------------------
# Operations consoles (hotels dispatch + transport dispatch)
# ----------------------------------------------------
def _ops_error(exc: Exception) -> HTTPException:
    from backend.ops.service import OpsConflict, OpsForbidden, OpsNotFound, OpsValidation
    if isinstance(exc, OpsNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, OpsForbidden):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, OpsConflict):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, OpsValidation):
        return HTTPException(status_code=422, detail=str(exc))
    raise exc


@router.get("/ops/accommodations", response_model=List[AccommodationAssignmentRead])
def ops_list_accommodations(
    status: Optional[str] = None, db: Session = Depends(get_db)
):
    """List persisted trip accommodation assignments, optionally by status."""
    from backend.ops.service import list_accommodation_assignments
    try:
        return list_accommodation_assignments(db, status)
    except Exception as exc:
        raise _ops_error(exc)


@router.get("/ops/accommodations/{trip_id}", response_model=AccommodationAssignmentRead)
def ops_get_accommodation(trip_id: str, db: Session = Depends(get_db)):
    """Fetch one trip's accommodation assignment."""
    from backend.ops.service import OpsNotFound, get_accommodation_assignment
    row = get_accommodation_assignment(db, trip_id)
    if not row:
        raise HTTPException(status_code=404, detail="Accommodation assignment not found for trip")
    return row


@router.post("/ops/accommodations", response_model=AccommodationAssignmentRead, status_code=201)
def ops_create_accommodation(payload: AccommodationAssignRequest, db: Session = Depends(get_db)):
    """Assign a property (from inventory) to a trip. 409 when one already exists."""
    from backend.ops.service import create_accommodation_assignment
    try:
        return create_accommodation_assignment(
            db, payload.trip_id, payload.hotel_id, payload.rooms, payload.room_type,
            payload.check_in_date, payload.check_out_date, payload.updated_by or "operator",
        )
    except Exception as exc:
        raise _ops_error(exc)


@router.put("/ops/accommodations/{trip_id}", response_model=AccommodationAssignmentRead)
def ops_replace_accommodation(
    trip_id: str, payload: AccommodationReplaceRequest, db: Session = Depends(get_db)
):
    """Change the property/rooms/dates of an existing trip assignment."""
    from backend.ops.service import replace_accommodation_assignment
    try:
        return replace_accommodation_assignment(
            db, trip_id, payload.hotel_id, payload.rooms, payload.room_type,
            payload.check_in_date, payload.check_out_date, payload.updated_by or "operator",
        )
    except Exception as exc:
        raise _ops_error(exc)


@router.put("/ops/accommodations/{trip_id}/rooms", response_model=AccommodationAssignmentRead)
def ops_update_rooms(trip_id: str, payload: RoomAllocationRequest, db: Session = Depends(get_db)):
    """Assign or update room allocation for a trip assignment."""
    from backend.ops.service import update_room_allocation
    try:
        return update_room_allocation(
            db, trip_id, payload.rooms, payload.room_type, payload.updated_by or "operator"
        )
    except Exception as exc:
        raise _ops_error(exc)


@router.post("/ops/accommodations/{trip_id}/flag-issue", response_model=AccommodationAssignmentRead)
def ops_flag_accommodation_issue(
    trip_id: str, payload: AccommodationIssueRequest, db: Session = Depends(get_db)
):
    """Flag an assignment as an operational issue with a reason."""
    from backend.ops.service import flag_accommodation_issue
    try:
        return flag_accommodation_issue(
            db, trip_id, payload.reason, payload.updated_by or "operator"
        )
    except Exception as exc:
        raise _ops_error(exc)


@router.post("/ops/accommodations/{trip_id}/resolve-issue", response_model=AccommodationAssignmentRead)
def ops_resolve_accommodation_issue(trip_id: str, db: Session = Depends(get_db)):
    """Clear a manual issue flag by recomputing backend status rules."""
    from backend.ops.service import resolve_accommodation_issue
    try:
        return resolve_accommodation_issue(db, trip_id)
    except Exception as exc:
        raise _ops_error(exc)


@router.get("/ops/properties", response_model=List[PropertyRead])
def ops_list_properties(
    active_only: bool = True, db: Session = Depends(get_db)
):
    """Property inventory from the database with live trip assignments."""
    from backend.ops.service import list_properties
    return list_properties(db, active_only)


@router.get("/ops/properties/{hotel_id}/trips", response_model=PropertyTripsResponse)
def ops_property_trips(hotel_id: str, db: Session = Depends(get_db)):
    """All trip assignments currently using one property."""
    from backend.ops.service import get_property_trips
    try:
        return get_property_trips(db, hotel_id)
    except Exception as exc:
        raise _ops_error(exc)


@router.get("/ops/vehicles", response_model=List[VehicleRead])
def ops_list_vehicles(active_only: bool = True, db: Session = Depends(get_db)):
    """Vehicle inventory from the database."""
    from backend.ops.service import list_vehicles
    return list_vehicles(db, active_only)


@router.post("/ops/vehicles", response_model=VehicleRead, status_code=201)
def ops_create_vehicle(payload: VehicleCreate, db: Session = Depends(get_db)):
    """Onboard a vehicle into dispatch inventory."""
    from backend.ops.service import create_vehicle
    try:
        return create_vehicle(
            db, payload.name, payload.registration_number,
            payload.vehicle_type, payload.capacity,
        )
    except Exception as exc:
        raise _ops_error(exc)


@router.get("/ops/drivers", response_model=List[DriverRead])
def ops_list_drivers(active_only: bool = True, db: Session = Depends(get_db)):
    """Driver roster from the database."""
    from backend.ops.service import list_drivers
    return list_drivers(db, active_only)


@router.post("/ops/drivers", response_model=DriverRead, status_code=201)
def ops_create_driver(payload: DriverCreate, db: Session = Depends(get_db)):
    """Onboard a driver into dispatch inventory."""
    from backend.ops.service import create_driver
    try:
        return create_driver(db, payload.name, payload.phone, payload.license_number)
    except Exception as exc:
        raise _ops_error(exc)


@router.get("/ops/transport", response_model=List[TransportAssignmentRead])
def ops_list_transport(status: Optional[str] = None, db: Session = Depends(get_db)):
    """List persisted trip transport assignments, optionally by status."""
    from backend.ops.service import list_transport_assignments
    try:
        return list_transport_assignments(db, status)
    except Exception as exc:
        raise _ops_error(exc)


@router.get("/ops/transport/{trip_id}", response_model=TransportAssignmentRead)
def ops_get_transport(trip_id: str, db: Session = Depends(get_db)):
    """Fetch one trip's transport assignment."""
    from backend.ops.service import get_transport_assignment
    row = get_transport_assignment(db, trip_id)
    if not row:
        raise HTTPException(status_code=404, detail="Transport assignment not found for trip")
    return row


@router.post("/ops/transport", response_model=TransportAssignmentRead, status_code=201)
def ops_create_transport(payload: TransportAssignRequest, db: Session = Depends(get_db)):
    """Assign vehicle/driver/route/timing to a trip. 409 on resource conflicts."""
    from backend.ops.service import create_transport_assignment
    try:
        return create_transport_assignment(
            db, payload.trip_id, payload.vehicle_id, payload.driver_id,
            payload.origin, payload.destination, payload.pickup_at, payload.dropoff_at,
            payload.updated_by or "operator",
        )
    except Exception as exc:
        raise _ops_error(exc)


@router.put("/ops/transport/{trip_id}", response_model=TransportAssignmentRead)
def ops_replace_transport(
    trip_id: str, payload: TransportReplaceRequest, db: Session = Depends(get_db)
):
    """Reassign vehicle/driver/route/timing for a trip."""
    from backend.ops.service import replace_transport_assignment
    try:
        return replace_transport_assignment(
            db, trip_id, payload.vehicle_id, payload.driver_id,
            payload.origin, payload.destination, payload.pickup_at, payload.dropoff_at,
            payload.updated_by or "operator",
        )
    except Exception as exc:
        raise _ops_error(exc)


@router.put("/ops/transport/{trip_id}/timing", response_model=TransportAssignmentRead)
def ops_update_timing(trip_id: str, payload: JourneyTimingRequest, db: Session = Depends(get_db)):
    """Update pickup/drop-off times and route for a trip assignment."""
    from backend.ops.service import update_journey_timing
    try:
        return update_journey_timing(
            db, trip_id, payload.pickup_at, payload.dropoff_at,
            payload.origin, payload.destination, payload.updated_by or "operator",
        )
    except Exception as exc:
        raise _ops_error(exc)


@router.post("/ops/transport/{trip_id}/status", response_model=TransportAssignmentRead)
def ops_transport_status(
    trip_id: str, payload: TransportStatusRequest, db: Session = Depends(get_db)
):
    """Advance the journey lifecycle (pending->assigned->en_route->completed, delay handling)."""
    from backend.ops.service import transition_transport_status
    try:
        return transition_transport_status(
            db, trip_id, payload.to_status, payload.delay_reason,
            payload.updated_by or "operator",
        )
    except Exception as exc:
        raise _ops_error(exc)


@router.post("/ops/transport/{trip_id}/notify", response_model=NotifyTravelerResponse, status_code=201)
def ops_notify_traveler(
    trip_id: str, payload: NotifyTravelerRequest, db: Session = Depends(get_db)
):
    """Persist a traveler notification built from real assignment facts."""
    from backend.ops.service import notify_traveler
    try:
        return notify_traveler(db, trip_id, payload.event, payload.note, payload.updated_by or "operator")
    except Exception as exc:
        raise _ops_error(exc)


@router.get("/ops/activities", response_model=List[ActivityAssignmentRead])
def ops_list_activities(
    trip_id: Optional[str] = None,
    status: Optional[str] = None,
    vendor_id: Optional[str] = None,
    scheduled_date: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List persisted trip activity assignments with backend filters."""
    from backend.ops.service import list_activity_assignments
    try:
        return list_activity_assignments(db, trip_id, status, vendor_id, scheduled_date)
    except Exception as exc:
        raise _ops_error(exc)


@router.get("/ops/activities/{assignment_id}", response_model=ActivityAssignmentRead)
def ops_get_activity(assignment_id: str, db: Session = Depends(get_db)):
    """Fetch one activity assignment with its full operational context."""
    from backend.ops.service import get_activity_assignment
    try:
        return get_activity_assignment(db, assignment_id)
    except Exception as exc:
        raise _ops_error(exc)


@router.post("/ops/activities", response_model=ActivityAssignmentRead, status_code=201)
def ops_create_activity(payload: ActivityAssignRequest, db: Session = Depends(get_db)):
    """Assign an activity (from inventory) to a trip. 409 on conflicts."""
    from backend.ops.service import create_activity_assignment
    try:
        return create_activity_assignment(
            db, payload.trip_id, payload.activity_id, payload.vendor_id,
            payload.scheduled_date, payload.start_time, payload.end_time,
            payload.participants, payload.updated_by or "operator",
        )
    except Exception as exc:
        raise _ops_error(exc)


@router.put("/ops/activities/{assignment_id}", response_model=ActivityAssignmentRead)
def ops_replace_activity(
    assignment_id: str, payload: ActivityReplaceRequest, db: Session = Depends(get_db)
):
    """Change activity/vendor/schedule/allocation of an assignment."""
    from backend.ops.service import replace_activity_assignment
    try:
        return replace_activity_assignment(
            db, assignment_id, payload.activity_id, payload.vendor_id,
            bool(payload.vendor_cleared), payload.scheduled_date, payload.start_time,
            payload.end_time, payload.participants, payload.updated_by or "operator",
        )
    except Exception as exc:
        raise _ops_error(exc)


@router.put("/ops/activities/{assignment_id}/allocation", response_model=ActivityAssignmentRead)
def ops_update_allocation(
    assignment_id: str, payload: ActivityAllocationRequest, db: Session = Depends(get_db)
):
    """Adjust participant allocation; capacity conflicts rejected with 409."""
    from backend.ops.service import update_activity_allocation
    try:
        return update_activity_allocation(
            db, assignment_id, payload.participants, payload.updated_by or "operator"
        )
    except Exception as exc:
        raise _ops_error(exc)


@router.post("/ops/activities/{assignment_id}/confirm", response_model=ActivityAssignmentRead)
def ops_confirm_activity(assignment_id: str, db: Session = Depends(get_db)):
    """Confirm a complete, validated assignment. 422 when incomplete."""
    from backend.ops.service import confirm_activity_assignment
    try:
        return confirm_activity_assignment(db, assignment_id)
    except Exception as exc:
        raise _ops_error(exc)


@router.post("/ops/activities/{assignment_id}/flag-issue", response_model=ActivityAssignmentRead)
def ops_flag_activity_issue(
    assignment_id: str, payload: ActivityIssueRequest, db: Session = Depends(get_db)
):
    """Flag an assignment as an operational issue with a reason."""
    from backend.ops.service import flag_activity_issue
    try:
        return flag_activity_issue(db, assignment_id, payload.reason)
    except Exception as exc:
        raise _ops_error(exc)


@router.post("/ops/activities/{assignment_id}/resolve-issue", response_model=ActivityAssignmentRead)
def ops_resolve_activity_issue(assignment_id: str, db: Session = Depends(get_db)):
    """Clear a manual issue flag by re-running backend validation."""
    from backend.ops.service import resolve_activity_issue
    try:
        return resolve_activity_issue(db, assignment_id)
    except Exception as exc:
        raise _ops_error(exc)


@router.get("/ops/activity-inventory/{activity_id}/vendors", response_model=ActivityVendorsResponse)
def ops_eligible_vendors(activity_id: str, db: Session = Depends(get_db)):
    """Verified vendors eligible for one activity."""
    from backend.ops.service import list_eligible_vendors
    try:
        return list_eligible_vendors(db, activity_id)
    except Exception as exc:
        raise _ops_error(exc)


@router.post("/ops/vendors", status_code=201)
def ops_create_vendor(payload: OpsVendorCreate, db: Session = Depends(get_db)):
    """Onboard a vendor into operations inventory."""
    from backend.ops.service import create_ops_vendor
    try:
        return create_ops_vendor(
            db, payload.name, payload.vendor_type, payload.contact_email, payload.phone
        )
    except Exception as exc:
        raise _ops_error(exc)


@router.get("/ops/vendors/{vendor_id}/assignments", response_model=VendorAssignmentsResponse)
def ops_vendor_assignments(vendor_id: str, db: Session = Depends(get_db)):
    """All activity assignments (and their trips) for one vendor."""
    from backend.ops.service import get_vendor_assignments
    try:
        return get_vendor_assignments(db, vendor_id)
    except Exception as exc:
        raise _ops_error(exc)


@router.get("/ops/vendors", response_model=List[OpsVendorRead])
def ops_list_vendors(vendor_type: Optional[str] = None, db: Session = Depends(get_db)):
    """Vendor inventory with live assignment counts."""
    from backend.ops.service import list_ops_vendors
    return list_ops_vendors(db, vendor_type)


@router.post("/ops/vendors/{vendor_id}/verify", response_model=OpsVendorRead)
def ops_verify_vendor(vendor_id: str, payload: VendorVerifyRequest, db: Session = Depends(get_db)):
    """Set vendor verified status (drives assignment eligibility)."""
    from backend.ops.service import set_vendor_verified
    try:
        return set_vendor_verified(db, vendor_id, payload.is_verified)
    except Exception as exc:
        raise _ops_error(exc)


@router.get("/ops/activity-inventory", response_model=List[ActivityInventoryRead])
def ops_activity_inventory(
    destination_id: Optional[str] = None, db: Session = Depends(get_db)
):
    """Bookable activity inventory for assignment pickers."""
    from backend.ops.service import list_activity_inventory
    return list_activity_inventory(db, destination_id)


@router.post("/trips/{trip_id}/confirm", response_model=TripConfirmResponse)
def confirm_trip_route(trip_id: str, payload: TripConfirmRequest, db: Session = Depends(get_db)):
    """Validate and persist traveler confirmation (planning -> confirmed).

    Idempotent: reconfirming returns the existing confirmed state without
    duplicating history or notifications. No ops assignments are fabricated;
    operators attach inventory through the consoles afterwards.
    """
    from backend.ops.service import confirm_trip
    try:
        result = confirm_trip(db, trip_id, payload.user_id)
    except Exception as exc:
        raise _ops_error(exc)
    trip = db.query(Trip).filter(Trip.id == trip_id).first()
    return {
        "status": "success",
        "already_confirmed": result["already_confirmed"],
        "confirmed_at": result["confirmed_at"],
        "trip": _trip_dict(trip, db),
    }


@router.post("/ops/trips/{trip_id}/approve", response_model=TripApprovalRead)
def ops_approve_trip(trip_id: str, payload: TripApprovalRequest, db: Session = Depends(get_db)):
    """Operator approves a traveler-confirmed trip (idempotent)."""
    from backend.ops.service import approve_trip
    try:
        return approve_trip(db, trip_id, payload.updated_by or "operator")
    except Exception as exc:
        raise _ops_error(exc)


@router.post("/ops/trips/{trip_id}/accept", response_model=TripApprovalRead)
def ops_accept_trip(trip_id: str, payload: TripApprovalRequest, db: Session = Depends(get_db)):
    """Start the assignment workflow (Accept & Assign). Requires approval."""
    from backend.ops.service import accept_trip_assignment
    try:
        return accept_trip_assignment(db, trip_id, payload.updated_by or "operator")
    except Exception as exc:
        raise _ops_error(exc)


@router.get("/ops/trips/{trip_id}/pipeline", response_model=TripPipelineResponse)
def ops_trip_pipeline(trip_id: str, db: Session = Depends(get_db)):
    """Combined approval + service-assignment state for the Assignment Center."""
    from backend.ops.service import get_trip_pipeline
    try:
        return get_trip_pipeline(db, trip_id)
    except Exception as exc:
        raise _ops_error(exc)


@router.get("/ops/approvals", response_model=List[TripApprovalRead])
def ops_list_approvals(db: Session = Depends(get_db)):
    """All operator pipeline states (drives dashboard Incoming/Active splits)."""
    from backend.ops.service import list_trip_approvals
    return list_trip_approvals(db)


@router.post("/ops/trips/{trip_id}/finalize", response_model=TripFinalizeResponse)
def ops_finalize_trip(trip_id: str, payload: TripFinalizeRequest, db: Session = Depends(get_db)):
    """Finalize an approved trip: validate completeness, lock, notify."""
    from backend.ops.service import finalize_trip
    try:
        return finalize_trip(db, trip_id, payload.require_activities, payload.updated_by or "operator")
    except Exception as exc:
        raise _ops_error(exc)


@router.get("/ops/messages/overview", response_model=List[TripMessageOverviewEntry])
def ops_messages_overview(db: Session = Depends(get_db)):
    """Per-trip internal message counts (drives the Communications trip list)."""
    from backend.ops.service import trip_messages_overview
    try:
        return trip_messages_overview(db)
    except Exception as exc:
        raise _ops_error(exc)


@router.get("/ops/trips/{trip_id}/messages", response_model=List[TripMessageRead])
def ops_list_trip_messages(
    trip_id: str, category: Optional[str] = None, db: Session = Depends(get_db)
):
    """Chronological internal operator messages for one trip, optionally by category."""
    from backend.ops.service import list_trip_messages
    try:
        return list_trip_messages(db, trip_id, category)
    except Exception as exc:
        raise _ops_error(exc)


@router.post("/ops/trips/{trip_id}/messages", response_model=TripMessageRead, status_code=201)
def ops_create_trip_message(
    trip_id: str, payload: TripMessageCreate, db: Session = Depends(get_db)
):
    """Persist one internal operator message for a trip (traveler-invisible)."""
    from backend.ops.service import OpsValidation, create_trip_message
    try:
        if payload.trip_id.strip() != trip_id.strip():
            raise OpsValidation("payload trip_id must match the path trip_id")
        return create_trip_message(
            db,
            trip_id,
            payload.body,
            payload.operator_name,
            payload.category,
            payload.is_urgent,
        )
    except Exception as exc:
        raise _ops_error(exc)


# ----------------------------------------------------
# Activities API
# ----------------------------------------------------
@router.get("/activities", response_model=List[ActivityRead])
def get_activities(
    destination_id: Optional[str] = None,
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Retrieve curated activities with optional destination and category filters."""
    query = db.query(Activity).filter(Activity.is_active == True)
    if destination_id:
        query = query.filter(Activity.destination_id == destination_id)
    if category:
        query = query.filter(Activity.category == category)
    return query.order_by(Activity.rating.desc()).all()

# ----------------------------------------------------
# Transport API
# ----------------------------------------------------
@router.get("/transport", response_model=List[TransportRead])
def get_transport_options(
    destination_id: Optional[str] = None,
    type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Retrieve transport options with optional filters."""
    query = db.query(TransportOption).filter(TransportOption.is_active == True)
    if destination_id:
        query = query.filter(TransportOption.destination_id == destination_id)
    if type:
        query = query.filter(TransportOption.type == type)
    return query.all()

# ----------------------------------------------------
# Trips API (The Central Entity)
# ----------------------------------------------------
@router.post("/trips")
def create_trip(trip_in: TripCreate, request: Request, db: Session = Depends(get_db)):
    """
    Create a new Trip entity with attached preferences, change history,
    and automatic initial itinerary generation.

    Ownership: a valid traveler session (Bearer JWT) wins over any
    client-sent user_id so the trip belongs to the logged-in user and the
    AI Guide can see it. Anonymous creation keeps the legacy fallback.
    """
    # 1. Ensure user exists or use default primary traveler
    user_id = trip_in.user_id
    try:
        header = (request.headers.get("authorization") or "") if request else ""
        scheme, _, token = header.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            from backend.auth.service import parse_traveler_token
            session_user_id = parse_traveler_token(token.strip())
            if db.query(User).filter(User.id == session_user_id,
                                     User.is_active == True).first():  # noqa: E712
                user_id = session_user_id
    except Exception:
        pass
    if not user_id:
        primary_user = db.query(User).first()
        if not primary_user:
            primary_user = User(
                email="alex.traveler@example.com",
                full_name="Alex Morgan",
                role="traveler"
            )
            db.add(primary_user)
            db.commit()
            db.refresh(primary_user)
        user_id = primary_user.id

    # 2. Resolve the planner's destination name/embedded destination to the
    # persisted catalog rather than accepting a disconnected frontend object.
    destination = _resolve_or_discover_destination(db, trip_in)
    duration_days = trip_in.duration_days or 4
    # Dates are the source of truth: a 16-21 Oct range is 6 days even if the
    # caller sent duration_days=3. Derive it so generation covers the range.
    if trip_in.start_date and trip_in.end_date:
        if trip_in.end_date < trip_in.start_date:
            raise HTTPException(status_code=422, detail="Trip end date is before start date")
        duration_days = max(
            1, (trip_in.end_date.date() - trip_in.start_date.date()).days + 1)
    currency = trip_in.currency or "INR"
    traveler_count = trip_in.traveler_count or 2
    try:
        _validate_generation_inventory(db, destination, currency, traveler_count, duration_days)
    except HTTPException as exc:
        # Curated catalog has gaps: complete them from configured live
        # providers (SerpApi hotels, OSM/Wikimedia places) and re-check.
        # Transport has no live provider and is traveler-arranged, so its
        # absence no longer blocks creation. Without a SerpApi key, hotels
        # cannot be filled, so the original error stands as well.
        if exc.status_code != 422 or not (settings.SERPAPI_API_KEY or "").strip():
            raise
        from backend.live_fill.service import fill_destination_inventory
        filled = fill_destination_inventory(
            db, destination, currency=currency, traveler_count=traveler_count,
            duration_days=duration_days,
            start_date=trip_in.start_date, end_date=trip_in.end_date,
        )
        try:
            _validate_generation_inventory(db, destination, currency, traveler_count, duration_days)
        except HTTPException as retry_exc:
            hint = ""
            hotel_error = str(filled.get("hotel_error") or "")
            if "429" in hotel_error:
                hint = (" Live hotel search is over its provider quota right now; "
                        "anything already found is saved, retry after the quota resets.")
            elif filled.get("hotel_error") or filled.get("activity_error"):
                hint = " Live providers were temporarily unreachable; retry in a bit."
            if hint:
                raise HTTPException(status_code=retry_exc.status_code,
                                    detail=f"{retry_exc.detail}{hint}")
            raise

    # 3. Create Trip entity
    trip = Trip(
        user_id=user_id,
        destination_id=destination.id,
        title=trip_in.title,
        status=trip_in.status or "planning",
        start_date=trip_in.start_date,
        end_date=trip_in.end_date,
        duration_days=duration_days,
        total_budget=trip_in.total_budget or 50000.0,
        currency=currency,
        traveler_count=traveler_count,
        pace=trip_in.pace or "balanced",
        discovery_session_id=destination.discovery_session_id if destination.inventory_source == "discovered" else None
    )
    db.add(trip)
    db.commit()
    db.refresh(trip)

    # 3. Create Preferences
    prefs_in = trip_in.preferences
    pref = TripPreference(
        trip_id=trip.id,
        budget_tier=prefs_in.budget_tier if prefs_in else "moderate",
        interests=prefs_in.interests if prefs_in else ["nature", "culture"],
        travel_companions=prefs_in.travel_companions if prefs_in else "couple",
        accommodation_types=prefs_in.accommodation_types if prefs_in else ["boutique"],
        transport_preferences=prefs_in.transport_preferences if prefs_in else ["private_suv"],
        dietary_requirements=prefs_in.dietary_requirements if prefs_in else [],
        special_requests=prefs_in.special_requests if prefs_in else None
    )
    db.add(pref)

    # 4. Add Change History Entry
    history = ChangeHistory(
        trip_id=trip.id,
        changed_by="user",
        action="trip_created",
        field_changed="all",
        new_value=trip.title,
        reason="Initial trip initialized via TourFlow AI Planner"
    )
    db.add(history)

    # 5. Add Welcome Notification
    notification = Notification(
        trip_id=trip.id,
        user_id=user_id,
        title="Trip Created Successfully",
        message=f"Your trip '{trip.title}' is initialized. AI itinerary is generated and ready for customization.",
        type="success"
    )
    db.add(notification)
    db.commit()

    # 6. Generate initial itinerary items
    generator = ItineraryGenerator(db)
    try:
        generator.generate_for_trip(trip.id)
    except ItineraryGenerationError as exc:
        db.rollback()
        db.query(ChangeHistory).filter(ChangeHistory.trip_id == trip.id).delete()
        db.query(Notification).filter(Notification.trip_id == trip.id).delete()
        db.query(TripPreference).filter(TripPreference.trip_id == trip.id).delete()
        db.query(ItineraryItem).filter(ItineraryItem.trip_id == trip.id).delete()
        db.query(Trip).filter(Trip.id == trip.id).delete()
        db.commit()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    db.refresh(trip)
    return _trip_dict(trip, db)

@router.get("/trips/{trip_id}")
def get_trip(trip_id: str, db: Session = Depends(get_db)):
    """
    Retrieve full Trip entity with all associated sub-entities:
    traveler, preferences, itinerary items, bookings, alerts, notifications, change history, reviews.
    """
    trip = db.query(Trip).filter(Trip.id == trip_id).first()
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    return _trip_dict(trip, db)

@router.put("/trips/{trip_id}")
def update_trip(trip_id: str, trip_in: TripUpdate, db: Session = Depends(get_db)):
    """Update Trip attributes and record change history."""
    trip = db.query(Trip).filter(Trip.id == trip_id).first()
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")

    update_data = trip_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        old_val = str(getattr(trip, field, ""))
        setattr(trip, field, value)
        # Log to change history
        history = ChangeHistory(
            trip_id=trip.id,
            changed_by="user",
            action=f"update_{field}",
            field_changed=field,
            old_value=old_val,
            new_value=str(value),
            reason=f"Traveler modified {field}"
        )
        db.add(history)

    db.commit()
    db.refresh(trip)
    return _trip_dict(trip, db)

@router.get("/trips/{trip_id}/preferences", response_model=TripPreferenceRead)
def get_trip_preferences(trip_id: str, db: Session = Depends(get_db)):
    """Get the preferences for a specific trip."""
    pref = db.query(TripPreference).filter(TripPreference.trip_id == trip_id).first()
    if not pref:
        raise HTTPException(status_code=404, detail="Trip preferences not found")
    return pref

@router.put("/trips/{trip_id}/preferences", response_model=TripPreferenceRead)
def update_trip_preferences(
    trip_id: str,
    pref_in: TripPreferenceUpdate,
    db: Session = Depends(get_db)
):
    """Update trip preferences and log change history."""
    pref = db.query(TripPreference).filter(TripPreference.trip_id == trip_id).first()
    if not pref:
        # Create if doesn't exist yet
        pref = TripPreference(trip_id=trip_id)
        db.add(pref)

    update_data = pref_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(pref, field, value)

    # Log change
    history = ChangeHistory(
        trip_id=trip_id,
        changed_by="user",
        action="preferences_updated",
        field_changed="preferences",
        new_value=str(update_data),
        reason="Traveler customized travel preferences and constraints"
    )
    db.add(history)

    db.commit()
    db.refresh(pref)
    return pref

# ----------------------------------------------------
# AI Planning & Gemini Intelligence APIs
# ----------------------------------------------------
def _operator_assistant_response(result: AssistantChatResult) -> Dict[str, Any]:
    return {
        "reply": result.response,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "suggested_actions": result.suggested_actions,
        "trip_id": result.trip_id,
        "source": result.source,
        "context_validated": result.context_validated,
        "references": [ref.model_dump() for ref in result.references],
    }


@router.post("/operator/ai-assistant")
def operator_ai_assistant(payload: AIChatRequest, db: Session = Depends(get_db)):
    """
    Operator-facing Gemini assistant.

    The frontend historically called /api/operator/ai-assistant while the
    FastAPI backend exposed only /api/ai/chat. Keep the operator route as a
    compatibility layer and return the response shape expected by the UI.
    """
    context_trip_id = payload.trip_id
    if not context_trip_id and payload.current_trip:
        current_trip_id = payload.current_trip.get("id") or payload.current_trip.get("trip_id")
        if current_trip_id:
            context_trip_id = str(current_trip_id)

    if context_trip_id:
        assistant_payload = AssistantChatContext(trip_id=context_trip_id, message=payload.message)
        service = AssistantService(db, gemini_service)
        try:
            return _operator_assistant_response(service.execute(assistant_payload))
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except AssistantValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AssistantExecutionError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    context = dict(payload.session_context or {})
    result = gemini_service.chat(
        message=payload.message,
        session_context=context
    )

    return {
        "reply": result.get("response", ""),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "suggested_actions": result.get("suggestions", []),
    }
@router.post("/ai/chat", response_model=AIChatResponse)
def ai_chat(payload: AIChatRequest, db: Session = Depends(get_db)):
    """Conversational AI endpoint for travel consultation."""
    context = dict(payload.session_context or {})
    if payload.current_trip:
        context["current_trip"] = payload.current_trip
    if payload.history:
        context["history"] = payload.history
    # Inject verified weather if destination resolvable (never fabricated)
    dest_for_weather = None
    if isinstance(payload.destination_id, str) and payload.destination_id.strip():
        try:
            from backend.models.models import Destination
            drow = db.query(Destination).filter((Destination.id==payload.destination_id.strip()) | (Destination.slug==payload.destination_id.strip())).first()
            if drow: dest_for_weather = drow.name
        except: pass
    if not dest_for_weather:
        # try session_context destination
        dc = context.get("destination") or context.get("current_trip", {}).get("destination") if isinstance(context.get("current_trip"), dict) else None
        if isinstance(dc, dict): dc = dc.get("name")
        if isinstance(dc, str) and dc.strip(): dest_for_weather = dc.strip()
    if dest_for_weather and settings.WEATHER_BASE_URL.strip():
        try:
            from backend.weather.service import fetch_weather, resolve_coordinates
            lat, lon, _ = resolve_coordinates(dest_for_weather, None, None, db, settings.NOMINATIM_API_URL, settings.PLACES_TIMEOUT_S)
            if lat is not None and lon is not None:
                w = fetch_weather(lat, lon, settings.WEATHER_BASE_URL, settings.WEATHER_TIMEOUT_S, days=5, api_key=settings.WEATHER_API_KEY)
                context["verified_weather"] = {"destination": dest_for_weather, "current": w["current"], "forecast": w["forecast"], "source": w["source"], "retrieved_at": w["retrieved_at"]}
                context["weather_instruction"] = "Distinguish live weather (current), forecast (future dates), and historical/static info. Never invent temperature."
        except Exception:
            pass
    result = gemini_service.chat(
        message=payload.message,
        session_context=context
    )
    return result

@router.post("/ai/extract-preferences")
def ai_extract_preferences(payload: AIExtractPreferencesRequest):
    """Extract structured travel parameters from natural language prompts."""
    return gemini_service.extract_preferences(
        text_prompt=payload.text_prompt,
        context=payload.context
    )

@router.post("/research", response_model=ResearchResult)
def research_destination(payload: ResearchContext, db: Session = Depends(get_db)):
    """Return non-mutating, catalog-grounded destination research."""
    service = DestinationResearchService(db, gemini_service)
    try:
        return service.execute(payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ResearchExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/accommodations/recommendations", response_model=AccommodationResult)
def recommend_accommodations(payload: AccommodationContext, db: Session = Depends(get_db)):
    """Return read-only, catalog-validated accommodation recommendations."""
    service = AccommodationRecommendationService(db, gemini_service)
    try:
        return service.execute(payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AccommodationExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/transportation/recommendations", response_model=TransportationResult)
def recommend_transportation(payload: TransportationContext, db: Session = Depends(get_db)):
    """Return read-only, catalog-validated transportation recommendations."""
    service = TransportationRecommendationService(db, gemini_service)
    try:
        return service.execute(payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TransportationExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/experiences/recommendations", response_model=ExperienceResult)
def recommend_experiences(payload: ExperienceContext, db: Session = Depends(get_db)):
    """Return read-only, catalog-validated experience recommendations."""
    service = ExperienceRecommendationService(db, gemini_service)
    try:
        return service.execute(payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ExperienceExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/itinerary/recommendations", response_model=ItineraryResult)
def recommend_itinerary(payload: ItineraryContext, db: Session = Depends(get_db)):
    """Return a read-only, catalog-validated multi-day itinerary recommendation."""
    service = ItineraryRecommendationService(db, gemini_service)
    try:
        return service.execute(payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ItineraryValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ItineraryExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/trips/management/recommendations", response_model=TripManagementResult)
def recommend_trip_management(payload: TripManagementContext, db: Session = Depends(get_db)):
    """Return a read-only, catalog-validated, booking-ready trip-management view."""
    service = TripManagementService(db, gemini_service)
    try:
        return service.execute(payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TripManagementValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except TripManagementExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/bookings/recommendations", response_model=BookingRecommendationResult)
def recommend_bookings(payload: BookingRecommendationContext, db: Session = Depends(get_db)):
    """Return a read-only, catalog-validated booking-readiness view for a trip."""
    service = BookingRecommendationService(db, gemini_service)
    try:
        return service.execute(payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BookingRecommendationValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except BookingRecommendationExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/assistant/chat", response_model=AssistantChatResult)
def assistant_chat(payload: AssistantChatContext, db: Session = Depends(get_db)):
    """Answer a traveler question from read-only, validated trip context."""
    service = AssistantService(db, gemini_service)
    try:
        return service.execute(payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AssistantValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AssistantExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

@router.post("/ai/recommend")
def ai_recommend(payload: AIRecommendRequest, db: Session = Depends(get_db)):
    """Get AI recommendation insights alongside matching catalogue items."""
    engine = RecommendationEngine(db)
    return engine.get_recommendations(
        destination_id=payload.destination_id,
        preferences=payload.preferences
    )

@router.post("/ai/generate-itinerary")
def ai_generate_itinerary(payload: AIGenerateItineraryRequest, db: Session = Depends(get_db)):
    """Generate dynamic day-by-day itinerary schema."""
    if payload.trip_id:
        generator = ItineraryGenerator(db)
        try:
            items = generator.generate_for_trip(payload.trip_id)
        except ItineraryGenerationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        trip = db.query(Trip).filter(Trip.id == payload.trip_id).first()
        return {
            "status": "success",
            "trip_id": payload.trip_id,
            "items_count": len(items),
            "trip": trip
        }
    raise HTTPException(status_code=422, detail="trip_id is required for catalog-grounded itinerary generation")


@router.post("/trips/{trip_id}/optimize")
def optimize_trip_itinerary(trip_id: str, db: Session = Depends(get_db)):
    """Replace proposed catalog selections with deterministic ranked selections."""
    trip = _trip_or_404(db, trip_id)
    try:
        items = ItineraryGenerator(db).optimize_for_trip(trip.id)
    except ItineraryGenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    db.refresh(trip)
    return {
        "status": "success",
        "trip_id": trip.id,
        "items_count": len(items),
        "trip": _trip_dict(trip, db),
    }


@router.post("/ai/replan")
def ai_replan(payload: AIReplanRequest, db: Session = Depends(get_db)):
    """Dynamically adjust itinerary based on external disruption event."""
    engine = ReplanningEngine(db)
    return engine.handle_disruption(
        trip_id=payload.trip_id,
        trigger_event=payload.trigger_event
    )


# ---------------------------------------------------------------------------
# PostgreSQL-backed compatibility routes used by the current React client.
# These replace Express's in-memory implementations without inventing data.
# ---------------------------------------------------------------------------
@router.get("/trips")
def list_trips(status: Optional[str] = None, search: Optional[str] = None,
               operator_id: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Trip)
    if status:
        query = query.filter(Trip.status == status)
    if search:
        like = f"%{search}%"
        query = query.filter((Trip.title.ilike(like)) | (Trip.id.ilike(like)))
    # The persisted schema has no operator assignment. Keep this query accepted
    # for client compatibility, but never fabricate an assignment.
    return [_trip_dict(t, db) for t in query.order_by(Trip.updated_at.desc()).all()]


@router.delete("/trips/{trip_id}")
def delete_trip(trip_id: str, db: Session = Depends(get_db)):
    trip = _trip_or_404(db, trip_id)
    db.delete(trip)
    db.commit()
    return {"success": True, "message": "Trip deleted"}


@router.post("/trips/{trip_id}/trigger-disruption")
def trigger_disruption(trip_id: str, db: Session = Depends(get_db)):
    trip = _trip_or_404(db, trip_id)
    alert = Alert(trip_id=trip.id, alert_type="weather", severity="critical",
                  title="Operational disruption reported",
                  description="A disruption was reported and requires itinerary review.")
    db.add(alert)
    _record_change(db, trip, "disruption_triggered", "alerts", alert.title, alert.description, "operator")
    db.commit(); db.refresh(trip)
    return {"success": True, "trip": _trip_dict(trip, db)}


@router.post("/trips/{trip_id}/impact-analysis")
def impact_analysis(trip_id: str, payload: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    trip = _trip_or_404(db, trip_id)
    disruption = payload.get("disruption") or {}
    unresolved = [a for a in trip.alerts if not a.is_resolved]
    affected = [i for i in trip.itinerary if i.status in ("proposed", "confirmed")]
    exposure = sum(b.amount for b in trip.bookings if b.status in ("pending", "cancelled"))
    return {"trip_id": trip.id, "disruption": disruption, "affected_items_count": len(affected),
            "unresolved_alerts_count": len(unresolved), "financial_exposure": {
                "unfulfilled_booking_cost": exposure, "currency": trip.currency},
            "recommendation": "Review active itinerary items and select a database-backed alternative."}


@router.post("/trips/{trip_id}/ai-replan-options")
def ai_replan_options(trip_id: str, payload: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    trip = _trip_or_404(db, trip_id)
    if not trip.destination_id:
        raise HTTPException(status_code=400, detail="Trip has no destination")
    active_ids = {i.activity_id for i in trip.itinerary if i.activity_id}
    candidates = db.query(Activity).filter(Activity.destination_id == trip.destination_id,
                                           Activity.is_active == True).all()
    return {"trip_id": trip.id, "candidates": [{"id": a.id, "title": a.title,
            "description": a.description, "cost": a.price_per_person * trip.traveler_count,
            "vendor_name": a.vendor.name if a.vendor else None, "location": a.meeting_point,
            "match_score": 1.0 if a.id not in active_ids else 0.7,
            "ai_rationale": "Available catalog activity for this trip destination."} for a in candidates]}


@router.post("/trips/{trip_id}/apply-replan")
def apply_replan(trip_id: str, payload: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    trip = _trip_or_404(db, trip_id)
    activity_id = payload.get("alternative_id")
    activity = db.query(Activity).filter(Activity.id == activity_id, Activity.is_active == True).first()
    if not activity or activity.destination_id != trip.destination_id:
        raise HTTPException(status_code=400, detail="Replan alternative is unavailable for this trip")
    target = next((i for i in trip.itinerary if i.item_type == "activity" and i.status != "completed"), None)
    if target is None:
        target = ItineraryItem(trip_id=trip.id, day_number=1, order_index=len(trip.itinerary) + 1,
                               item_type="activity", title=activity.title)
        db.add(target)
    target.activity_id, target.title, target.description = activity.id, activity.title, activity.description
    target.location, target.cost, target.status = activity.meeting_point, activity.price_per_person * trip.traveler_count, "confirmed"
    for alert in trip.alerts:
        if not alert.is_resolved:
            alert.is_resolved = True
    _record_change(db, trip, "replan_applied", "itinerary", activity.title,
                   payload.get("notes") or "Operator approved an available destination activity.", "operator")
    db.add(Notification(trip_id=trip.id, user_id=trip.user_id, title="Itinerary updated",
                        message=f"Your itinerary has been updated to include {activity.title}.", type="update"))
    db.commit(); db.refresh(trip)
    return {"success": True, "summary": {"new_activity": activity.title,
            "booking_reference": None, "cost_savings": 0}, "trip": _trip_dict(trip, db)}


def _set_trip_request_status(trip_id: str, status: str, db: Session):
    trip = _trip_or_404(db, trip_id)
    trip.status = status
    _record_change(db, trip, f"request_{status}", "status", status, f"Trip request {status} by operator.", "operator")
    db.commit(); db.refresh(trip)
    return {"success": True, "trip": _trip_dict(trip, db)}


@router.post("/trips/{trip_id}/accept-request")
def accept_trip_request(trip_id: str, db: Session = Depends(get_db)):
    return _set_trip_request_status(trip_id, "confirmed", db)


@router.post("/trips/{trip_id}/decline-request")
def decline_trip_request(trip_id: str, db: Session = Depends(get_db)):
    return _set_trip_request_status(trip_id, "cancelled", db)


@router.get("/operator/dashboard")
def operator_dashboard(db: Session = Depends(get_db)):
    trips = db.query(Trip).all()
    return {"total_trips": len(trips), "planning_trips": sum(t.status == "planning" for t in trips),
            "active_trips": sum(t.status in ("confirmed", "ongoing") for t in trips),
            "unresolved_alerts": db.query(Alert).filter(Alert.is_resolved == False).count(),
            "pending_bookings": db.query(Booking).filter(Booking.status == "pending").count()}


def _vendor_dict(vendor: Any) -> Dict[str, Any]:
    return {"id": vendor.id, "name": vendor.name, "category": vendor.vendor_type,
            "location": None, "phone": vendor.phone, "contact_person": vendor.contact_email,
            "rating": vendor.rating, "is_available": vendor.is_verified,
            "active_bookings_count": sum(b.status in ("pending", "confirmed") for b in vendor.bookings)}


@router.get("/operator/vendors")
def operator_vendors(db: Session = Depends(get_db)):
    from backend.models.models import Vendor
    return [_vendor_dict(v) for v in db.query(Vendor).order_by(Vendor.name).all()]


@router.post("/operator/vendors/{vendor_id}/toggle")
def toggle_vendor(vendor_id: str, db: Session = Depends(get_db)):
    from backend.models.models import Vendor
    vendor = db.query(Vendor).filter(Vendor.id == vendor_id).first()
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")
    vendor.is_verified = not vendor.is_verified
    db.commit(); db.refresh(vendor)
    return _vendor_dict(vendor)


def _booking_dict(booking: Booking) -> Dict[str, Any]:
    return {"id": booking.id, "trip_id": booking.trip_id, "vendor_id": booking.vendor_id,
            "booking_reference": booking.booking_reference, "item_type": booking.item_type,
            "item_id": booking.item_id, "amount": booking.amount, "currency": booking.currency,
            "status": booking.status, "payment_status": booking.payment_status,
            "booking_date": booking.booking_date.isoformat()}


@router.get("/operator/bookings")
def operator_bookings(db: Session = Depends(get_db)):
    return [_booking_dict(b) for b in db.query(Booking).order_by(Booking.booking_date.desc()).all()]


@router.post("/operator/bookings/{booking_id}/action")
def operator_booking_action(booking_id: str, payload: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    booking = db.query(Booking).filter(Booking.id == booking_id).first()
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    action = payload.get("action")
    if action not in {"confirm", "cancel", "rebook"}:
        raise HTTPException(status_code=422, detail="Action must be confirm, cancel, or rebook")
    booking.status = {"confirm": "confirmed", "cancel": "cancelled", "rebook": "pending"}[action]
    _record_change(db, booking.trip, f"booking_{action}", "booking", booking.booking_reference,
                   f"Operator requested booking {action}.", "operator")
    db.commit(); db.refresh(booking)
    return _booking_dict(booking)


@router.get("/operator/alerts")
def operator_alerts(db: Session = Depends(get_db)):
    return [{"id": a.id, "trip_id": a.trip_id, "alert_type": a.alert_type, "severity": a.severity,
             "title": a.title, "description": a.description, "is_resolved": a.is_resolved,
             "created_at": a.created_at.isoformat()} for a in db.query(Alert).order_by(Alert.created_at.desc()).all()]


@router.post("/operator/alerts/{alert_id}/resolve")
def resolve_operator_alert(alert_id: str, db: Session = Depends(get_db)):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.is_resolved = True
    _record_change(db, alert.trip, "alert_resolved", "alert", alert.title, "Operator resolved alert.", "operator")
    db.commit()
    return {"success": True, "alert_id": alert.id}


@router.get("/operator/analytics")
def operator_analytics(db: Session = Depends(get_db)):
    trips = db.query(Trip).all()
    bookings = db.query(Booking).all()
    return {"overview": {"total_tours_operated": len(trips),
            "active_tours": sum(t.status in ("confirmed", "ongoing") for t in trips),
            "total_travelers_hosted": sum(t.traveler_count for t in trips),
            "total_gross_revenue": sum(b.amount for b in bookings if b.status == "confirmed"),
            "disruption_recovery_rate": 0 if not db.query(Alert).count() else round(100 * db.query(Alert).filter(Alert.is_resolved == True).count() / db.query(Alert).count(), 1)}}


@router.post("/auth/operator-login")
def operator_login(payload: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    email, password = (payload.get("email") or "").strip().lower(), payload.get("password") or ""
    configured_password = os.getenv("OPERATOR_LOGIN_PASSWORD")
    if not configured_password:
        raise HTTPException(status_code=503, detail="Operator login is not configured")
    user = db.query(User).filter(User.email == email, User.role.in_(["operator", "admin"]), User.is_active == True).first()
    if not user or not secrets.compare_digest(password, configured_password):
        raise HTTPException(status_code=401, detail="Invalid operator credentials")
    return {"success": True, "user": {"id": user.id, "email": user.email, "name": user.full_name,
            "role": user.role, "operator_name": user.full_name}, "token": secrets.token_urlsafe(32)}


def _commit_trip(db: Session, trip: Trip, action: str, field: str, value: str, reason: str) -> Dict[str, Any]:
    _record_change(db, trip, action, field, value, reason)
    db.commit(); db.refresh(trip)
    return _trip_dict(trip, db)


@router.post("/trips/{trip_id}/change-transport")
def change_transport(trip_id: str, payload: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    trip = _trip_or_404(db, trip_id)
    transport = db.query(TransportOption).filter(TransportOption.id == payload.get("transport_id"),
                                                  TransportOption.destination_id == trip.destination_id,
                                                  TransportOption.is_active == True).first()
    if not transport:
        raise HTTPException(status_code=400, detail="Transport option not found")
    item = next((i for i in trip.itinerary if i.item_type == "transport"), None)
    if not item:
        item = ItineraryItem(trip_id=trip.id, day_number=1, order_index=1, item_type="transport", title=transport.name)
        db.add(item)
    item.transport_id, item.title, item.description, item.cost, item.status = transport.id, transport.name, f"{transport.route_from} to {transport.route_to}", transport.price, "confirmed"
    return _commit_trip(db, trip, "transport_changed", "transport", transport.name, "Traveler selected a catalog transport option.")


def _change_accommodation(trip_id: str, payload: Dict[str, Any], db: Session, daily: bool) -> Dict[str, Any]:
    trip = _trip_or_404(db, trip_id)
    hotel = db.query(Hotel).filter(Hotel.id == payload.get("accommodation_id"), Hotel.destination_id == trip.destination_id,
                                   Hotel.is_active == True).first()
    if not hotel:
        raise HTTPException(status_code=400, detail="Accommodation option not found")
    day = int(payload.get("day_number") or 1) if daily else 1
    item = next((i for i in trip.itinerary if i.item_type == "hotel" and i.day_number == day), None)
    if not item:
        item = ItineraryItem(trip_id=trip.id, day_number=day, order_index=99, item_type="hotel", title=hotel.name)
        db.add(item)
    item.hotel_id, item.title, item.description, item.location, item.cost, item.status = hotel.id, hotel.name, hotel.description, hotel.address, hotel.price_per_night, "confirmed"
    return _commit_trip(db, trip, "hotel_changed", "hotel", hotel.name, "Traveler selected a catalog hotel.")


@router.post("/trips/{trip_id}/change-accommodation")
def change_accommodation(trip_id: str, payload: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    return _change_accommodation(trip_id, payload, db, False)


@router.post("/trips/{trip_id}/change-daily-accommodation")
@router.post("/trips/{trip_id}/change-day-accommodation")
def change_daily_accommodation(trip_id: str, payload: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    return _change_accommodation(trip_id, payload, db, True)


@router.post("/trips/{trip_id}/select-hotel")
def select_hotel(trip_id: str, selection: SelectHotelRequest, db: Session = Depends(get_db)):
    """Persist a traveler-selected (e.g. SerpApi live) hotel against the trip.

    Reuses the existing itinerary hotel-item structure: hotel_id stays None
    (no catalog row exists) while the property token, dates, price, and
    provider metadata are retained in meta_data for later identification.
    """
    trip = _trip_or_404(db, trip_id)
    total = selection.total_price
    if total is None and selection.price_per_night is not None:
        total = selection.price_per_night * max(1, (trip.duration_days or 2) - 1)
    item = next((i for i in trip.itinerary if i.item_type == "hotel" and i.day_number == selection.day_number), None)
    if not item:
        item = ItineraryItem(trip_id=trip.id, day_number=selection.day_number, order_index=99,
                             item_type="hotel", title=selection.name)
        db.add(item)
    item.hotel_id = None
    item.title = selection.name
    item.description = selection.description
    item.location = selection.location
    item.cost = float(total or 0)
    item.status = "confirmed"
    item.meta_data = {"provider": "serpapi", "property_token": selection.property_token,
                      "name": selection.name, "location": selection.location,
                      "image_url": selection.image_url, "description": selection.description,
                      "price_per_night": selection.price_per_night, "total_price": total,
                      "currency": (selection.currency or trip.currency or "INR").upper(),
                      "rating": selection.rating, "hotel_class": selection.hotel_class,
                      "amenities": selection.amenities or [],
                      "check_in_date": selection.check_in_date, "check_out_date": selection.check_out_date}
    return _commit_trip(db, trip, "hotel_selected", "hotel", selection.name,
                        "Traveler selected a live hotel search result.")


@router.post("/trips/{trip_id}/add-activity")
def add_activity(trip_id: str, payload: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    trip = _trip_or_404(db, trip_id)
    title = (payload.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=422, detail="Activity title is required")
    day = int(payload.get("day_number") or 1)
    item = ItineraryItem(trip_id=trip.id, day_number=day,
        order_index=max([i.order_index for i in trip.itinerary if i.day_number == day] or [0]) + 1,
        item_type=payload.get("item_type") or "activity", title=title, description=payload.get("description"),
        start_time=payload.get("start_time"), end_time=payload.get("end_time"), cost=float(payload.get("cost") or 0),
        location=payload.get("location"), status="confirmed",
        meta_data={"ui": {k: payload[k] for k in ("image_url", "duration", "walking_intensity", "rest_buffer_minutes") if k in payload}})
    db.add(item)
    return _commit_trip(db, trip, "item_added", "itinerary", title, "Traveler added a custom itinerary activity.")


@router.post("/trips/{trip_id}/delete-activity")
def delete_activity(trip_id: str, payload: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    trip = _trip_or_404(db, trip_id)
    item = db.query(ItineraryItem).filter(ItineraryItem.id == payload.get("item_id"), ItineraryItem.trip_id == trip.id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Itinerary item not found")
    title = item.title; db.delete(item)
    return _commit_trip(db, trip, "item_deleted", "itinerary", title, "Traveler deleted an itinerary item.")


@router.post("/trips/{trip_id}/swap-activity")
def swap_activity(trip_id: str, payload: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    trip = _trip_or_404(db, trip_id)
    item = db.query(ItineraryItem).filter(ItineraryItem.id == payload.get("item_id"), ItineraryItem.trip_id == trip.id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Itinerary item not found")
    for field in ("title", "description", "cost"):
        if payload.get(f"new_{field}") is not None: setattr(item, field, payload[f"new_{field}"])
    if payload.get("new_image_url"):
        item.meta_data = {**(item.meta_data or {}), "ui": {**(item.meta_data or {}).get("ui", {}), "image_url": payload["new_image_url"]}}
    return _commit_trip(db, trip, "activity_swapped", "itinerary", item.title, "Traveler swapped an itinerary activity.")


@router.post("/trips/{trip_id}/edit-activity")
def edit_activity(trip_id: str, payload: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    trip = _trip_or_404(db, trip_id)
    item = db.query(ItineraryItem).filter(ItineraryItem.id == payload.get("item_id"), ItineraryItem.trip_id == trip.id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Itinerary item not found")
    for field in ("title", "description", "start_time", "end_time", "cost"):
        if field in payload and payload[field] is not None: setattr(item, field, payload[field])
    return _commit_trip(db, trip, "activity_edited", "itinerary", item.title, "Traveler edited an itinerary activity.")


def _verify_trip_ownership_optional(request: Request, db: Session, trip: Trip):
    """If request carries a valid traveler token, enforce ownership; otherwise allow open operator flow."""
    auth = request.headers.get("authorization") or ""
    if not auth.lower().startswith("bearer "):
        return
    token = auth.split(" ", 1)[1].strip()
    if not token:
        return
    try:
        from backend.auth.service import _traveler_from_token
        user = _traveler_from_token(db, token)
        if trip.user_id != user.id:
            raise HTTPException(status_code=403, detail="Trip belongs to another traveler")
    except HTTPException:
        raise
    except Exception:
        # invalid token -> treat as open (do not block operator flow); but if token was present and invalid, still 401 for authenticated traveler book? For restaurant, keep open.
        return


def _parse_restaurant_cost(price_for_two: Any) -> float:
    if price_for_two is None:
        return 0.0
    if isinstance(price_for_two, (int, float)):
        try:
            v = float(price_for_two)
            return v if v >= 0 else 0.0
        except:
            return 0.0
    if isinstance(price_for_two, str):
        import re
        m = re.search(r"[\d,.]+", price_for_two)
        if not m:
            return 0.0
        try:
            v = float(m.group().replace(",", ""))
            return v if v >= 0 else 0.0
        except:
            return 0.0
    return 0.0


@router.post("/trips/{trip_id}/add-restaurant")
def add_restaurant(trip_id: str, payload: AddRestaurantRequest, request: Request, db: Session = Depends(get_db)):
    trip = _trip_or_404(db, trip_id)
    _verify_trip_ownership_optional(request, db, trip)
    if payload.day_number < 1 or payload.day_number > (trip.duration_days or 1):
        raise HTTPException(status_code=422, detail=f"day_number must be between 1 and {trip.duration_days}")
    # name already validated by schema min_length 1
    cost = _parse_restaurant_cost(payload.price_for_two)
    order_index = max([i.order_index for i in trip.itinerary if i.day_number == payload.day_number] or [0]) + 1
    meta = {
        "restaurant": {
            "name": payload.name,
            "location": payload.location,
            "description": payload.description,
            "image_url": payload.image_url,
            "rating": payload.rating,
            "price_for_two": payload.price_for_two,
            "cuisine": payload.cuisine,
            "meal_type": payload.meal_type,
            "latitude": payload.latitude,
            "longitude": payload.longitude,
            "source": payload.source or "serpapi",
        },
        "ui": {
            "image_url": payload.image_url,
        } if payload.image_url else {}
    }
    item = ItineraryItem(
        trip_id=trip.id,
        day_number=payload.day_number,
        order_index=order_index,
        item_type="meal",
        title=payload.name.strip(),
        description=payload.description,
        location=payload.location,
        cost=cost,
        status="confirmed",
        meta_data=meta,
    )
    db.add(item)
    db.flush()
    _record_change(db, trip, "restaurant_added", "itinerary", payload.name.strip(), f"Restaurant added to day {payload.day_number} as meal.", "user")
    db.commit()
    db.refresh(trip)
    return _trip_dict(trip, db)


@router.get("/trips/{trip_id}/restaurants", response_model=List[RestaurantItemRead])
def list_restaurants(trip_id: str, request: Request, db: Session = Depends(get_db)):
    trip = _trip_or_404(db, trip_id)
    _verify_trip_ownership_optional(request, db, trip)
    items = db.query(ItineraryItem).filter(ItineraryItem.trip_id == trip.id, ItineraryItem.item_type == "meal").order_by(ItineraryItem.day_number, ItineraryItem.order_index).all()
    return items


@router.delete("/trips/{trip_id}/restaurants/{item_id}")
def delete_restaurant(trip_id: str, item_id: str, request: Request, db: Session = Depends(get_db)):
    trip = _trip_or_404(db, trip_id)
    _verify_trip_ownership_optional(request, db, trip)
    item = db.query(ItineraryItem).filter(ItineraryItem.id == item_id, ItineraryItem.trip_id == trip.id, ItineraryItem.item_type == "meal").first()
    if not item:
        raise HTTPException(status_code=404, detail="Restaurant itinerary item not found")
    title = item.title
    db.delete(item)
    _record_change(db, trip, "restaurant_removed", "itinerary", title, "Restaurant removed from itinerary.", "user")
    db.commit()
    db.refresh(trip)
    return _trip_dict(trip, db)


@router.post("/trips/{trip_id}/toggle-activity")
def toggle_activity(trip_id: str, payload: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    trip = _trip_or_404(db, trip_id)
    item = db.query(ItineraryItem).filter(ItineraryItem.id == payload.get("item_id"), ItineraryItem.trip_id == trip.id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Itinerary item not found")
    item.status = "confirmed" if item.status == "skipped" else "skipped"
    return _commit_trip(db, trip, "activity_toggled", "itinerary", item.title, f"Activity marked {item.status}.")


@router.post("/trips/{trip_id}/add-day-leg")
def add_day_leg(trip_id: str, db: Session = Depends(get_db)):
    trip = _trip_or_404(db, trip_id)
    trip.duration_days += 1
    day = trip.duration_days
    db.add(ItineraryItem(trip_id=trip.id, day_number=day, order_index=1, item_type="leisure",
           title="Free time for local exploration", description="Flexible time reserved for traveler-selected activities.",
           cost=0, status="proposed", location=trip.destination.name if trip.destination else None))
    return _commit_trip(db, trip, "day_leg_added", "duration_days", str(day), "Trip duration extended by one day.")


@router.post("/trips/{trip_id}/remove-day-leg")
def remove_day_leg(trip_id: str, payload: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    trip = _trip_or_404(db, trip_id)
    if trip.duration_days <= 2:
        raise HTTPException(status_code=400, detail="Trip cannot have less than 2 days")
    day = int(payload.get("day_number") or trip.duration_days)
    db.query(ItineraryItem).filter(ItineraryItem.trip_id == trip.id, ItineraryItem.day_number == day).delete()
    for item in db.query(ItineraryItem).filter(ItineraryItem.trip_id == trip.id, ItineraryItem.day_number > day):
        item.day_number -= 1
    trip.duration_days -= 1
    return _commit_trip(db, trip, "day_leg_removed", "duration_days", str(trip.duration_days), "Trip duration reduced by one day.")


@router.get("/possible-options")
def possible_options(destination: str, trip_id: Optional[str] = None, db: Session = Depends(get_db)):
    dest = db.query(Destination).filter((Destination.name.ilike(destination)) | (Destination.slug.ilike(destination))).first()
    if not dest:
        return []
    return [{"id": a.id, "title": a.title, "category": a.category, "location": a.meeting_point or dest.name,
             "duration": f"{a.duration_hours:g} hours", "cost": a.price_per_person,
             "description": a.description, "image_url": (a.images or [None])[0],
             "tags": [a.category], "walking_intensity": "moderate"} for a in
            db.query(Activity).filter(Activity.destination_id == dest.id, Activity.is_active == True).all()]


@router.post("/trips/{trip_id}/lock-booking")
def lock_booking(trip_id: str, payload: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    trip = _trip_or_404(db, trip_id)
    details = payload.get("details") or {}
    item_id, item_type = payload.get("item_id"), payload.get("item_type") or "service"
    vendor_id = None
    if item_type == "hotel" and item_id:
        record = db.query(Hotel).filter(Hotel.id == item_id).first(); vendor_id = record.vendor_id if record else None
    elif item_type == "activity" and item_id:
        record = db.query(Activity).filter(Activity.id == item_id).first(); vendor_id = record.vendor_id if record else None
    elif item_type == "transport" and item_id:
        record = db.query(TransportOption).filter(TransportOption.id == item_id).first(); vendor_id = record.vendor_id if record else None
    mode = payload.get("booking_mode")
    if mode not in {"ai_guide", "self_booking"}:
        raise HTTPException(status_code=422, detail="booking_mode must be ai_guide or self_booking")
    booking = Booking(trip_id=trip.id, vendor_id=vendor_id, item_type=item_type, item_id=item_id,
        amount=float(details.get("amount") or 0), currency=trip.currency,
        status="confirmed" if mode == "ai_guide" else "pending", payment_status="paid" if mode == "ai_guide" else "pending")
    db.add(booking); db.flush()
    _record_change(db, trip, "booking_locked", "booking", booking.booking_reference, "Booking choice saved.", "ai" if mode == "ai_guide" else "user")
    db.commit(); db.refresh(trip)
    return {"success": True, "booking": _booking_dict(booking), "trip": _trip_dict(trip, db)}


# ----------------------------------------------------
# Traveler password authentication + owned trip snapshots
# (Operator login above is separate and untouched.)
# ----------------------------------------------------
def _traveler_auth_error(exc: Exception) -> HTTPException:
    from backend.auth.service import AuthConflict, AuthError, AuthValidation, TripForbidden, TripNotFound
    if isinstance(exc, AuthError):
        return HTTPException(status_code=401, detail=str(exc))
    if isinstance(exc, AuthConflict):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, AuthValidation):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, TripForbidden):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, TripNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    raise exc


@router.post("/auth/traveler/signup", response_model=TravelerAuthResponse, status_code=201)
def traveler_signup(payload: TravelerSignupRequest, db: Session = Depends(get_db)):
    """Create a traveler account (bcrypt-hashed password) and start a session."""
    from backend.auth.service import signup_traveler
    try:
        return signup_traveler(db, payload.full_name, payload.email, payload.password)
    except Exception as exc:
        raise _traveler_auth_error(exc)


@router.post("/auth/traveler/login", response_model=TravelerAuthResponse)
def traveler_login(payload: TravelerLoginRequest, db: Session = Depends(get_db)):
    """Traveler email + password login. 401 on invalid credentials."""
    from backend.auth.service import login_traveler
    try:
        return login_traveler(db, payload.email, payload.password)
    except Exception as exc:
        raise _traveler_auth_error(exc)


@router.get("/auth/traveler/me", response_model=TravelerRead)
def traveler_me(request: Request, db: Session = Depends(get_db)):
    """Validate the Bearer session and return the traveler (401 when expired/invalid)."""
    from backend.auth.service import get_current_traveler, traveler_dict
    try:
        return traveler_dict(get_current_traveler(request, db))
    except HTTPException:
        raise
    except Exception as exc:
        raise _traveler_auth_error(exc)


@router.get("/traveler/trips", response_model=List[TravelerTripSummary])
def traveler_list_trips(request: Request, db: Session = Depends(get_db)):
    """Trip summaries owned by the authenticated traveler only."""
    from backend.auth.service import get_current_traveler, list_traveler_trips
    try:
        user = get_current_traveler(request, db)
        return list_traveler_trips(db, user)
    except HTTPException:
        raise
    except Exception as exc:
        raise _traveler_auth_error(exc)


@router.get("/traveler/trips/{trip_id}")
def traveler_get_trip(trip_id: str, request: Request, db: Session = Depends(get_db)):
    """Full canonical snapshot for one owned trip. 404 unless owned."""
    from backend.auth.service import get_current_traveler, get_traveler_trip
    try:
        user = get_current_traveler(request, db)
        return get_traveler_trip(db, user, trip_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise _traveler_auth_error(exc)


@router.post("/traveler/trips", response_model=TravelerTripSaveResponse, status_code=201)
def traveler_save_trip(payload: TravelerTripSaveRequest, request: Request, db: Session = Depends(get_db)):
    """Create-or-update the traveler's own canonical snapshot (no duplicates;
    403 when the trip id belongs to another traveler)."""
    from backend.auth.service import get_current_traveler, save_traveler_trip
    try:
        user = get_current_traveler(request, db)
        return save_traveler_trip(db, user, payload.trip_id, payload.trip)
    except HTTPException:
        raise
    except Exception as exc:
        raise _traveler_auth_error(exc)


# ----------------------------------------------------
# Traveler profile & preferences (mobile app)
# ----------------------------------------------------
def _get_or_create_traveler_profile(db: Session, user: User):
    from backend.models.models import TravelerProfile
    profile = db.query(TravelerProfile).filter(TravelerProfile.user_id == user.id).first()
    if not profile:
        profile = TravelerProfile(user_id=user.id)
        db.add(profile)
        db.commit()
        db.refresh(profile)
    return profile


def _profile_response(user: User, profile) -> dict:
    return {
        "user_id": user.id,
        "full_name": user.full_name,
        "email": user.email,
        "phone": user.phone,
        "travel_style": profile.travel_style or "balanced",
        "dietary_preferences": profile.dietary_preferences or [],
        "fitness_level": profile.fitness_level or "moderate",
        "preferred_currency": profile.preferred_currency or "INR",
        "language": profile.language or "English",
        "bio": profile.bio,
        "created_at": profile.created_at or user.created_at,
        "updated_at": profile.updated_at or user.updated_at,
    }


@router.get("/traveler/profile", response_model=TravelerProfileResponse)
def get_traveler_profile(request: Request, db: Session = Depends(get_db)):
    """Return the authenticated traveler's full profile (auto-creates if missing)."""
    from backend.auth.service import get_current_traveler
    user = get_current_traveler(request, db)
    profile = _get_or_create_traveler_profile(db, user)
    return _profile_response(user, profile)


@router.put("/traveler/profile", response_model=TravelerProfileResponse)
def update_traveler_profile(payload: TravelerProfileUpdate, request: Request, db: Session = Depends(get_db)):
    """Full update of traveler profile (all fields optional, validates preferences)."""
    from backend.auth.service import get_current_traveler
    user = get_current_traveler(request, db)
    profile = _get_or_create_traveler_profile(db, user)
    data = payload.model_dump(exclude_unset=True)
    if "full_name" in data and data["full_name"] is not None:
        user.full_name = data["full_name"].strip()
    if "phone" in data:
        user.phone = data["phone"]
    if "bio" in data:
        profile.bio = data["bio"]
    for field in ("travel_style", "dietary_preferences", "fitness_level", "preferred_currency", "language"):
        if field in data and data[field] is not None:
            setattr(profile, field, data[field])
    db.commit()
    db.refresh(user)
    db.refresh(profile)
    return _profile_response(user, profile)


@router.patch("/traveler/profile", response_model=TravelerProfileResponse)
def patch_traveler_profile(payload: TravelerProfileUpdate, request: Request, db: Session = Depends(get_db)):
    """Partial update of traveler profile (same validation as PUT)."""
    from backend.auth.service import get_current_traveler
    user = get_current_traveler(request, db)
    profile = _get_or_create_traveler_profile(db, user)
    data = payload.model_dump(exclude_unset=True)
    if "full_name" in data and data["full_name"] is not None:
        user.full_name = data["full_name"].strip()
    if "phone" in data:
        user.phone = data["phone"]
    if "bio" in data:
        profile.bio = data["bio"]
    for field in ("travel_style", "dietary_preferences", "fitness_level", "preferred_currency", "language"):
        if field in data and data[field] is not None:
            setattr(profile, field, data[field])
    db.commit()
    db.refresh(user)
    db.refresh(profile)
    return _profile_response(user, profile)


@router.get("/traveler/preferences", response_model=TravelerPreferencesRead)
def get_traveler_preferences(request: Request, db: Session = Depends(get_db)):
    """Return only traveler preference fields."""
    from backend.auth.service import get_current_traveler
    user = get_current_traveler(request, db)
    profile = _get_or_create_traveler_profile(db, user)
    return {
        "travel_style": profile.travel_style or "balanced",
        "dietary_preferences": profile.dietary_preferences or [],
        "fitness_level": profile.fitness_level or "moderate",
        "preferred_currency": profile.preferred_currency or "INR",
        "language": profile.language or "English",
    }


@router.put("/traveler/preferences", response_model=TravelerPreferencesRead)
def update_traveler_preferences(payload: TravelerPreferencesUpdate, request: Request, db: Session = Depends(get_db)):
    """Update traveler preferences only."""
    from backend.auth.service import get_current_traveler
    user = get_current_traveler(request, db)
    profile = _get_or_create_traveler_profile(db, user)
    data = payload.model_dump(exclude_unset=True)
    for field in ("travel_style", "dietary_preferences", "fitness_level", "preferred_currency", "language"):
        if field in data and data[field] is not None:
            setattr(profile, field, data[field])
    db.commit()
    db.refresh(profile)
    return {
        "travel_style": profile.travel_style or "balanced",
        "dietary_preferences": profile.dietary_preferences or [],
        "fitness_level": profile.fitness_level or "moderate",
        "preferred_currency": profile.preferred_currency or "INR",
        "language": profile.language or "English",
    }


# ----------------------------------------------------
# Traveler notifications (authenticated)
# ----------------------------------------------------
@router.get("/traveler/notifications", response_model=NotificationListResponse)
def list_traveler_notifications(
    request: Request,
    db: Session = Depends(get_db),
    unread_only: bool = Query(default=False),
    trip_id: Optional[str] = Query(default=None, max_length=36),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List notifications belonging to the authenticated traveler, newest first."""
    from backend.auth.service import get_current_traveler
    user = get_current_traveler(request, db)
    q = db.query(Notification).filter(Notification.user_id == user.id)
    if unread_only:
        q = q.filter(Notification.is_read == False)  # noqa: E712
    if trip_id:
        q = q.filter(Notification.trip_id == trip_id)
    total = q.count()
    unread_count = db.query(Notification).filter(Notification.user_id == user.id, Notification.is_read == False).count()  # noqa: E712
    items = q.order_by(Notification.created_at.desc()).offset(offset).limit(limit).all()
    return {"notifications": items, "total": total, "unread_count": unread_count, "limit": limit, "offset": offset}


@router.get("/traveler/notifications/unread-count", response_model=UnreadCountResponse)
def traveler_unread_count(request: Request, db: Session = Depends(get_db)):
    from backend.auth.service import get_current_traveler
    user = get_current_traveler(request, db)
    count = db.query(Notification).filter(Notification.user_id == user.id, Notification.is_read == False).count()  # noqa: E712
    return {"count": count}


@router.patch("/traveler/notifications/{notification_id}/read", response_model=TravelerNotificationRead)
def mark_notification_read(notification_id: str, request: Request, db: Session = Depends(get_db)):
    from backend.auth.service import get_current_traveler
    user = get_current_traveler(request, db)
    notif = db.query(Notification).filter(Notification.id == notification_id, Notification.user_id == user.id).first()
    if not notif:
        raise HTTPException(status_code=404, detail="Notification not found")
    notif.is_read = True
    db.commit()
    db.refresh(notif)
    return notif


@router.post("/traveler/notifications/mark-all-read", response_model=MarkAllReadResponse)
def mark_all_notifications_read(request: Request, db: Session = Depends(get_db)):
    from backend.auth.service import get_current_traveler
    user = get_current_traveler(request, db)
    updated = db.query(Notification).filter(Notification.user_id == user.id, Notification.is_read == False).update({"is_read": True})  # noqa: E712
    db.commit()
    return {"updated": updated}


@router.delete("/traveler/notifications/{notification_id}")
def delete_notification(notification_id: str, request: Request, db: Session = Depends(get_db)):
    from backend.auth.service import get_current_traveler
    user = get_current_traveler(request, db)
    notif = db.query(Notification).filter(Notification.id == notification_id, Notification.user_id == user.id).first()
    if not notif:
        raise HTTPException(status_code=404, detail="Notification not found")
    db.delete(notif)
    db.commit()
    return {"success": True, "id": notification_id}


# ----------------------------------------------------
# Traveler bookings (authenticated)
# ----------------------------------------------------
@router.get("/traveler/bookings", response_model=TravelerBookingListResponse)
def list_traveler_bookings(
    request: Request,
    db: Session = Depends(get_db),
    trip_id: Optional[str] = Query(default=None, max_length=36),
    status: Optional[str] = Query(default=None, max_length=50),
    item_type: Optional[str] = Query(default=None, max_length=50),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    from backend.auth.service import get_current_traveler
    from backend.booking.traveler_service import list_traveler_bookings as svc_list
    user = get_current_traveler(request, db)
    bookings, total = svc_list(db, user.id, trip_id=trip_id, status=status, item_type=item_type, limit=limit, offset=offset)
    return {"bookings": bookings, "total": total, "limit": limit, "offset": offset}


@router.get("/traveler/bookings/{booking_id}/status", response_model=BookingStatusResponse)
def get_traveler_booking_status(booking_id: str, request: Request, db: Session = Depends(get_db)):
    from backend.auth.service import get_current_traveler
    from backend.booking.traveler_service import get_booking_status as svc_status
    user = get_current_traveler(request, db)
    return svc_status(db, user.id, booking_id)


@router.get("/traveler/bookings/{booking_id}", response_model=TravelerBookingRead)
def get_traveler_booking(booking_id: str, request: Request, db: Session = Depends(get_db)):
    from backend.auth.service import get_current_traveler
    from backend.booking.traveler_service import get_traveler_booking as svc_get
    user = get_current_traveler(request, db)
    return svc_get(db, user.id, booking_id)


@router.post("/traveler/bookings/{booking_id}/cancel", response_model=TravelerBookingRead)
def cancel_traveler_booking(
    booking_id: str, request: Request, db: Session = Depends(get_db), payload: Optional[BookingCancelRequest] = None
):
    from backend.auth.service import get_current_traveler
    from backend.booking.traveler_service import cancel_traveler_booking as svc_cancel
    user = get_current_traveler(request, db)
    return svc_cancel(db, user.id, booking_id)


@router.get("/traveler/trips/{trip_id}/bookings", response_model=List[TravelerBookingRead])
def list_trip_traveler_bookings(trip_id: str, request: Request, db: Session = Depends(get_db)):
    from backend.auth.service import get_current_traveler
    from backend.booking.traveler_service import list_trip_bookings as svc_trip
    user = get_current_traveler(request, db)
    return svc_trip(db, user.id, trip_id)

# ----------------------------------------------------
# TourFlow AI Guide - persistent, context-aware chat.
# Canonical: POST /api/guide/chat {message, tripId?}
# Alias: POST /api/chat (same payload, frontend compat).
# History: GET /api/guide/history?tripId=...
# Greeting: GET /api/guide/greeting?tripId=...
# Auth: traveler Bearer session; user_id derived server-side only.
# Isolation: every query scoped by (user_id, trip_id) + ownership check.
# ----------------------------------------------------
def _guide_current_user(request: Request, db: Session):
    from backend.auth.service import get_current_traveler
    return get_current_traveler(request, db)


def _guide_resolve_trip(db: Session, user, trip_id: Optional[str]):
    """Returns (trip, error_response). Explicit-but-unknown IDs yield a 404
    that ALSO carries the traveler's own active trip (no leak: own data
    only) so the UI can self-heal instead of dead-ending."""
    from fastapi.responses import JSONResponse
    from backend.guide.context_service import GuideContextService
    svc = GuideContextService(db)
    if trip_id:
        cleaned = str(trip_id).strip()
        trip = svc.get_owned_trip(user.id, cleaned) if cleaned else None
        if trip is not None:
            return trip, None
        active = svc.get_active_trip(user.id)
        active_info = None
        if active is not None:
            dest = active.destination
            active_info = {"trip_id": active.id, "title": active.title,
                           "destination": dest.name if dest else None,
                           "status": active.status}
        return None, JSONResponse(
            status_code=404,
            content={"detail": "Trip not found",
                     "active_trip": active_info,
                     "has_active_trip": active is not None})
    return svc.get_active_trip(user.id), None


def _guide_trip_card(ctx) -> Optional[Dict[str, Any]]:
    trip = (ctx or {}).get("trip")
    if not trip:
        return None
    budget = (ctx or {}).get("budget") or {}
    return {"trip_id": trip.get("id"), "title": trip.get("title"),
            "destination": trip.get("destination"), "status": trip.get("status"),
            "duration_days": trip.get("duration_days"),
            "current_trip_day": trip.get("current_trip_day"),
            "traveler_count": trip.get("traveler_count"),
            "total_budget": budget.get("total_budget"),
            "spent": budget.get("current_spend"),
            "remaining": budget.get("remaining_budget"),
            "currency": budget.get("currency"),
            "bookings_count": len((ctx or {}).get("bookings") or [])}


def _guide_chat_impl(payload_message: str, payload_trip_id: Optional[str], request: Request, db: Session):
    from backend.guide.chat_service import GuideChatService
    user = _guide_current_user(request, db)
    trip, guide_error = _guide_resolve_trip(db, user, payload_trip_id)
    if guide_error is not None:
        return guide_error
    service = GuideChatService(db, gemini_service)
    if trip is None:
        greeting = "I do not see an active trip yet. Create or select a trip and I will help you plan it."
        return {"response": greeting, "trip_id": None, "greeting": greeting, "action": None,
                "suggestions": ["Create a trip", "Browse destinations"], "trip_card": None}
    service.save_message(user.id, trip.id, "user", payload_message)
    ctx = service.context.buildContext(user.id, trip.id)
    try:
        reply = service.generate(payload_message, ctx)
    except Exception:
        reply = service.answer_from_context(payload_message, ctx)
    action_result = None
    try:
        parsed = GuideChatService.parse_action(payload_message)
        if parsed is None and chr(34) + "intent" + chr(34) in (reply or ""):
            parsed = GuideChatService.parse_action(reply)
        if parsed is not None:
            outcome = service.execute_action(user.id, trip, parsed)
            action_result = {"applied": bool(outcome.get("applied")),
                             "intent": parsed.get("intent"),
                             "action": parsed.get("action") or outcome.get("action"),
                             "item_id": outcome.get("item_id"),
                             "reason": outcome.get("reason")}
            if outcome.get("applied"):
                ctx = service.context.buildContext(user.id, trip.id)
                reply = (service.describe_action(outcome, parsed)
                         or service.answer_from_context(payload_message, ctx))
            elif parsed is not None:
                reason = outcome.get("reason") or "that did not match anything"
                reply = (f"I could not apply that change ({reason}). "
                         f"Tell me the exact stop name as it appears in your itinerary?")
    except Exception:
        action_result = None
    service.save_message(user.id, trip.id, "assistant", reply)
    try:
        service.context.maybe_update_summary(user.id, trip.id)
    except Exception:
        pass
    greeting = service.context.build_greeting(ctx)
    suggestions = ["What should I do tomorrow?", "Show my bookings", "How much have I spent?"]
    return {"response": reply, "trip_id": trip.id, "greeting": greeting,
            "action": action_result, "suggestions": suggestions,
            "trip_card": _guide_trip_card(ctx)}


@router.post("/guide/chat")
def guide_chat(payload: Dict[str, Any] = Body(default={}), request: Request = None, db: Session = Depends(get_db)):
    from backend.schemas.schemas import GuideChatRequest
    try:
        parsed = GuideChatRequest(message=payload.get("message", ""), tripId=payload.get("tripId") or payload.get("trip_id"))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _guide_chat_impl(parsed.message, parsed.tripId, request, db)


@router.post("/chat")
def guide_chat_alias(payload: Dict[str, Any] = Body(default={}), request: Request = None, db: Session = Depends(get_db)):
    from backend.schemas.schemas import GuideChatRequest
    try:
        parsed = GuideChatRequest(message=payload.get("message", ""), tripId=payload.get("tripId") or payload.get("trip_id"))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _guide_chat_impl(parsed.message, parsed.tripId, request, db)


@router.get("/guide/history")
def guide_history(tripId: Optional[str] = None, trip_id: Optional[str] = None, request: Request = None, db: Session = Depends(get_db)):
    from backend.guide.chat_service import GuideChatService
    user = _guide_current_user(request, db)
    trip, guide_error = _guide_resolve_trip(db, user, tripId or trip_id)
    if guide_error is not None:
        return guide_error
    service = GuideChatService(db, gemini_service)
    if trip is None:
        return {"trip_id": None, "greeting": "I do not see an active trip yet. Create or select a trip and I will help you plan it.",
                "messages": [], "has_active_trip": False, "trip_card": None}
    ctx = service.context.buildContext(user.id, trip.id)
    messages = service.load_history(user.id, trip.id, limit=50)
    return {"trip_id": trip.id, "greeting": service.context.build_greeting(ctx),
            "messages": messages, "has_active_trip": True,
            "trip_card": _guide_trip_card(ctx)}


@router.get("/guide/greeting")
def guide_greeting(tripId: Optional[str] = None, trip_id: Optional[str] = None, request: Request = None, db: Session = Depends(get_db)):
    from backend.guide.chat_service import GuideChatService
    user = _guide_current_user(request, db)
    trip, guide_error = _guide_resolve_trip(db, user, tripId or trip_id)
    if guide_error is not None:
        return guide_error
    service = GuideChatService(db, gemini_service)
    if trip is None:
        return {"trip_id": None, "greeting": "I do not see an active trip yet. Create or select a trip and I will help you plan it.",
                "has_active_trip": False, "user_name": user.full_name}
    ctx = service.context.buildContext(user.id, trip.id)
    first = (ctx.get("user") or {}).get("first_name")
    return {"trip_id": trip.id, "greeting": service.context.build_greeting(ctx),
            "has_active_trip": True, "user_name": first or user.full_name}