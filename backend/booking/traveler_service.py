"""Traveler-owned booking lifecycle management (read, cancel, status).

Uses existing Booking table and Trip ownership. No payment gateway.
"""
from typing import Optional, List
from sqlalchemy.orm import Session
from fastapi import HTTPException

from backend.models.models import Booking, Trip, Vendor


class BookingAccessError(HTTPException):
    pass


def _get_owned_booking(db: Session, user_id: str, booking_id: str) -> Booking:
    booking = db.query(Booking).filter(Booking.id == booking_id).first()
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    trip = db.query(Trip).filter(Trip.id == booking.trip_id, Trip.user_id == user_id).first()
    if not trip:
        raise HTTPException(status_code=404, detail="Booking not found")
    return booking


def _get_owned_trip(db: Session, user_id: str, trip_id: str) -> Trip:
    trip = db.query(Trip).filter(Trip.id == trip_id, Trip.user_id == user_id).first()
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")
    return trip


def _booking_to_read(db: Session, booking: Booking) -> dict:
    trip = db.query(Trip).filter(Trip.id == booking.trip_id).first()
    vendor = None
    if booking.vendor_id:
        v = db.query(Vendor).filter(Vendor.id == booking.vendor_id).first()
        if v:
            vendor = {"id": v.id, "name": v.name, "vendor_type": v.vendor_type,
                      "contact_email": v.contact_email, "phone": v.phone,
                      "rating": v.rating, "is_verified": v.is_verified}
    dest_name = None
    trip_title = None
    if trip:
        trip_title = trip.title
        if trip.destination:
            dest_name = trip.destination.name
    return {
        "id": booking.id,
        "booking_reference": booking.booking_reference,
        "trip_id": booking.trip_id,
        "trip_title": trip_title,
        "destination": dest_name,
        "item_type": booking.item_type,
        "item_id": booking.item_id,
        "vendor": vendor,
        "amount": booking.amount,
        "currency": booking.currency,
        "status": booking.status,
        "payment_status": booking.payment_status,
        "booking_date": booking.booking_date,
    }


def list_traveler_bookings(
    db: Session, user_id: str,
    trip_id: Optional[str] = None,
    status: Optional[str] = None,
    item_type: Optional[str] = None,
    limit: int = 50, offset: int = 0,
):
    # trips owned by user
    trip_ids_q = db.query(Trip.id).filter(Trip.user_id == user_id)
    if trip_id:
        # verify trip owned
        _get_owned_trip(db, user_id, trip_id)
        trip_ids_q = trip_ids_q.filter(Trip.id == trip_id)
    trip_ids = [r[0] for r in trip_ids_q.all()]
    if not trip_ids:
        return [], 0
    q = db.query(Booking).filter(Booking.trip_id.in_(trip_ids))
    if status:
        q = q.filter(Booking.status == status)
    if item_type:
        q = q.filter(Booking.item_type == item_type)
    total = q.count()
    items = q.order_by(Booking.booking_date.desc()).offset(offset).limit(limit).all()
    return [_booking_to_read(db, b) for b in items], total


def get_traveler_booking(db: Session, user_id: str, booking_id: str) -> dict:
    booking = _get_owned_booking(db, user_id, booking_id)
    return _booking_to_read(db, booking)


def cancel_traveler_booking(db: Session, user_id: str, booking_id: str) -> dict:
    booking = _get_owned_booking(db, user_id, booking_id)
    if booking.status in ("cancelled", "refunded"):
        raise HTTPException(status_code=409, detail="Booking is already cancelled or refunded")
    # prevent double cancel? already handled
    old_status = booking.status
    old_payment = booking.payment_status
    booking.status = "cancelled"
    # If paid, move to refund_pending rather than falsely claiming refunded
    if old_payment == "paid":
        booking.payment_status = "refund_pending"
    elif old_payment == "refunded":
        # already refunded but status check above should have caught
        booking.payment_status = "refunded"
    else:
        # pending stays pending, but cancelled
        pass
    # record change history if trip exists
    trip = db.query(Trip).filter(Trip.id == booking.trip_id).first()
    if trip:
        from backend.models.models import ChangeHistory
        db.add(ChangeHistory(
            trip_id=trip.id, changed_by="user", action="booking_cancelled",
            field_changed="booking", old_value=f"{old_status}/{old_payment}",
            new_value=f"{booking.status}/{booking.payment_status}",
            reason="Traveler cancelled booking"
        ))
    db.commit()
    db.refresh(booking)
    return _booking_to_read(db, booking)


def list_trip_bookings(db: Session, user_id: str, trip_id: str):
    _get_owned_trip(db, user_id, trip_id)
    bookings = db.query(Booking).filter(Booking.trip_id == trip_id).order_by(Booking.booking_date.desc()).all()
    return [_booking_to_read(db, b) for b in bookings]


def get_booking_status(db: Session, user_id: str, booking_id: str) -> dict:
    booking = _get_owned_booking(db, user_id, booking_id)
    return {
        "booking_id": booking.id,
        "booking_reference": booking.booking_reference,
        "status": booking.status,
        "payment_status": booking.payment_status,
        "item_type": booking.item_type,
        "amount": booking.amount,
        "currency": booking.currency,
        "booking_date": booking.booking_date,
    }
