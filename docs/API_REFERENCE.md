# TourFlow AI - API Reference

## Base URL
`/api`

## Core Endpoints

### System & Health
- `GET /api/health`
  - Returns backend health, database connection state, destination/trip record counts, and Gemini AI readiness.

### Destinations
- `GET /api/destinations`
  - Optional query: `featured_only=true`
  - Returns array of destinations (Manali, Goa, Kerala, Rajasthan, Kashmir).
- `GET /api/destinations/search?q=&featured_only=&category=&tag=&country=&state_region=&limit=20&offset=0`
  - Searches catalog destinations across `name`, `state_region`, `country`, `tags`, `description` (case-insensitive, never fabricated). Optional filters `featured_only`, `category`/`tag` (matches tags), `country`, `state_region`. Returns `{ results: Destination[], total, limit, offset }` with pagination (`limit` 1-100, `offset` >=0). Unknown queries return empty `results`, never creates rows.
- `GET /api/destinations/featured`
  - Returns featured destinations (`is_featured=true`) ordered by name.
- `GET /api/destinations/categories`
  - Returns `{ categories: string[], total }` derived from distinct tags in DB.
- `GET /api/destinations/nearby?latitude=&longitude=&radius_km=500&limit=20`
  - Returns destinations within `radius_km` using actual coordinates, sorted by distance, with `distance_km` field. Validates `latitude` -90..90, `longitude` -180..180 (`422` on invalid). Uses haversine, no external maps.
- `GET /api/destinations/{id}`
  - Returns full destination entity by UUID or slug.

### Catalog Services
- `GET /api/hotels`
  - Optional query: `destination_id`, `category`
  - Returns verified hotels with nightly rates, ratings, and amenities.
- `GET /api/activities`
  - Optional query: `destination_id`, `category`
  - Returns curated experiences, durations, difficulties, and pricing.
- `GET /api/transport`
  - Optional query: `destination_id`, `type`
  - Returns private SUVs, Volvo buses, and self-drive rentals.

### Trips (Central Entity)
- `POST /api/trips`
  - Body: `{ title, destination_id, duration_days, total_budget, pace, preferences: {...} }`
  - Initializes Trip, attaches preferences, creates welcome notification, records change history, and generates itinerary.
- `GET /api/trips/{trip_id}`
  - Retrieves trip with nested traveler, preferences, itinerary items, bookings, alerts, notifications, and change history.
- `PUT /api/trips/{trip_id}`
  - Updates trip metadata and logs mutation in change_history.
- `GET /api/trips/{trip_id}/preferences`
  - Retrieves preference configuration for the trip.
- `PUT /api/trips/{trip_id}/preferences`
  - Updates preference constraints and logs mutation.

### Traveler Profile & Preferences (Authenticated - Bearer token)
- `GET /api/traveler/profile` - Requires `Authorization: Bearer <token>`. Returns `{ user_id, full_name, email, phone, travel_style, dietary_preferences, fitness_level, preferred_currency, language, bio, created_at, updated_at }`. Auto-creates a `TravelerProfile` with defaults if the traveler has none.
- `PUT /api/traveler/profile` - Update full profile fields `{ full_name, phone, travel_style, dietary_preferences, fitness_level, preferred_currency, language, bio }`. Validates travel_style, currency, language, dietary array. Returns updated profile.
- `PATCH /api/traveler/profile` - Partial update (same fields/validation as PUT). Only supplied fields are updated.
- `GET /api/traveler/preferences` - Returns only `{ travel_style, dietary_preferences, fitness_level, preferred_currency, language }`.
- `PUT /api/traveler/preferences` - Update preference fields `{ travel_style, dietary_preferences, fitness_level, preferred_currency, language }` with same validation. Returns updated preferences.

Validation rules:
- `travel_style`: one of `luxury, budget, adventure, cultural, relaxed, balanced`
- `fitness_level`: one of `low, moderate, high`
- `preferred_currency`: 3-letter uppercase code matching `^[A-Z]{3}$` (e.g. INR, USD)
- `language`: 2-50 chars
- `dietary_preferences`: array of non-empty strings (each <=50 chars)
- Invalid values return `422`, missing/invalid token returns `401`.

### Traveler Notifications (Authenticated - Bearer token)
- `GET /api/traveler/notifications?unread_only=false&trip_id=&limit=50&offset=0` - List own notifications newest-first. Returns `{ notifications: [{id,trip_id,title,message,type,is_read,created_at}], total, unread_count, limit, offset }`. Supports `unread_only`, `trip_id` filtering, pagination (`limit` 1-100, `offset` >=0).
- `GET /api/traveler/notifications/unread-count` - Returns `{ count: number }` of unread notifications.
- `PATCH /api/traveler/notifications/{notification_id}/read` - Marks one own notification as read. Returns the notification. 404 if not owned.
- `POST /api/traveler/notifications/mark-all-read` - Marks all own unread notifications as read. Returns `{ updated: number }`.
- `DELETE /api/traveler/notifications/{notification_id}` - Deletes/dismisses own notification. Returns `{ success: true, id }`. 404 if not owned.

All require `Authorization: Bearer <token>` via `get_current_traveler()`; 401 if missing/invalid, 404 if accessing another traveler's notification.

### Traveler Bookings (Authenticated - Bearer token)
- `GET /api/traveler/bookings?trip_id=&status=&item_type=&limit=50&offset=0` - List bookings for trips owned by traveler. Filters: `trip_id`, `status`, `item_type` (hotel/activity/transport). Returns `{ bookings: [{id,booking_reference,trip_id,trip_title,destination,item_type,item_id,vendor,amount,currency,status,payment_status,booking_date}], total, limit, offset }`.
- `GET /api/traveler/bookings/{booking_id}` - Get single owned booking with vendor info. 404 if not owned.
- `GET /api/traveler/bookings/{booking_id}/status` - Compact status `{ booking_id, booking_reference, status, payment_status, item_type, amount, currency, booking_date }`.
- `POST /api/traveler/bookings/{booking_id}/cancel` - Cancel owned booking. Sets `status=cancelled`; if `payment_status=paid` → `refund_pending`, otherwise unchanged. Rejects already `cancelled`/`refunded` with 409. Records change history.
- `GET /api/traveler/trips/{trip_id}/bookings` - List bookings for one owned trip. 404 if trip not owned.

All require `Authorization: Bearer <token>`; 401 if missing/invalid. No payment gateway; refund is `refund_pending` not `refunded`.

### Restaurants & Meals
- `GET /api/restaurants/search?destination=&meal_type=&cuisine=&...` - Provider-backed live restaurant search (SerpApi). Returns `{ destination, meal_type, cuisine, results: [{id, place_id, name, address, rating, latitude, longitude, types, ...}], source }`.
- `POST /api/trips/{trip_id}/add-restaurant` - Add a restaurant (from search or custom) to a specific day as `item_type="meal"`. Body: `{ day_number: int (1..duration_days), name: string (required), location, description, image_url, rating (0-5), price_for_two (numeric or string, parsed; else 0), cuisine, meal_type, latitude, longitude, source }`. Validates trip exists (`404`), day range (`422`), `name` required (`422`). Assigns next `order_index` for that day, preserves restaurant metadata in `meta_data.restaurant`, `cost` is parsed numeric price or `0` (no invented price). Records `ChangeHistory` (`restaurant_added`), returns updated trip (`_trip_dict`). If `Authorization: Bearer` present, enforces trip ownership (`403` if other traveler).
- `GET /api/trips/{trip_id}/restaurants` - List meal/restaurant items for trip ordered by `day_number, order_index`. Returns `RestaurantItemRead[]`.
- `DELETE /api/trips/{trip_id}/restaurants/{item_id}` - Remove a meal item (`404` if not found or not meal). Records `restaurant_removed`, returns updated trip. Ownership checked as above. Existing `POST /api/trips/{trip_id}/delete-activity` also supports `item_type="meal"` without breaking behavior.

Adding a restaurant is **not** a booking/reservation; no payment or vendor booking is created.

### Weather & Live Conditions
- `GET /api/weather?destination=&latitude=&longitude=&date=&days=5` — Live weather (current + forecast) via server-side provider (Open-Meteo, `WEATHER_BASE_URL`). Resolves `lat/lng` from explicit params → `Destination` lat/lng → Nominatim geocode. Returns `{ destination, latitude, longitude, current: {temperature, feels_like, condition, humidity, wind_speed}, forecast: [{date, temperature_min, temperature_max, condition, precipitation_probability}], source, retrieved_at }`. Errors: `422` invalid destination/date/days, `503` provider not configured (`WEATHER_BASE_URL` empty), `502` provider failure. Never fabricated; short TTL cache (600s) to avoid repeated AI hits. Keys kept in env, never exposed.

### AI Planning Services (weather-aware)
- `POST /api/ai/chat`
  - Body: `{ message, session_context, destination_id }`
  - Conversational concierge interface powered by Gemini.
- `POST /api/ai/extract-preferences`
  - Body: `{ text_prompt }`
  - Structured travel parameter extractor.
- `POST /api/ai/recommend`
  - Body: `{ destination_id, preferences }`
  - Ranked recommendations matching traveler criteria.
- `POST /api/ai/generate-itinerary`
  - Body: `{ trip_id, duration_days, preferences }`
  - Dynamic itinerary synthesis.
- `POST /api/ai/replan`
  - Body: `{ trip_id, trigger_event: { type, severity, description } }`
  - Dynamic replanning engine execution with alert creation and history audit.
