"""GuideChatService — persistent, trip-scoped Guide chat with validated actions.

Flow per message:
1. Verify ownership (user_id, trip_id).
2. Persist the user message.
3. Load recent messages (latest 10-20) for this user + trip.
4. Build trip context via GuideContextService (today/tomorrow/bookings/budget).
5. Generate grounded response (Gemini when available, else deterministic fallback).
6. Parse optional structured action, validate IDs against the active trip, execute.
7. Persist the AI response. Never lose the user message on AI failure.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.guide.context_service import (
    GUIDE_SYSTEM_INSTRUCTION,
    GuideContextService,
)
from backend.models.models import GuideMessage, ItineraryItem, Trip

logger = logging.getLogger(__name__)

VALID_INTENTS = {"UPDATE_ITINERARY", "ADD_ACTIVITY", "REMOVE_ACTIVITY", "CHANGE_HOTEL", "CHANGE_BUDGET", "ADD_BOOKING_NOTE", "MOVE_ITEM"}
VALID_ACTIONS = {"MOVE_ITEM", "ADD_ACTIVITY", "REMOVE_ACTIVITY", "CHANGE_HOTEL", "UPDATE_TIME", "UPDATE_ITEM", "CHANGE_BUDGET", "ADD_BOOKING_NOTE", "NONE"}


class GuideChatService:
    def __init__(self, db: Session, gemini_service: Any):
        self.db = db
        self.gemini = gemini_service
        self.context = GuideContextService(db)

    # -- persistence ---------------------------------------------------
    def save_message(self, user_id: str, trip_id: str, role: str, message: str) -> GuideMessage:
        from backend.models.models import generate_uuid
        row = GuideMessage(
            id=generate_uuid(), user_id=user_id, trip_id=trip_id,
            role=role, message=message, created_at=datetime.utcnow(),
        )
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row

    def load_history(self, user_id: str, trip_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        rows = (
            self.db.query(GuideMessage)
            .filter(GuideMessage.user_id == user_id, GuideMessage.trip_id == trip_id)
            .order_by(GuideMessage.created_at.asc())
            .limit(limit)
            .all()
        )
        return [
            {"role": r.role, "message": r.message,
             "created_at": r.created_at.isoformat() if r.created_at else None}
            for r in rows
        ]

    # -- structured actions --------------------------------------------
    @staticmethod
    def parse_action(text: str) -> Optional[Dict[str, Any]]:
        """Parse a structured action from free text or an embedded JSON block.

        Never trusts model-generated IDs — caller must validate against the trip.
        """
        if not text:
            return None
        # 1. Explicit JSON block.
        try:
            candidate = text.strip()
            if candidate.startswith("{"):
                data = json.loads(candidate)
                if isinstance(data, dict) and data.get("intent") in VALID_INTENTS:
                    return data
            m = re.search(r"\{[^}]*\"intent\"\s*:\s*\"(UPDATE_ITINERARY|ADD_ACTIVITY|REMOVE_ACTIVITY|CHANGE_BUDGET|ADD_BOOKING_NOTE|MOVE_ITEM)\"[^}]*\}", text)
            if m:
                return json.loads(m.group(0))
        except Exception:
            pass
        low = text.lower()
        # 2. Natural-language "move X to <time>".
        move = re.search(r"move\s+(.+?)\s+to\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", low)
        if move:
            return {"intent": "UPDATE_ITINERARY", "action": "MOVE_ITEM",
                    "item_title": move.group(1).strip(), "new_time": move.group(2).strip()}
        # 3. Natural-language "remove/delete/cancel/drop X" (validated
        # against real trip items before anything is deleted).
        rem = re.search(r"(remove|delete|cancel|drop)\s+(?:the\s+)?(.+?)(?:\s+from\s+.+)?$", low.strip())
        if rem and len(rem.group(2).strip()) > 2 and "book" not in low:
            return {"intent": "REMOVE_ACTIVITY", "action": "REMOVE_ACTIVITY",
                    "item_title": rem.group(2).strip()}
        # 4. Natural-language "add X on day N" (vague asks are ignored so no
        # junk rows are ever created; the assistant asks for specifics).
        add = re.search(r"\badd\s+(.+?)\s+(?:on|to|for)\s+day\s*(\d+)", low)
        if add:
            title = add.group(1).strip()
            vague = {"activity", "activities", "an activity", "something", "anything",
                     "stuff", "things", "it", "that", "this", "one", "more"}
            if title and title not in vague and "something" not in title and len(title) > 2:
                return {"intent": "ADD_ACTIVITY", "action": "ADD_ACTIVITY",
                        "title": title[:120], "day_number": int(add.group(2))}
        # 5. Natural-language "change/switch (my) hotel to X".
        ch = re.search(r"(change|switch)\s+(?:my\s+|the\s+)?hotel\s+to\s+(.+)$", low.strip())
        if ch and len(ch.group(2).strip()) > 2:
            return {"intent": "CHANGE_HOTEL", "action": "CHANGE_HOTEL",
                    "hotel_name": ch.group(2).strip()}
        return None

    def execute_action(self, user_id: str, trip: Trip, action: Dict[str, Any]) -> Dict[str, Any]:
        """Validate against the authenticated user's active trip, then apply."""
        intent = (action or {}).get("intent")
        act = (action or {}).get("action", intent)
        if intent not in VALID_INTENTS:
            return {"applied": False, "reason": "unsupported intent"}

        if intent in ("UPDATE_ITINERARY", "MOVE_ITEM") or act == "MOVE_ITEM":
            item = None
            item_id = action.get("item_id")
            if item_id:
                item = (
                    self.db.query(ItineraryItem)
                    .filter(ItineraryItem.id == item_id, ItineraryItem.trip_id == trip.id)
                    .first()
                )
                if item is None:
                    return {"applied": False, "reason": "item does not belong to this trip"}
            else:
                title_hint = (action.get("item_title") or "").lower()
                if title_hint:
                    candidates = (
                        self.db.query(ItineraryItem)
                        .filter(ItineraryItem.trip_id == trip.id)
                        .all()
                    )
                    for c in candidates:
                        if title_hint[:12] in (c.title or "").lower():
                            item = c
                            break
                if item is None:
                    return {"applied": False, "reason": "could not identify itinerary item"}
            new_time = action.get("new_time")
            if new_time:
                item.start_time = str(new_time)
            self.db.commit()
            return {"applied": True, "intent": intent, "action": "MOVE_ITEM",
                    "item_id": item.id, "title": item.title}

        if intent == "ADD_ACTIVITY":
            title = (action.get("title") or "").strip()
            if not title:
                return {"applied": False, "reason": "title required"}
            day = int(action.get("day_number") or 1)
            row = ItineraryItem(
                trip_id=trip.id, day_number=day,
                order_index=len([i for i in trip.itinerary if i.day_number == day]) + 1,
                item_type="activity", title=title[:255],
                description=action.get("description"),
                start_time=action.get("start_time"), end_time=action.get("end_time"),
                cost=float(action.get("cost") or 0), location=action.get("location"),
                status="proposed",
            )
            self.db.add(row)
            self.db.commit()
            return {"applied": True, "intent": intent, "action": "ADD_ACTIVITY",
                    "item_id": row.id, "title": row.title}

        if intent == "REMOVE_ACTIVITY":
            item_id = action.get("item_id")
            q = self.db.query(ItineraryItem).filter(ItineraryItem.trip_id == trip.id)
            item = q.filter(ItineraryItem.id == item_id).first() if item_id else None
            if item is None:
                title_hint = (action.get("item_title") or "").lower()
                if title_hint:
                    for c in q.all():
                        if title_hint[:12] in (c.title or "").lower():
                            item = c
                            break
            if item is None:
                return {"applied": False, "reason": "item does not belong to this trip"}
            title = item.title
            self.db.delete(item)
            self.db.commit()
            return {"applied": True, "intent": intent, "action": "REMOVE_ACTIVITY",
                    "title": title}

        if intent == "CHANGE_HOTEL" or act == "CHANGE_HOTEL":
            from backend.models.models import Hotel as _Hotel

            hint = str((action or {}).get("hotel_name") or "").strip().lower()
            candidates = (
                self.db.query(_Hotel)
                .filter(_Hotel.destination_id == trip.destination_id,
                        _Hotel.is_active == True)  # noqa: E712
                .order_by(_Hotel.rating.desc()).all()
            )
            hotel = None
            if hint:
                for c in candidates:
                    if hint in (c.name or "").lower() or (c.name or "").lower() in hint:
                        hotel = c
                        break
            if hotel is None:
                return {"applied": False, "reason": "no matching hotel in this destination"}
            item = next((i for i in sorted(trip.itinerary, key=lambda x: (x.day_number, x.order_index))
                         if i.item_type == "hotel"), None)
            if item is None:
                item = ItineraryItem(trip_id=trip.id, day_number=1, order_index=1,
                                     item_type="hotel", title=hotel.name, status="proposed")
                self.db.add(item)
            item.hotel_id = hotel.id
            item.title = hotel.name
            item.description = hotel.description
            item.location = hotel.address
            item.cost = float(hotel.price_per_night or 0)
            item.status = "confirmed"
            self.db.commit()
            return {"applied": True, "intent": "CHANGE_HOTEL", "action": "CHANGE_HOTEL",
                    "item_id": item.id, "title": hotel.name}

        # CHANGE_BUDGET / ADD_BOOKING_NOTE: return validated request, no silent writes.
        return {"applied": False, "intent": intent, "reason": "handled as proposal; use explicit trip endpoints"}

    # -- grounded response ----------------------------------------------
    @staticmethod
    def _strip_greeting(text: str) -> str:
        return re.sub(
            r"^(hi+|hii+|hello+|hey+|yo|namaste|good\s*(morning|afternoon|evening))\b[,.!\s]*",
            "", (text or "").strip(), flags=re.IGNORECASE).strip()

    def answer_from_context(self, message: str, ctx: Dict[str, Any]) -> str:
        """Deterministic grounded conversational engine (also the Gemini-down path).

        Answers open travel questions from real trip/catalog/conversation
        context only. Never invents: every fact comes from ctx; gaps are
        stated explicitly and honestly.
        """
        trip = ctx.get("trip")
        if trip is None:
            return "I don't see an active trip yet. Create or select a trip and I'll help you plan it."
        # No blanket "unavailable" gate: bookings/budget/catalog/destination
        # answers work without itinerary items; day/meal/hotel branches each
        # report their own gaps honestly.
        it = ctx.get("itinerary") or {}
        low_full = (message or "").strip().lower()
        user = ctx.get("user") or {}
        first = user.get("first_name")
        who = f" {first}" if first else ""
        bookings = ctx.get("bookings") or []
        budget = ctx.get("budget") or {}
        cur = budget.get("currency", "INR")
        dest = trip.get("destination") or "your destination"
        title = trip.get("title") or "your trip"

        # Pure small talk first: never dump the itinerary on a "hii".
        if re.fullmatch(r"(hi+|hii+|hello+|hey+|yo|namaste|good\s*(morning|afternoon|evening)|thanks?(\s+you)?|thank\s+you|bye(bye)?|good\s*night)[!. ]*",
                        low_full):
            if low_full.startswith("thank"):
                return f"You're welcome{who}! Anything else about your {title} trip — days, bookings, budget?"
            if low_full.startswith("bye") or low_full.startswith("good night"):
                return f"Have a wonderful trip{who}! I'll be here when you need me."
            return (f"Hi{who}! I'm tracking your {title} trip in {dest}. "
                    f"What do you want to know — your days, bookings, or budget?")
        # "Hi, what about tomorrow?" -> route on the remainder.
        low = self._strip_greeting(message).lower() or low_full

        def fmt_items(items: List[Dict[str, Any]], limit: int = 4) -> str:
            bits = []
            for i in items[:limit]:
                t = f"{i['title']}"
                if i.get("start_time"):
                    t += f" at {i['start_time']}"
                    if i.get("end_time"):
                        t += f"–{i['end_time']}"
                if i.get("location"):
                    t += f" ({i['location']})"
                if i.get("cost"):
                    t += f" — {cur} {i['cost']:,.0f}"
                bits.append(t)
            return "; ".join(bits)

        def fmt_money(amount: Any) -> str:
            try:
                return f"{cur} {float(amount or 0):,.0f}"
            except (TypeError, ValueError):
                return f"{cur} 0"

        # Conversation memory references take precedence over topic branches.
        if any(k in low for k in ("earlier", "you said", "i said", "we planned", "mentioned", "told you")):
            recent = ctx.get("recentMessages") or []
            mine = [m for m in recent if m["role"] == "user"][-3:]
            if mine:
                return "Earlier you said: " + " | ".join(f"“{m['message'][:140]}”" for m in mine)
            summary = ctx.get("conversationSummary") or ""
            if summary:
                return f"Here's what we covered earlier: {summary[:400]}"
            return "I don't have earlier messages in this trip's conversation yet."

        if any(k in low for k in ("what can you", "what do you do", "help", "how do i use",
                                  "abilities", "what else")) and "book" not in low:
            return ("I can help with your days (today, tomorrow, day 4), your hotel and "
                    "transport, bookings, budget and spending, activities by interest, "
                    "dining in your plan, and trip facts like dates and travelers. "
                    "I can also move, add, or remove itinerary stops — just say the word.")

        if any(k in low for k in ("booking", "reservation", "booked")):
            if not bookings:
                return f"You have no bookings recorded for your {title} trip yet."
            parts = [f"{b['item_type']}: {b['booking_reference']} ({b['status']}, {b['currency']} {b['amount']:,.0f})" for b in bookings[:10]]
            return f"Your bookings for {dest}: " + "; ".join(parts) + "."

        if any(k in low for k in ("how do i book", "how can i book", "how to book",
                                  "make a booking", "reserve", "payment")):
            return ("To lock a booking, open the stop on your itinerary and choose it — "
                    "AI-guide bookings confirm instantly, self-bookings stay pending for "
                    "you to pay the vendor. Say 'show my bookings' anytime to review them.")

        if "breakdown" in low or "per person" in low or "per day" in low or "split" in low:
            total = float(budget.get("total_budget") or 0)
            travelers = max(1, int(trip.get("traveler_count") or 1))
            days = max(1, int(trip.get("duration_days") or 1))
            by_type: Dict[str, float] = {}
            for i in (it.get("upcoming") or []) + (it.get("completed") or []):
                by_type[i["item_type"]] = by_type.get(i["item_type"], 0.0) + float(i.get("cost") or 0)
            top = sorted(by_type.items(), key=lambda kv: -kv[1])[:4]
            detail = "; ".join(f"{k}: {fmt_money(v)}" for k, v in top) if top else "no itemized costs yet"
            return (f"Budget breakdown for {title}: total {fmt_money(total)} "
                    f"({fmt_money(total / travelers)} per person, ~{fmt_money(total / days)} per day). "
                    f"By type — {detail}. Bookings so far: {fmt_money(budget.get('current_spend'))}.")
        if any(k in low for k in ("spent", "spend", "budget", "cost", "how much")) and ("hotel" not in low and "lunch" not in low and "dinner" not in low or "spent" in low or "budget" in low):
            if "spent" in low or "budget" in low or "how much have" in low:
                return (f"Your total budget is {cur} {budget.get('total_budget', 0):,.0f}. "
                        f"Recorded bookings total {cur} {budget.get('current_spend', 0):,.0f}, "
                        f"leaving {cur} {budget.get('remaining_budget', 0):,.0f} remaining.")

        if any(k in low for k in ("weather", "rain", "forecast", "temperature")):
            info = ctx.get("destination_info") or {}
            season = info.get("best_time_to_visit")
            extra = f" Best season on record: {season}." if season else ""
            return (f"I don't have live weather data for {dest}, so I won't guess.{extra} "
                    f"Check a forecast app before you head out?")

        if (any(k in low for k in ("move", "reschedule", "shift", "change my", "change the",
                                   "change hotel", "switch hotel", "replace", "swap",
                                   "remove", "delete", "cancel my", "add "))
                and "book" not in low):
            parsed = self.parse_action(message)
            if parsed is not None:
                return "On it — applying that change now."
            return ("I can move, add, or remove stops — e.g. 'move Solang Valley to 11 AM', "
                    "'remove the market walk', or 'add paragliding on day 3'. What should I change?")
        if "tomorrow" in low:
            items = it.get("tomorrow") or []
            if not items:
                return f"Your itinerary information for tomorrow is currently unavailable — nothing is planned for that day yet."
            return f"Tomorrow's plan in {dest}: {fmt_items(items)}."
        if "today" in low and "best time" not in low:
            items = it.get("today") or []
            if not items:
                return "Your itinerary information for today is currently unavailable — nothing is planned for today yet."
            return f"Today's plan in {dest}: {fmt_items(items)}."
        day_m = re.search(r"day\s*(\d+)", low)
        if day_m:
            d = int(day_m.group(1))
            items = [i for i in (it.get("upcoming") or []) if i["day_number"] == d]
            if not items:
                return f"Your itinerary information for day {d} is currently unavailable."
            return f"Day {d} in {dest}: {fmt_items(items)}."

        if "best time" in low:
            season = (ctx.get("destination_info") or {}).get("best_time_to_visit")
            if season:
                return f"Best time to visit {dest}: {season}."
            return f"I don't have season information saved for {dest} yet."
        if any(k in low for k in ("tell me about", "about ", "describe", "famous for", "known for", "what is ")) and dest.lower() in low:
            desc = (ctx.get("destination_info") or {}).get("description")
            if desc:
                return f"{dest}: {desc[:400]}"
            return f"I don't have a description saved for {dest} yet."

        if any(k in low for k in ("lunch", "dinner", "breakfast", "eat", "restaurant", "food", "meal", "cuisine", "veg", "diet")):
            pool = (it.get("today") or []) + (it.get("tomorrow") or []) or (it.get("upcoming") or [])[:8]
            meals = [i for i in pool if i["item_type"] in ("meal",) or "lunch" in (i["title"] or "").lower() or "dinner" in (i["title"] or "").lower() or "restaurant" in (i["title"] or "").lower()]
            diet = (ctx.get("preferences") or {}).get("dietary_requirements") or []
            diet_note = f" Noted your preference: {', '.join(diet)}." if diet else ""
            if meals:
                return f"For meals, your plan has: {fmt_items(meals)}.{diet_note}"
            if pool:
                return (f"I don't have a specific restaurant saved for that meal — your nearest planned items are: "
                        f"{fmt_items(pool[:3])}.{diet_note} Tell me the day and I'll narrow it down.")
            return "I don't have restaurant information saved for this trip yet."

        if any(k in low for k in ("adventure", "thrill", "trek", "paraglid", "rafting", "ski", "adrenaline")):
            return self._catalog_answer(ctx, "adventure", "adventure", cur, dest)
        if any(k in low for k in ("culture", "cultural", "heritage", "temple", "museum", "history", "art")):
            return self._catalog_answer(ctx, "culture", "culture", cur, dest)
        if any(k in low for k in ("free", "cheap", "cheapest", "budget activit", "low cost")):
            return self._catalog_answer(ctx, None, "budget", cur, dest, sort_by="price")
        if any(k in low for k in ("activities", "experiences", "things to do", "what to do", "sightseeing", "attractions")):
            return self._catalog_answer(ctx, None, "top-rated", cur, dest)

        if any(k in low for k in ("hotels", "stays under", "accommodation options", "where can i stay",
                                  "compare", "cheapest stay", "list of hotels")):
            hotels = ((ctx.get("catalog") or {}).get("hotels") or [])[:5]
            if not hotels:
                return f"I don't have hotel options saved for {dest} yet."
            bits = [f"{h['name']} ({h['category']}, {h['currency']} {h['price_per_night']:,.0f}/night, rated {h['rating']})"
                    for h in hotels]
            return f"Hotel options in {dest}: " + "; ".join(bits) + ". Say 'change my hotel to …' and I'll switch it."
        if any(k in low for k in ("hotel", "stay", "accommodation", "nearby", "where is")):
            pool = (it.get("upcoming") or [])[:20]
            hotels = [i for i in pool if i["item_type"] == "hotel"]
            if hotels:
                h = hotels[0]
                extra = f" It's at {h['location']}." if h.get("location") else ""
                return f"Your hotel is {h['title']}.{extra} Let me know if you want details for a specific day."
            if pool:
                loc = pool[0].get("location")
                if loc:
                    return f"Your stays aren't itemized yet, but your activities center around {loc}. Add your hotel and I can check proximity."
            return "Your hotel information is currently unavailable for this trip."

        if any(k in low for k in ("transport", "cab", "taxi", "bus", "flight", "train",
                                  "transfer", "pickup", "pick-up", "drop", "self-drive", "drive")):
            pool = (it.get("upcoming") or [])[:20]
            trs = [i for i in pool if i["item_type"] == "transport"]
            if trs:
                t = trs[0]
                when = f" at {t['start_time']}" if t.get("start_time") else ""
                return f"Your planned transport: {t['title']}{when}. Anything to change about it?"
            return "No transfers are pre-booked for this trip — you'll arrange local transport yourself."

        if any(k in low for k in ("how long", "duration", "how many days", "long is")):
            days = trip.get("duration_days")
            s, e = self._fmt_dates(trip)
            return f"Your {title} trip is {days} days{(f' ({s} to {e})') if s else ''}."
        if any(k in low for k in ("when do", "when is", "start date", "end date", "dates", "when are")):
            s, e = self._fmt_dates(trip)
            if s:
                return f"Your {title} trip runs {s} to {e}."
            return "Your trip dates aren't set yet — adjust them on the trip page."
        if any(k in low for k in ("how many", "traveler", "people", "persons", "who is", "who's",
                                  "companions", "solo", "couple", "family", "friends")):
            n = trip.get("traveler_count") or 1
            comp = trip.get("companion_label") or f"{n} traveler(s)"
            return f"You're traveling as {comp} ({n} traveler(s)) on this trip."
        if "status" in low or "confirm" in low:
            st = trip.get("status") or "planning"
            if st == "confirmed":
                return f"Your {title} trip is confirmed — you're all set!"
            return f"Your {title} trip is currently {st}. Confirm it on the trip page when ready."

        if "book" in low and any(k in low for k in ("cancel", "change", "modify", "update")):
            if not bookings:
                return f"You have no bookings to change on your {title} trip yet."
            return ("I can't modify bookings from chat — manage them on the Trips page. "
                    f"Current bookings: " + "; ".join(f"{b['booking_reference']} ({b['status']})" for b in bookings[:5]) + ".")

        # Default: short snapshot, never a wall of text.
        upcoming = it.get("upcoming") or []
        if upcoming:
            return (f"For your {title} trip in {dest}: {fmt_items(upcoming[:2])}. "
                    f"Ask me about a specific day, your hotel, bookings, or budget?")
        return (f"For your {title} trip in {dest}, I have your budget "
                f"({fmt_money(budget.get('total_budget'))}) and {len(bookings)} booking(s) loaded. "
                f"What would you like to know?")

    @staticmethod
    def _fmt_dates(trip: Dict[str, Any]) -> tuple:
        s = (trip.get("start_date") or "")[:10] or None
        e = (trip.get("end_date") or "")[:10] or None
        return s, e

    def _catalog_answer(self, ctx: Dict[str, Any], category: Optional[str],
                        label: str, cur: str, dest: str,
                        sort_by: str = "rating") -> str:
        """Answer open activity questions from the real destination catalog."""
        acts = list(((ctx.get("catalog") or {}).get("activities") or []))
        if category:
            acts = [a for a in acts if (a.get("category") or "").lower() == category]
        if not acts:
            return f"I don't have {label} activities saved for {dest} yet."
        if sort_by == "price":
            acts = sorted(acts, key=lambda a: float(a.get("price_per_person") or 0))
        else:
            acts = sorted(acts, key=lambda a: -(float(a.get("rating") or 0)))
        bits = []
        for a in acts[:5]:
            price = float(a.get("price_per_person") or 0)
            price_txt = f"{a.get('currency') or cur} {price:,.0f}" if price else "free entry"
            bits.append(f"{a['title']} ({a.get('category')}, {price_txt}, rated {a.get('rating')})")
        prefs = (ctx.get("preferences") or {}).get("interests") or []
        tail = f" Matches your interest in {', '.join(prefs[:2])}." if prefs and category and category in prefs else ""
        return f"{label.capitalize()} picks in {dest}: " + "; ".join(bits) + f".{tail} Want one added to a day?"

    @staticmethod
    def describe_action(outcome: Optional[Dict[str, Any]],
                        parsed: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Human confirmation for an applied structured action."""
        if not outcome or not outcome.get("applied"):
            return None
        title = outcome.get("title") or "that stop"
        intent = outcome.get("intent")
        if intent in ("UPDATE_ITINERARY", "MOVE_ITEM"):
            new_time = (parsed or {}).get("new_time")
            return f"Done — moved {title}{f' to {new_time}' if new_time else ''}." if new_time else f"Done — updated {title}."
        if intent == "ADD_ACTIVITY":
            day = (parsed or {}).get("day_number")
            return f"Added {title}{f' to day {day}' if day else ''}."
        if intent == "REMOVE_ACTIVITY":
            return f"Removed {title} from your itinerary."
        if intent == "CHANGE_HOTEL":
            return f"Switched your hotel to {title}."
        return f"Done — {intent} applied to {title}."

    def generate(self, message: str, ctx: Dict[str, Any]) -> str:
        """Gemini-backed generation with compact grounded prompt; falls back safely."""
        if self.gemini is None or not self.gemini.is_available():
            return self.answer_from_context(message, ctx)
        try:
            catalog = ctx.get("catalog") or {}
            compact = {
                "system": GUIDE_SYSTEM_INSTRUCTION,
                "user": ctx.get("user"),
                "trip": ctx.get("trip"),
                "today": (ctx.get("itinerary") or {}).get("today"),
                "tomorrow": (ctx.get("itinerary") or {}).get("tomorrow"),
                "upcoming_first_8": ((ctx.get("itinerary") or {}).get("upcoming") or [])[:8],
                "bookings": ctx.get("bookings"),
                "budget": ctx.get("budget"),
                "destination_info": ctx.get("destination_info"),
                "preferences": ctx.get("preferences"),
                "catalog_activities": (catalog.get("activities") or [])[:8],
                "catalog_hotels": (catalog.get("hotels") or [])[:5],
                "summary": ctx.get("conversationSummary"),
                "recent": ctx.get("recentMessages"),
            }
            import json as _json
            prompt = (GUIDE_SYSTEM_INSTRUCTION
                      + "\n\nREAL CONTEXT (JSON, source of truth):\n" + _json.dumps(compact, default=str)[:6000]
                      + "\n\nTraveler message: " + message
                      + "\n\nReply concisely, grounded only in the context above. "
                        "If info is missing, say it is unavailable. "
                        "If the user requests an itinerary change, describe it in words AND append a JSON block "
                        '{"intent":"UPDATE_ITINERARY","action":"MOVE_ITEM","item_title":"...","new_time":"..."} '
                        "only when a real item matches.")
            from backend.ai.gemini_service import GEMINI_MODEL_FALLBACKS
            models = list(GEMINI_MODEL_FALLBACKS)
            for m in models:
                try:
                    resp = self.gemini.client.models.generate_content(model=m, contents=prompt)
                    text = (resp.text or "").strip()
                    if text:
                        return text
                except Exception as exc:
                    logger.warning("Guide model %s failed: %s", m, exc)
        except Exception as exc:
            logger.warning("Guide generation failed, using grounded fallback: %s", exc)
        return self.answer_from_context(message, ctx)
