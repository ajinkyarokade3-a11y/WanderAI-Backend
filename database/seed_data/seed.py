import logging
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from backend.database.connection import SessionLocal, engine, Base
from backend.models.models import (
    User, TravelerProfile, Destination, Vendor, Hotel, Activity,
    TransportOption, Trip, TripPreference, ItineraryItem, Booking,
    Notification, Alert, ChangeHistory, Review, Vehicle, Driver
)

logger = logging.getLogger("tourflow_seed")


def _goa_catalog_rows():
    """Real Goa catalog inventory shared by fresh-seed and backfill paths.

    Goa shipped with a hotel but zero activities / transport options, so the
    trip validator honestly refused to build Goa trips. These rows close that
    gap (5 activities support up to 6-day trips; 3 transports cover solo to
    group capacities). All PKs are new and never collide with existing seed.
    """
    vendors = [
        Vendor(id="vnd-goa-004", name="Konkan Coastal Stays", vendor_type="hotel",
               contact_email="stay@konkancoastal.in", phone="+91 83224 00104",
               rating=4.7, is_verified=True),
        Vendor(id="vnd-goa-005", name="Goa Adventure & Watersports Co.", vendor_type="activity",
               contact_email="hello@goawatersports.in", phone="+91 98221 30045",
               rating=4.6, is_verified=True),
        Vendor(id="vnd-goa-006", name="Konkan Coastal Mobility", vendor_type="transport",
               contact_email="dispatch@konkanmobility.in", phone="+91 83224 00106",
               rating=4.7, is_verified=True),
    ]
    hotels = [
        Hotel(id="htl-goa-002", destination_id="dest-goa-002", vendor_id="vnd-goa-004",
              name="Casa Baga Beach Boutique", category="mid-range",
              price_per_night=7500.0, currency="INR", rating=4.5,
              address="Tito's Lane, Baga, Goa 403516",
              amenities=["Rooftop Pool", "Yoga Deck", "Beach Shacks Nearby", "Co-working Nook"],
              images=["https://images.unsplash.com/photo-1520250497591-112f2f40a3f4?auto=format&fit=crop&w=800&q=80"],
              description="Design-led boutique stay two minutes from Baga beach with a rooftop pool and slow-morning café.",
              latitude=15.5557, longitude=73.7513),
    ]
    activities = [
        Activity(id="act-goa-001", destination_id="dest-goa-002", vendor_id="vnd-goa-005",
                 title="Dudhsagar Falls & Mollem Spice Plantation Trail", category="nature",
                 duration_hours=6.0, price_per_person=2200.0, currency="INR",
                 difficulty_level="easy", rating=4.7, capacity=24,
                 images=["https://images.unsplash.com/photo-1506905925346-21bda4d32df4?auto=format&fit=crop&w=800&q=80"],
                 description="Jeep trail through Mollem forest to the 310m Dudhsagar cascade, plus a guided spice plantation lunch.",
                 meeting_point="Mollem National Park Gate",
                 latitude=15.3144, longitude=74.3144),
        Activity(id="act-goa-002", destination_id="dest-goa-002", vendor_id="vnd-goa-005",
                 title="Baga Parasailing & Jet Ski Watersports Combo", category="adventure",
                 duration_hours=2.0, price_per_person=2800.0, currency="INR",
                 difficulty_level="moderate", rating=4.6, capacity=12,
                 images=["https://images.unsplash.com/photo-1502680390469-be75c86b636f?auto=format&fit=crop&w=800&q=80"],
                 description="Tandem parasail over Baga bay followed by a jet ski session with certified instructors and safety boat.",
                 meeting_point="Baga Beach Watersports Kiosk",
                 latitude=15.5553, longitude=73.7517),
        Activity(id="act-goa-003", destination_id="dest-goa-002", vendor_id="vnd-goa-005",
                 title="Old Goa Basilica & Fontainhas Heritage Walk", category="culture",
                 duration_hours=3.0, price_per_person=1200.0, currency="INR",
                 difficulty_level="easy", rating=4.8, capacity=20,
                 images=["https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?auto=format&fit=crop&w=800&q=80"],
                 description="UNESCO-listed Basilica of Bom Jesus and Se Cathedral, then the Latin-quarter lanes of Fontainhas with a local historian.",
                 meeting_point="Basilica of Bom Jesus Forecourt",
                 latitude=15.5009, longitude=73.9116),
        Activity(id="act-goa-004", destination_id="dest-goa-002", vendor_id="vnd-goa-005",
                 title="Mandovi River Sunset Cruise with Goan Folk Music", category="relaxation",
                 duration_hours=2.0, price_per_person=1500.0, currency="INR",
                 difficulty_level="easy", rating=4.5, capacity=60,
                 images=["https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=800&q=80"],
                 description="Evening cruise from Panaji jetty with live mando music, Goan snacks, and dolphin-spotting on lucky days.",
                 meeting_point="Santa Monica Jetty, Panaji",
                 latitude=15.4989, longitude=73.8278),
        Activity(id="act-goa-005", destination_id="dest-goa-002", vendor_id="vnd-goa-005",
                 title="Anjuna Flea Market & Beach Café Tasting Trail", category="culinary",
                 duration_hours=3.0, price_per_person=1000.0, currency="INR",
                 difficulty_level="easy", rating=4.6, capacity=25,
                 images=["https://images.unsplash.com/photo-1555396273-367ea4eb4db5?auto=format&fit=crop&w=800&q=80"],
                 description="Wednesday flea market bargaining plus a guided tasting of bebinca, poi, and single-estate Goan coffee.",
                 meeting_point="Anjuna Market Main Gate",
                 latitude=15.5730, longitude=73.7403),
        Activity(id="act-goa-006", destination_id="dest-goa-002", vendor_id="vnd-goa-005",
                 title="Fort Aguada Sunset Point & Lighthouse Visit", category="culture",
                 duration_hours=2.0, price_per_person=800.0, currency="INR",
                 difficulty_level="easy", rating=4.6, capacity=30,
                 images=["https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?auto=format&fit=crop&w=800&q=80"],
                 description="Portuguese ramparts, lighthouse views, and sunset over the Arabian Sea.",
                 meeting_point="Aguada Fort Gate", latitude=15.4920, longitude=73.7730),
        Activity(id="act-goa-007", destination_id="dest-goa-002", vendor_id="vnd-goa-005",
                 title="Morjim Turtle Watch & Beach Morning", category="nature",
                 duration_hours=3.0, price_per_person=1300.0, currency="INR",
                 difficulty_level="easy", rating=4.5, capacity=20,
                 images=["https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=800&q=80"],
                 description="Quiet Olive Ridley nesting beach with a naturalist, plus Russian-café breakfast.",
                 meeting_point="Morjim Beach Entry", latitude=15.6290, longitude=73.7370),
        Activity(id="act-goa-008", destination_id="dest-goa-002", vendor_id="vnd-goa-005",
                 title="Ponda Feni Distillery Tour & Tasting", category="culinary",
                 duration_hours=2.5, price_per_person=1400.0, currency="INR",
                 difficulty_level="easy", rating=4.6, capacity=15,
                 images=["https://images.unsplash.com/photo-1555396273-367ea4eb4db5?auto=format&fit=crop&w=800&q=80"],
                 description="Cashew feni distillation, tasting flight, and Goan-Portuguese snacks at a heritage still.",
                 meeting_point="Ponda Distillery Gate", latitude=15.4020, longitude=74.0120),
    ]
    transports = [
        TransportOption(id="trn-goa-001", destination_id="dest-goa-002", vendor_id="vnd-goa-006",
                        type="private_cab", name="Toyota Innova Crysta Coastal Cab",
                        route_from="Goa International Airport (Dabolim)",
                        route_to="Benaulim & South Goa Resorts",
                        duration_hours=1.5, price=3200.0, currency="INR", capacity=6,
                        features=["AC", "Luggage Carrier", "Flight Tracking", "Child Seat on Request"]),
        TransportOption(id="trn-goa-002", destination_id="dest-goa-002", vendor_id="vnd-goa-006",
                        type="volvo_bus", name="Intercity AC Sleeper Coach",
                        route_from="Mumbai Dadar",
                        route_to="Panaji Kadamba Terminal",
                        duration_hours=12.0, price=1600.0, currency="INR", capacity=32,
                        features=["Sleeper Berths", "Blankets", "Charging Points", "Live Tracking"]),
        TransportOption(id="trn-goa-003", destination_id="dest-goa-002", vendor_id="vnd-goa-006",
                        type="self_drive", name="Honda Activa Scooter Rental (Helmets Included)",
                        route_from="Baga Hub",
                        route_to="North Goa Beach Circuit",
                        duration_hours=24.0, price=600.0, currency="INR", capacity=2,
                        features=["Two Helmets", "Phone Mount", "Full Tank Option", "Roadside Assistance"]),
    ]
    return {Vendor: vendors, Hotel: hotels, Activity: activities, TransportOption: transports}


def _kashmir_catalog_rows():
    """Real Kashmir inventory: 5 activities cover up to 6-day trips."""
    vendors = [
        Vendor(id="vnd-kas-007", name="Dal Lake Heritage Stays", vendor_type="hotel",
               contact_email="stay@dallakeheritage.in", phone="+91 19423 10007",
               rating=4.8, is_verified=True),
        Vendor(id="vnd-kas-008", name="Himalayan Valley Adventures", vendor_type="activity",
               contact_email="hello@valleyadventures.in", phone="+91 99065 30008",
               rating=4.7, is_verified=True),
        Vendor(id="vnd-kas-009", name="Kashmir Valley Mobility", vendor_type="transport",
               contact_email="dispatch@kashmirmobility.in", phone="+91 19423 10009",
               rating=4.7, is_verified=True),
    ]
    hotels = [
        Hotel(id="htl-kas-001", destination_id="dest-kashmir-005", vendor_id="vnd-kas-007",
              name="Dal Lake Deluxe Heritage Houseboat", category="luxury",
              price_per_night=18000.0, currency="INR", rating=4.8,
              address="Dal Lake, Srinagar 190001",
              amenities=["Kahwa on Deck", "Kani Shawl Decor", "Shikara Transfers", "Wazwan Dining"],
              images=["https://images.unsplash.com/photo-1566837945700-30057527ade0?auto=format&fit=crop&w=800&q=80"],
              description="Hand-carved walnut wood houseboat with sun deck facing the Zabarwan mountains.",
              latitude=34.1144, longitude=74.8655),
        Hotel(id="htl-kas-002", destination_id="dest-kashmir-005", vendor_id="vnd-kas-007",
              name="Srinagar Old City Boutique Hotel", category="mid-range",
              price_per_night=8500.0, currency="INR", rating=4.5,
              address="Rajbagh, Srinagar 190008",
              amenities=["Chinar Garden", "Kahwa Lounge", "Heated Rooms", "City Tours"],
              images=["https://images.unsplash.com/photo-1595815771614-ade9d652a65d?auto=format&fit=crop&w=800&q=80"],
              description="Boutique stay under chinar trees, minutes from Lal Chowk and the Jhelum riverfront.",
              latitude=34.0837, longitude=74.7973),
    ]
    activities = [
        Activity(id="act-kas-001", destination_id="dest-kashmir-005", vendor_id="vnd-kas-008",
                 title="Dal Lake Sunrise Shikara Ride & Floating Market", category="relaxation",
                 duration_hours=3.0, price_per_person=1800.0, currency="INR",
                 difficulty_level="easy", rating=4.9, capacity=20,
                 images=["https://images.unsplash.com/photo-1566837945700-30057527ade0?auto=format&fit=crop&w=800&q=80"],
                 description="Dawn shikara glide past lotus gardens to the floating vegetable market with kahwa onboard.",
                 meeting_point="Dal Gate 1 Jetty", latitude=34.1144, longitude=74.8655),
        Activity(id="act-kas-002", destination_id="dest-kashmir-005", vendor_id="vnd-kas-008",
                 title="Gulmarg Gondola & Apharwat Snow Experience", category="adventure",
                 duration_hours=6.0, price_per_person=4500.0, currency="INR",
                 difficulty_level="moderate", rating=4.8, capacity=24,
                 images=["https://images.unsplash.com/photo-1517824806704-9040b037703b?auto=format&fit=crop&w=800&q=80"],
                 description="Phase-1 and 2 gondola ascent to 3,980m with guided snow walks and ski options in season.",
                 meeting_point="Gulmarg Gondola Base", latitude=34.0484, longitude=74.3805),
        Activity(id="act-kas-003", destination_id="dest-kashmir-005", vendor_id="vnd-kas-008",
                 title="Pahalgam Betaab Valley & Aru Meadows Day Trip", category="nature",
                 duration_hours=7.0, price_per_person=2800.0, currency="INR",
                 difficulty_level="easy", rating=4.7, capacity=30,
                 images=["https://images.unsplash.com/photo-1506905925346-21bda4d32df4?auto=format&fit=crop&w=800&q=80"],
                 description="Lidder river picnic spots, Betaab valley pines, and shepherd trails in Aru with a local guide.",
                 meeting_point="Pahalgam Taxi Stand", latitude=34.0151, longitude=75.1920),
        Activity(id="act-kas-004", destination_id="dest-kashmir-005", vendor_id="vnd-kas-008",
                 title="Mughal Gardens Heritage Walk (Nishat & Shalimar)", category="culture",
                 duration_hours=3.0, price_per_person=1200.0, currency="INR",
                 difficulty_level="easy", rating=4.7, capacity=25,
                 images=["https://images.unsplash.com/photo-1587474260584-136574528ed5?auto=format&fit=crop&w=800&q=80"],
                 description="Terraced chinar gardens, fountains, and pavilions with stories of Shah Jahan's Kashmir.",
                 meeting_point="Nishat Bagh Gate", latitude=34.1244, longitude=74.8799),
        Activity(id="act-kas-005", destination_id="dest-kashmir-005", vendor_id="vnd-kas-008",
                 title="Traditional Kashmiri Wazwan Tasting Dinner", category="culinary",
                 duration_hours=2.5, price_per_person=1600.0, currency="INR",
                 difficulty_level="easy", rating=4.8, capacity=40,
                 images=["https://images.unsplash.com/photo-1555396273-367ea4eb4db5?auto=format&fit=crop&w=800&q=80"],
                 description="Seven-course wazwan with gustaba, rogan josh, and phirni at a heritage Srinagar kitchen.",
                 meeting_point="Downtown Srinagar Wazwan House", latitude=34.0837, longitude=74.7973),
        Activity(id="act-kas-006", destination_id="dest-kashmir-005", vendor_id="vnd-kas-008",
                 title="Shankaracharya Hill Sunrise Viewpoint", category="nature",
                 duration_hours=2.5, price_per_person=1000.0, currency="INR",
                 difficulty_level="easy", rating=4.7, capacity=20,
                 images=["https://images.unsplash.com/photo-1506905925346-21bda4d32df4?auto=format&fit=crop&w=800&q=80"],
                 description="Dawn over Dal Lake and the city from the ancient hill temple with a local guide.",
                 meeting_point="Shankaracharya Gate", latitude=34.0735, longitude=74.8445),
        Activity(id="act-kas-007", destination_id="dest-kashmir-005", vendor_id="vnd-kas-008",
                 title="Dachigam Wildlife Safari Drive", category="adventure",
                 duration_hours=4.0, price_per_person=3500.0, currency="INR",
                 difficulty_level="moderate", rating=4.6, capacity=12,
                 images=["https://images.unsplash.com/photo-1517824806704-9040b037703b?auto=format&fit=crop&w=800&q=80"],
                 description="Hangul deer and black bear country in the Zabarwan forests with a wildlife ranger.",
                 meeting_point="Dachigam Gate", latitude=34.1341, longitude=74.9011),
        Activity(id="act-kas-008", destination_id="dest-kashmir-005", vendor_id="vnd-kas-008",
                 title="Papier-Mâché Craft Workshop", category="culture",
                 duration_hours=2.0, price_per_person=1100.0, currency="INR",
                 difficulty_level="easy", rating=4.6, capacity=15,
                 images=["https://images.unsplash.com/photo-1587474260584-136574528ed5?auto=format&fit=crop&w=800&q=80"],
                 description="Paint your own box with master artisans in a downtown Srinagar karkhana.",
                 meeting_point="Downtown Craft Centre", latitude=34.0900, longitude=74.8000),
    ]
    transports = [
        TransportOption(id="trn-kas-001", destination_id="dest-kashmir-005", vendor_id="vnd-kas-009",
                        type="private_cab", name="Toyota Innova Valley Cab",
                        route_from="Srinagar International Airport",
                        route_to="Dal Lake & Srinagar Hotels",
                        duration_hours=1.0, price=2800.0, currency="INR", capacity=6,
                        features=["AC", "Heating", "Flight Tracking", "Local Driver Guide"]),
        TransportOption(id="trn-kas-002", destination_id="dest-kashmir-005", vendor_id="vnd-kas-009",
                        type="volvo_bus", name="Intercity AC Coach Service",
                        route_from="Jammu Tawi",
                        route_to="Srinagar TRC Terminal",
                        duration_hours=8.0, price=1400.0, currency="INR", capacity=32,
                        features=["Pushback Seats", "Blankets", "Charging Points", "Live Tracking"]),
        TransportOption(id="trn-kas-003", destination_id="dest-kashmir-005", vendor_id="vnd-kas-009",
                        type="self_drive", name="XUV700 Valley Rental SUV",
                        route_from="Srinagar Hub",
                        route_to="Gulmarg & Pahalgam Circuit",
                        duration_hours=24.0, price=5200.0, currency="INR", capacity=5,
                        features=["AWD", "Heated Seats", "GPS Navigation", "Zero-Dep Insurance"]),
    ]
    return {Vendor: vendors, Hotel: hotels, Activity: activities, TransportOption: transports}


def _kerala_catalog_rows():
    """Real Kerala inventory: 5 activities cover up to 6-day trips."""
    vendors = [
        Vendor(id="vnd-ker-010", name="Backwater Heritage Stays", vendor_type="hotel",
               contact_email="stay@backwaterheritage.in", phone="+91 48423 10010",
               rating=4.7, is_verified=True),
        Vendor(id="vnd-ker-011", name="Malabar Experiences Co.", vendor_type="activity",
               contact_email="hello@malabarexp.in", phone="+91 98470 30011",
               rating=4.7, is_verified=True),
        Vendor(id="vnd-ker-012", name="Kerala Coastal Mobility", vendor_type="transport",
               contact_email="dispatch@keralamobility.in", phone="+91 48423 10012",
               rating=4.6, is_verified=True),
    ]
    hotels = [
        Hotel(id="htl-ker-001", destination_id="dest-kerala-003", vendor_id="vnd-ker-010",
              name="Alleppey Premium Backwater Houseboat", category="luxury",
              price_per_night=20000.0, currency="INR", rating=4.8,
              address="Punnamada Finishing Point, Alappuzha 688013",
              amenities=["Sun Deck", "Kerala Meals Onboard", "AC Bedrooms", "Sunset Cruise"],
              images=["https://images.unsplash.com/photo-1602216056096-3b40cc0c9944?auto=format&fit=crop&w=800&q=80"],
              description="Two-bedroom kettuvallam with chef onboard, drifting through palm-fringed canals.",
              latitude=9.4981, longitude=76.3388),
        Hotel(id="htl-ker-002", destination_id="dest-kerala-003", vendor_id="vnd-ker-010",
              name="Fort Kochi Heritage Boutique", category="boutique",
              price_per_night=9000.0, currency="INR", rating=4.6,
              address="Rose Street, Fort Kochi 682001",
              amenities=["Portuguese Courtyard", "Ayurveda Room", "Art Café", "Harbour Walks"],
              images=["https://images.unsplash.com/photo-1582719478250-c89cae4dc85b?auto=format&fit=crop&w=800&q=80"],
              description="300-year-old Portuguese merchant home turned boutique stay near the Chinese nets.",
              latitude=9.9658, longitude=76.2421),
    ]
    activities = [
        Activity(id="act-ker-001", destination_id="dest-kerala-003", vendor_id="vnd-ker-011",
                 title="Alleppey Backwater Day Cruise with Village Visit", category="relaxation",
                 duration_hours=5.0, price_per_person=2500.0, currency="INR",
                 difficulty_level="easy", rating=4.8, capacity=30,
                 images=["https://images.unsplash.com/photo-1602216056096-3b40cc0c9944?auto=format&fit=crop&w=800&q=80"],
                 description="Shikara cruise through canals with a coir-village stop and toddy-shop lunch option.",
                 meeting_point="Punnamada Jetty", latitude=9.4981, longitude=76.3388),
        Activity(id="act-ker-002", destination_id="dest-kerala-003", vendor_id="vnd-ker-011",
                 title="Munnar Tea Estate Walk & Factory Tasting", category="nature",
                 duration_hours=4.0, price_per_person=1800.0, currency="INR",
                 difficulty_level="easy", rating=4.7, capacity=25,
                 images=["https://images.unsplash.com/photo-1506905925346-21bda4d32df4?auto=format&fit=crop&w=800&q=80"],
                 description="Guided walk through Kolukkumalai slopes with orthodox tea tasting at the factory.",
                 meeting_point="Munnar Tea Museum", latitude=10.0889, longitude=77.0595),
        Activity(id="act-ker-003", destination_id="dest-kerala-003", vendor_id="vnd-ker-011",
                 title="Kathakali & Kalaripayattu Evening Show", category="culture",
                 duration_hours=2.0, price_per_person=900.0, currency="INR",
                 difficulty_level="easy", rating=4.7, capacity=60,
                 images=["https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?auto=format&fit=crop&w=800&q=80"],
                 description="Classical dance-drama with live chenda percussion plus a martial-arts demonstration.",
                 meeting_point="Fort Kochi Cultural Centre", latitude=9.9658, longitude=76.2421),
        Activity(id="act-ker-004", destination_id="dest-kerala-003", vendor_id="vnd-ker-011",
                 title="Authentic Ayurvedic Abhyanga Spa Session", category="relaxation",
                 duration_hours=2.0, price_per_person=2200.0, currency="INR",
                 difficulty_level="easy", rating=4.6, capacity=10,
                 images=["https://images.unsplash.com/photo-1544161515-4ab6ce6db874?auto=format&fit=crop&w=800&q=80"],
                 description="Doctor-consulted four-hand oil massage with herbal steam at a NABH-accredited centre.",
                 meeting_point="Kochi Ayurveda Centre", latitude=9.9312, longitude=76.2673),
        Activity(id="act-ker-005", destination_id="dest-kerala-003", vendor_id="vnd-ker-011",
                 title="Fort Kochi Seafood & Spice Trail", category="culinary",
                 duration_hours=3.0, price_per_person=1400.0, currency="INR",
                 difficulty_level="easy", rating=4.6, capacity=20,
                 images=["https://images.unsplash.com/photo-1555396273-367ea4eb4db5?auto=format&fit=crop&w=800&q=80"],
                 description="Chinese-net auction viewing, spice warehouse stories, and karimeen tastings with a chef.",
                 meeting_point="Fort Kochi Beach Nets", latitude=9.9658, longitude=76.2421),
        Activity(id="act-ker-006", destination_id="dest-kerala-003", vendor_id="vnd-ker-011",
                 title="Kovalam Beginner Surf Lesson", category="adventure",
                 duration_hours=2.0, price_per_person=2000.0, currency="INR",
                 difficulty_level="moderate", rating=4.5, capacity=10,
                 images=["https://images.unsplash.com/photo-1602216056096-3b40cc0c9944?auto=format&fit=crop&w=800&q=80"],
                 description="Pop-up practice on the sand then guided waves with ISA-certified instructors.",
                 meeting_point="Kovalam Surf School", latitude=8.4004, longitude=76.9787),
        Activity(id="act-ker-007", destination_id="dest-kerala-003", vendor_id="vnd-ker-011",
                 title="Jew Town & Synagogue Heritage Walk", category="culture",
                 duration_hours=2.5, price_per_person=1000.0, currency="INR",
                 difficulty_level="easy", rating=4.7, capacity=20,
                 images=["https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?auto=format&fit=crop&w=800&q=80"],
                 description="Antique lanes, spice godowns, and the 16th-century Paradesi Synagogue with a historian.",
                 meeting_point="Jew Town Gate", latitude=9.9577, longitude=76.2596),
        Activity(id="act-ker-008", destination_id="dest-kerala-003", vendor_id="vnd-ker-011",
                 title="Periyar Wildlife Boat Safari", category="nature",
                 duration_hours=3.5, price_per_person=2400.0, currency="INR",
                 difficulty_level="easy", rating=4.7, capacity=30,
                 images=["https://images.unsplash.com/photo-1506905925346-21bda4d32df4?auto=format&fit=crop&w=800&q=80"],
                 description="Elephant and sambar sightings from the lake launch with a forest guide.",
                 meeting_point="Thekkady Boat Landing", latitude=9.6026, longitude=76.9426),
    ]
    transports = [
        TransportOption(id="trn-ker-001", destination_id="dest-kerala-003", vendor_id="vnd-ker-012",
                        type="private_cab", name="Toyota Innova Backwater Cab",
                        route_from="Kochi International Airport",
                        route_to="Fort Kochi & Alleppey Stays",
                        duration_hours=2.0, price=3400.0, currency="INR", capacity=6,
                        features=["AC", "Flight Tracking", "Luggage Carrier", "English Driver"]),
        TransportOption(id="trn-ker-002", destination_id="dest-kerala-003", vendor_id="vnd-ker-012",
                        type="volvo_bus", name="Intercity AC Sleeper Coach",
                        route_from="Bengaluru Majestic",
                        route_to="Kochi Vyttila Hub",
                        duration_hours=11.0, price=1500.0, currency="INR", capacity=32,
                        features=["Sleeper Berths", "Blankets", "Charging Points", "Live Tracking"]),
        TransportOption(id="trn-ker-003", destination_id="dest-kerala-003", vendor_id="vnd-ker-012",
                        type="self_drive", name="Creta Coastal Rental SUV",
                        route_from="Kochi Hub",
                        route_to="Alleppey & Munnar Circuit",
                        duration_hours=24.0, price=4800.0, currency="INR", capacity=5,
                        features=["GPS Navigation", "Zero-Dep Insurance", "Extra Driver Free"]),
    ]
    return {Vendor: vendors, Hotel: hotels, Activity: activities, TransportOption: transports}


def _rajasthan_catalog_rows():
    """Real Rajasthan inventory: 5 activities cover up to 6-day trips."""
    vendors = [
        Vendor(id="vnd-raj-013", name="Rajputana Palace Stays", vendor_type="hotel",
               contact_email="stay@rajputanapalace.in", phone="+91 14123 10013",
               rating=4.8, is_verified=True),
        Vendor(id="vnd-raj-014", name="Desert Heritage Experiences", vendor_type="activity",
               contact_email="hello@desertheritage.in", phone="+91 98290 30014",
               rating=4.7, is_verified=True),
        Vendor(id="vnd-raj-015", name="Rajputana Royal Mobility", vendor_type="transport",
               contact_email="dispatch@rajputanamobility.in", phone="+91 14123 10015",
               rating=4.7, is_verified=True),
    ]
    hotels = [
        Hotel(id="htl-raj-001", destination_id="dest-rajasthan-004", vendor_id="vnd-raj-013",
              name="Jaipur Heritage Palace Hotel", category="luxury",
              price_per_night=22000.0, currency="INR", rating=4.8,
              address="Amer Road, Jaipur 302002",
              amenities=["Frescoed Suites", "Rooftop with Fort View", "Royal Spa", "Heritage Pool"],
              images=["https://images.unsplash.com/photo-1599661046289-e31897846e41?auto=format&fit=crop&w=800&q=80"],
              description="200-year-old Rajput palace with mirror-work halls facing Amer Fort.",
              latitude=26.9124, longitude=75.7873),
        Hotel(id="htl-raj-002", destination_id="dest-rajasthan-004", vendor_id="vnd-raj-013",
              name="Pink City Courtyard Haveli", category="mid-range",
              price_per_night=8000.0, currency="INR", rating=4.5,
              address="Johari Bazaar, Jaipur 302003",
              amenities=["Painted Courtyard", "Rooftop Café", "Block-Print Workshops", "Old City Walks"],
              images=["https://images.unsplash.com/photo-1477587458883-47145ed94245?auto=format&fit=crop&w=800&q=80"],
              description="Restored merchant haveli with frescoed rooms inside the walled city.",
              latitude=26.9124, longitude=75.7873),
    ]
    activities = [
        Activity(id="act-raj-001", destination_id="dest-rajasthan-004", vendor_id="vnd-raj-014",
                 title="Amber Fort & Sheesh Mahal Guided Tour", category="culture",
                 duration_hours=4.0, price_per_person=2000.0, currency="INR",
                 difficulty_level="easy", rating=4.8, capacity=30,
                 images=["https://images.unsplash.com/photo-1599661046289-e31897846e41?auto=format&fit=crop&w=800&q=80"],
                 description="Rampart walk, mirror palace, and light-and-sound show option with a historian.",
                 meeting_point="Amber Fort Main Gate", latitude=26.9855, longitude=75.8513),
        Activity(id="act-raj-002", destination_id="dest-rajasthan-004", vendor_id="vnd-raj-014",
                 title="Sam Sand Dunes Camel Safari & Folk Night", category="adventure",
                 duration_hours=5.0, price_per_person=3200.0, currency="INR",
                 difficulty_level="moderate", rating=4.7, capacity=40,
                 images=["https://images.unsplash.com/photo-1509316785289-025f5b846b35?auto=format&fit=crop&w=800&q=80"],
                 description="Sunset camel safari with Kalbeliya dancers and dinner under desert stars.",
                 meeting_point="Sam Dunes Camp Gate", latitude=26.8139, longitude=70.5056),
        Activity(id="act-raj-003", destination_id="dest-rajasthan-004", vendor_id="vnd-raj-014",
                 title="City Palace, Jantar Mantar & Bazaar Walk", category="culture",
                 duration_hours=3.5, price_per_person=1500.0, currency="INR",
                 difficulty_level="easy", rating=4.7, capacity=25,
                 images=["https://images.unsplash.com/photo-1477587458883-47145ed94245?auto=format&fit=crop&w=800&q=80"],
                 description="Royal residence museums, the stone observatory, and Johari Bazaar with a shopping guide.",
                 meeting_point="City Palace Gate", latitude=26.9245, longitude=75.8267),
        Activity(id="act-raj-004", destination_id="dest-rajasthan-004", vendor_id="vnd-raj-014",
                 title="Rajasthani Royal Thali Tasting Trail", category="culinary",
                 duration_hours=3.0, price_per_person=1300.0, currency="INR",
                 difficulty_level="easy", rating=4.6, capacity=20,
                 images=["https://images.unsplash.com/photo-1555396273-367ea4eb4db5?auto=format&fit=crop&w=800&q=80"],
                 description="Dal-baati feasts, ghewar tasting, and masala-chai stops across the old city.",
                 meeting_point="Bapu Bazaar Gate", latitude=26.9124, longitude=75.7873),
        Activity(id="act-raj-005", destination_id="dest-rajasthan-004", vendor_id="vnd-raj-014",
                 title="Pushkar Lake Ghats & Brahma Temple Visit", category="relaxation",
                 duration_hours=4.0, price_per_person=1700.0, currency="INR",
                 difficulty_level="easy", rating=4.6, capacity=25,
                 images=["https://images.unsplash.com/photo-1506905925346-21bda4d32df4?auto=format&fit=crop&w=800&q=80"],
                 description="Sacred ghat aarti, camel-fair grounds, and the world's rare Brahma temple.",
                 meeting_point="Pushkar Ghat Steps", latitude=26.4897, longitude=74.5511),
        Activity(id="act-raj-006", destination_id="dest-rajasthan-004", vendor_id="vnd-raj-014",
                 title="Nahargarh Sunset & Stepwell Tour", category="nature",
                 duration_hours=3.0, price_per_person=1400.0, currency="INR",
                 difficulty_level="easy", rating=4.6, capacity=25,
                 images=["https://images.unsplash.com/photo-1477587458883-47145ed94245?auto=format&fit=crop&w=800&q=80"],
                 description="Fort ramparts at golden hour plus the Chand Baori-style stepwell with a photographer guide.",
                 meeting_point="Nahargarh Gate", latitude=26.9374, longitude=75.8152),
        Activity(id="act-raj-007", destination_id="dest-rajasthan-004", vendor_id="vnd-raj-014",
                 title="Chokhi Dhani Folk Evening with Dinner", category="culture",
                 duration_hours=4.0, price_per_person=2200.0, currency="INR",
                 difficulty_level="easy", rating=4.7, capacity=50,
                 images=["https://images.unsplash.com/photo-1599661046289-e31897846e41?auto=format&fit=crop&w=800&q=80"],
                 description="Puppet shows, Kalbeliya dance, camel rides, and a Rajasthani village feast.",
                 meeting_point="Chokhi Dhani Gate", latitude=26.7680, longitude=75.6610),
        Activity(id="act-raj-008", destination_id="dest-rajasthan-004", vendor_id="vnd-raj-014",
                 title="Ranthambore Tiger Safari Drive", category="adventure",
                 duration_hours=4.0, price_per_person=4500.0, currency="INR",
                 difficulty_level="moderate", rating=4.8, capacity=12,
                 images=["https://images.unsplash.com/photo-1509316785289-025f5b846b35?auto=format&fit=crop&w=800&q=80"],
                 description="Morning gypsy safari through lakes and ruins with a naturalist tracker.",
                 meeting_point="Ranthambore Gate", latitude=26.0173, longitude=76.5026),
    ]
    transports = [
        TransportOption(id="trn-raj-001", destination_id="dest-rajasthan-004", vendor_id="vnd-raj-015",
                        type="private_cab", name="Toyota Innova Heritage Cab",
                        route_from="Jaipur International Airport",
                        route_to="Pink City & Amer Hotels",
                        duration_hours=1.0, price=2600.0, currency="INR", capacity=6,
                        features=["AC", "Flight Tracking", "Guide on Request", "Luggage Carrier"]),
        TransportOption(id="trn-raj-002", destination_id="dest-rajasthan-004", vendor_id="vnd-raj-015",
                        type="volvo_bus", name="Intercity AC Sleeper Coach",
                        route_from="Delhi Kashmere Gate",
                        route_to="Jaipur Sindhi Camp",
                        duration_hours=6.0, price=1100.0, currency="INR", capacity=32,
                        features=["Sleeper Berths", "Blankets", "Charging Points", "Live Tracking"]),
        TransportOption(id="trn-raj-003", destination_id="dest-rajasthan-004", vendor_id="vnd-raj-015",
                        type="self_drive", name="Thar Desert Rental 4x4",
                        route_from="Jaipur Hub",
                        route_to="Pushkar & Ajmer Circuit",
                        duration_hours=24.0, price=5500.0, currency="INR", capacity=4,
                        features=["4-Wheel Drive", "GPS Navigation", "Zero-Dep Insurance"]),
    ]
    return {Vendor: vendors, Hotel: hotels, Activity: activities, TransportOption: transports}


def _udaipur_catalog_rows():
    """Udaipur as its own catalog destination (users type the city name).

    Includes the Destination row itself so older databases gain it via the
    backfill; rows are PK-keyed and only inserted when absent.
    """
    vendors = [
        Vendor(id="vnd-uda-016", name="Lake City Royal Stays", vendor_type="hotel",
               contact_email="stay@lakecityroyal.in", phone="+91 29423 10016",
               rating=4.8, is_verified=True),
        Vendor(id="vnd-uda-017", name="Mewar Heritage Experiences", vendor_type="activity",
               contact_email="hello@mewarexp.in", phone="+91 98290 30017",
               rating=4.7, is_verified=True),
        Vendor(id="vnd-uda-018", name="Lake City Mobility", vendor_type="transport",
               contact_email="dispatch@lakecitymobility.in", phone="+91 29423 10018",
               rating=4.6, is_verified=True),
    ]
    hotels = [
        Hotel(id="htl-uda-001", destination_id="dest-udaipur-006", vendor_id="vnd-uda-016",
              name="Lake Pichola Palace Hotel", category="luxury",
              price_per_night=25000.0, currency="INR", rating=4.9,
              address="Lake Pichola, Udaipur 313001",
              amenities=["Lake-Facing Suites", "Sunset Terrace", "Royal Spa", "Private Ghat"],
              images=["https://images.unsplash.com/photo-1568495286058-9c3e0b8b0e0e?auto=format&fit=crop&w=800&q=80"],
              description="White-marble lakefront palace with balconies over Pichola's evening lights.",
              latitude=24.5720, longitude=73.6790),
        Hotel(id="htl-uda-002", destination_id="dest-udaipur-006", vendor_id="vnd-uda-016",
              name="Old City Lakeside Haveli", category="boutique",
              price_per_night=8500.0, currency="INR", rating=4.6,
              address="Gangaur Ghat, Udaipur 313001",
              amenities=["Rooftop Restaurant", "Miniature Art Gallery", "Ghat Steps Access"],
              images=["https://images.unsplash.com/photo-1520250497591-112f2f40a3f4?auto=format&fit=crop&w=800&q=80"],
              description="Family-run haveli on the ghats with rooftop dining over the lake.",
              latitude=24.5761, longitude=73.6836),
    ]
    activities = [
        Activity(id="act-uda-001", destination_id="dest-udaipur-006", vendor_id="vnd-uda-017",
                 title="City Palace Complex & Crystal Gallery Tour", category="culture",
                 duration_hours=3.5, price_per_person=1900.0, currency="INR",
                 difficulty_level="easy", rating=4.8, capacity=30,
                 images=["https://images.unsplash.com/photo-1568495286058-9c3e0b8b0e0e?auto=format&fit=crop&w=800&q=80"],
                 description="Mewar royal courtyards, armoury, and the famed crystal gallery with a historian.",
                 meeting_point="City Palace Main Gate", latitude=24.5761, longitude=73.6836),
        Activity(id="act-uda-002", destination_id="dest-udaipur-006", vendor_id="vnd-uda-017",
                 title="Lake Pichola Sunset Boat Ride", category="relaxation",
                 duration_hours=2.0, price_per_person=1200.0, currency="INR",
                 difficulty_level="easy", rating=4.8, capacity=40,
                 images=["https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=800&q=80"],
                 description="Evening boat past Jag Mandir island palace with Aravalli sunset views.",
                 meeting_point="City Palace Jetty", latitude=24.5720, longitude=73.6790),
        Activity(id="act-uda-003", destination_id="dest-udaipur-006", vendor_id="vnd-uda-017",
                 title="Sajjangarh Monsoon Palace Sunset Point", category="nature",
                 duration_hours=3.0, price_per_person=1100.0, currency="INR",
                 difficulty_level="easy", rating=4.6, capacity=25,
                 images=["https://images.unsplash.com/photo-1506905925346-21bda4d32df4?auto=format&fit=crop&w=800&q=80"],
                 description="Hilltop palace panorama over lakes and the wildlife sanctuary below.",
                 meeting_point="Sajjangarh Gate", latitude=24.5946, longitude=73.6378),
        Activity(id="act-uda-004", destination_id="dest-udaipur-006", vendor_id="vnd-uda-017",
                 title="Mewari Cooking Class with Haveli Lunch", category="culinary",
                 duration_hours=4.0, price_per_person=1500.0, currency="INR",
                 difficulty_level="easy", rating=4.7, capacity=12,
                 images=["https://images.unsplash.com/photo-1555396273-367ea4eb4db5?auto=format&fit=crop&w=800&q=80"],
                 description="Cook gatte-ki-sabzi and dal-baati with a home chef, then eat on the terrace.",
                 meeting_point="Old City Cooking Studio", latitude=24.5761, longitude=73.6836),
        Activity(id="act-uda-005", destination_id="dest-udaipur-006", vendor_id="vnd-uda-017",
                 title="Kumbhalgarh Fort & Wall Day Excursion", category="adventure",
                 duration_hours=7.0, price_per_person=2600.0, currency="INR",
                 difficulty_level="moderate", rating=4.7, capacity=20,
                 images=["https://images.unsplash.com/photo-1599661046289-e31897846e41?auto=format&fit=crop&w=800&q=80"],
                 description="UNESCO hill fort with the world's second-longest wall and valley viewpoints.",
                 meeting_point="Udaipur Tour Hub", latitude=25.1478, longitude=73.5851),
        Activity(id="act-uda-006", destination_id="dest-udaipur-006", vendor_id="vnd-uda-017",
                 title="Bagore Ki Haveli Folk Dance Show", category="culture",
                 duration_hours=2.0, price_per_person=900.0, currency="INR",
                 difficulty_level="easy", rating=4.7, capacity=60,
                 images=["https://images.unsplash.com/photo-1568495286058-9c3e0b8b0e0e?auto=format&fit=crop&w=800&q=80"],
                 description="Dharohar evening: Ghoomar, puppetry, and turban-tying on the lakeside courtyard.",
                 meeting_point="Bagore Haveli Gate", latitude=24.5770, longitude=73.6820),
        Activity(id="act-uda-007", destination_id="dest-udaipur-006", vendor_id="vnd-uda-017",
                 title="Fateh Sagar Morning Kayaking", category="adventure",
                 duration_hours=2.0, price_per_person=1300.0, currency="INR",
                 difficulty_level="easy", rating=4.5, capacity=12,
                 images=["https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=800&q=80"],
                 description="Calm-water paddle past Nehru Island with an instructor and sunrise views.",
                 meeting_point="Fateh Sagar Boathouse", latitude=24.5960, longitude=73.6760),
        Activity(id="act-uda-008", destination_id="dest-udaipur-006", vendor_id="vnd-uda-017",
                 title="Sahelion Ki Bari Garden Morning", category="nature",
                 duration_hours=2.0, price_per_person=700.0, currency="INR",
                 difficulty_level="easy", rating=4.5, capacity=25,
                 images=["https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=800&q=80"],
                 description="Fountains, lotus pools, and marble elephants in the royal ladies' garden.",
                 meeting_point="Sahelion Gate", latitude=24.5900, longitude=73.6900),
    ]
    transports = [
        TransportOption(id="trn-uda-001", destination_id="dest-udaipur-006", vendor_id="vnd-uda-018",
                        type="private_cab", name="Toyota Innova Lake City Cab",
                        route_from="Maharana Pratap Airport",
                        route_to="Old City & Lake Hotels",
                        duration_hours=1.0, price=2400.0, currency="INR", capacity=6,
                        features=["AC", "Flight Tracking", "Luggage Carrier", "Local Guide"]),
        TransportOption(id="trn-uda-002", destination_id="dest-udaipur-006", vendor_id="vnd-uda-018",
                        type="volvo_bus", name="Intercity AC Sleeper Coach",
                        route_from="Ahmedabad Geeta Mandir",
                        route_to="Udaipur Central Bus Stand",
                        duration_hours=5.0, price=1000.0, currency="INR", capacity=32,
                        features=["Sleeper Berths", "Blankets", "Charging Points", "Live Tracking"]),
        TransportOption(id="trn-uda-003", destination_id="dest-udaipur-006", vendor_id="vnd-uda-018",
                        type="self_drive", name="XUV City Rental SUV",
                        route_from="Udaipur Hub",
                        route_to="Kumbhalgarh & Ranakpur Circuit",
                        duration_hours=24.0, price=5000.0, currency="INR", capacity=5,
                        features=["GPS Navigation", "Zero-Dep Insurance", "Extra Driver Free"]),
    ]
    destination = [
        Destination(id="dest-udaipur-006", name="Udaipur", slug="udaipur",
                    country="India", state_region="Rajasthan",
                    description="Venice of the East: mirror-calm lakes, white-marble palaces, and romantic old-city lanes in the Mewar heartland.",
                    hero_image_url="https://images.unsplash.com/photo-1568495286058-9c3e0b8b0e0e?auto=format&fit=crop&w=1400&q=80",
                    best_time_to_visit="October to March",
                    tags=["lakes", "palaces", "romance", "heritage", "sunsets"],
                    is_featured=True, latitude=24.5854, longitude=73.7125),
    ]
    return {Destination: destination, Vendor: vendors, Hotel: hotels,
            Activity: activities, TransportOption: transports}


def _manali_extra_rows():
    """3 more Manali activities (8 total) for fuller multi-day itineraries."""
    return {Activity: [
        Activity(id="act-manali-006", destination_id="dest-manali-001", vendor_id="vnd-adv-002",
                 title="Hadimba Cedar Temple & Forest Nature Trail", category="nature",
                 duration_hours=2.0, price_per_person=900.0, currency="INR",
                 difficulty_level="easy", rating=4.7, capacity=30,
                 images=["https://images.unsplash.com/photo-1432821596592-e2c18b78144f?auto=format&fit=crop&w=800&q=80"],
                 description="Wooden pagoda temple amid ancient deodars with a guided nature loop and yak rides nearby.",
                 meeting_point="Hadimba Temple Gate",
                 latitude=32.2394, longitude=77.1891),
        Activity(id="act-manali-007", destination_id="dest-manali-001", vendor_id="vnd-adv-002",
                 title="Mall Road Street Food Crawl with Local Guide", category="culinary",
                 duration_hours=2.5, price_per_person=1100.0, currency="INR",
                 difficulty_level="easy", rating=4.6, capacity=25,
                 images=["https://images.unsplash.com/photo-1554118811-1e0d58224f24?auto=format&fit=crop&w=800&q=80"],
                 description="Siddu, trout tikkas, and Old Manali café hopping with food stories.",
                 meeting_point="Mall Road Entry Gate",
                 latitude=32.2396, longitude=77.1887),
        Activity(id="act-manali-008", destination_id="dest-manali-001", vendor_id="vnd-adv-002",
                 title="Hampta Valley Day Hike with Pack Lunch", category="adventure",
                 duration_hours=7.0, price_per_person=3200.0, currency="INR",
                 difficulty_level="moderate", rating=4.8, capacity=14,
                 images=["https://images.unsplash.com/photo-1506012787146-f92b2d7d6d96?auto=format&fit=crop&w=800&q=80"],
                 description="Alpine meadows, shepherd camps, and river crossings on the classic Hampta trail.",
                 meeting_point="Jobra Trailhead",
                 latitude=32.2657, longitude=77.2989),
    ]}


def _all_catalog_builders():
    return [_manali_extra_rows, _goa_catalog_rows, _kashmir_catalog_rows, _kerala_catalog_rows,
            _rajasthan_catalog_rows, _udaipur_catalog_rows]


# Real coordinates for original catalog rows that shipped without them
# (needed so every itinerary stop plots on the trip map). Only fills
# blanks — never overwrites existing values.
_CATALOG_COORDINATES = {
    ("Hotel", "htl-manali-001"): (32.2390, 77.1910),
    ("Hotel", "htl-manali-002"): (32.2205, 77.1895),
    ("Hotel", "htl-manali-003"): (32.1820, 77.1980),
    ("Hotel", "htl-manali-004"): (32.2458, 77.1805),
    ("Hotel", "htl-goa-001"): (15.2589, 73.9235),
    ("Activity", "act-manali-001"): (32.0283, 77.1667),
    ("Activity", "act-manali-002"): (32.3710, 77.2400),
    ("Activity", "act-manali-003"): (32.2589, 77.1720),
    ("Activity", "act-manali-004"): (32.2458, 77.1805),
    ("Activity", "act-manali-005"): (32.1030, 77.1470),
    ("Activity", "act-manali-006"): (32.2394, 77.1891),
    ("Activity", "act-manali-007"): (32.2396, 77.1887),
    ("Activity", "act-manali-008"): (32.2657, 77.2989),
    ("TransportOption", "trn-manali-001"): (32.2396, 77.1887),
    ("TransportOption", "trn-manali-002"): (32.2396, 77.1887),
    ("TransportOption", "trn-manali-003"): (32.2396, 77.1887),
    # Transport rows are routes: anchor at the destination town so trip
    # maps can draw them. One entry per seeded transport row.
    ("TransportOption", "trn-goa-001"): (15.4909, 73.8278),
    ("TransportOption", "trn-goa-002"): (15.4909, 73.8278),
    ("TransportOption", "trn-goa-003"): (15.4909, 73.8278),
    ("TransportOption", "trn-kas-001"): (34.0837, 74.7973),
    ("TransportOption", "trn-kas-002"): (34.0837, 74.7973),
    ("TransportOption", "trn-kas-003"): (34.0837, 74.7973),
    ("TransportOption", "trn-ker-001"): (9.9312, 76.2673),
    ("TransportOption", "trn-ker-002"): (9.9312, 76.2673),
    ("TransportOption", "trn-ker-003"): (9.9312, 76.2673),
    ("TransportOption", "trn-raj-001"): (26.9124, 75.7873),
    ("TransportOption", "trn-raj-002"): (26.9124, 75.7873),
    ("TransportOption", "trn-raj-003"): (26.9124, 75.7873),
    ("TransportOption", "trn-uda-001"): (24.5854, 73.7125),
    ("TransportOption", "trn-uda-002"): (24.5854, 73.7125),
    ("TransportOption", "trn-uda-003"): (24.5854, 73.7125),
}


def ensure_catalog_coordinates(db) -> int:
    """Fill blank lat/long on catalog rows. Idempotent; returns count fixed."""
    from backend.models.models import Activity, Hotel, TransportOption
    models = {"Hotel": Hotel, "Activity": Activity, "TransportOption": TransportOption}
    fixed = 0
    for (model_name, row_id), (lat, lng) in _CATALOG_COORDINATES.items():
        row = db.query(models[model_name]).filter(models[model_name].id == row_id).first()
        if row is not None and (row.latitude is None or row.longitude is None):
            row.latitude, row.longitude = lat, lng
            fixed += 1
    if fixed:
        logger.info(f"Catalog coordinates: filled {fixed} blank rows.")
    return fixed


def ensure_catalog_backfill(db: Session) -> None:
    """Idempotent backfill: insert catalog rows missing from older databases.

    Fresh databases get these rows in the main seed flow below; existing
    databases (seeded before this inventory existed) gain them here on next
    startup. Rows are keyed by stable PKs and only inserted when absent.
    """
    inserted = 0
    for builder in _all_catalog_builders():
        for model, rows in builder().items():
            for row in rows:
                if db.query(model).filter(model.id == row.id).first() is None:
                    db.add(row)
                    inserted += 1
    if inserted:
        logger.info(f"Catalog backfill: inserted {inserted} missing catalog inventory rows.")


def run_seed():
    """Deterministic Database Seeding for TourFlow AI Foundation."""
    # Ensure tables exist
    Base.metadata.create_all(bind=engine)
    db: Session = SessionLocal()

    try:
        # Check if already seeded
        if db.query(Destination).filter(Destination.slug == "manali").first():
            logger.info("Database already seeded with foundation data.")
            ensure_catalog_backfill(db)
            ensure_catalog_coordinates(db)
            db.commit()
            return

        logger.info("Starting deterministic seed data injection...")

        # ----------------------------------------------------
        # 1. Primary Users & Traveler Profiles
        # ----------------------------------------------------
        primary_user = User(
            id="usr-alex-morgan-001",
            email="alex.morgan@tourflow.ai",
            full_name="Alex Morgan",
            phone="+91 98765 43210",
            role="traveler",
            is_active=True
        )
        db.add(primary_user)
        db.flush()

        profile = TravelerProfile(
            id="prof-alex-001",
            user_id=primary_user.id,
            travel_style="adventurous_luxury",
            dietary_preferences=["vegetarian", "artisan_coffee"],
            fitness_level="high",
            preferred_currency="INR",
            language="English",
            bio="Alpine enthusiast, outdoor photographer, and culture seeker."
        )
        db.add(profile)

        operator_user = User(
            id="usr-rahul-sharma-002",
            email="rahul.operator@tourflow.ai",
            full_name="Rahul Sharma",
            phone="+91 98111 22334",
            role="operator",
            is_active=True
        )
        db.add(operator_user)

        # ----------------------------------------------------
        # 2. Vendors
        # ----------------------------------------------------
        v_himalayan = Vendor(
            id="vnd-him-001",
            name="Himalayan Heritage Resorts & Expeditions",
            vendor_type="hotel",
            contact_email="concierge@thehimalayan.com",
            phone="+91 1902 250123",
            rating=4.9,
            is_verified=True
        )
        v_adventure = Vendor(
            id="vnd-adv-002",
            name="Pir Panjal High Altitude Adventures",
            vendor_type="activity",
            contact_email="fly@pirpanjaladventure.com",
            phone="+91 98160 55443",
            rating=4.8,
            is_verified=True
        )
        v_trans = Vendor(
            id="vnd-tra-003",
            name="North Bound Luxury Alpine Mobility",
            vendor_type="transport",
            contact_email="dispatch@northboundmobility.com",
            phone="+91 98050 11223",
            rating=4.9,
            is_verified=True
        )
        db.add_all([v_himalayan, v_adventure, v_trans])
        db.flush()

        # ----------------------------------------------------
        # 3. Destinations
        # ----------------------------------------------------
        d_manali = Destination(
            id="dest-manali-001",
            name="Manali",
            slug="manali",
            country="India",
            state_region="Himachal Pradesh",
            description="Perched at 2,050m in the majestic Kullu Valley, Manali combines snow-clad Pir Panjal peaks, aromatic pine forests, thrilling alpine passes, and vibrant bohemian café culture.",
            hero_image_url="https://images.unsplash.com/photo-1626621341517-bbf3d9990a23?auto=format&fit=crop&w=1400&q=80",
            best_time_to_visit="October to June (Snow: Dec-Feb)",
            tags=["mountains", "snow", "adventure", "rivers", "cafes", "trekking"],
            is_featured=True,
            latitude=32.2396,
            longitude=77.1887
        )

        d_goa = Destination(
            id="dest-goa-002",
            name="Goa",
            slug="goa",
            country="India",
            state_region="Goa",
            description="Sun-drenched tropical coastline famed for golden sand beaches, Portuguese colonial architecture, bohemian flea markets, and world-class seafood cuisine.",
            hero_image_url="https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?auto=format&fit=crop&w=1400&q=80",
            best_time_to_visit="November to March",
            tags=["beaches", "sunsets", "nightlife", "heritage", "water_sports"],
            is_featured=True,
            latitude=15.2993,
            longitude=74.1240
        )

        d_kerala = Destination(
            id="dest-kerala-003",
            name="Kerala",
            slug="kerala",
            country="India",
            state_region="Kerala",
            description="God's Own Country, featuring serene palm-fringed backwaters, emerald tea estates in Munnar, Ayurvedic rejuvenation sanctuaries, and spice-laden coastal breezes.",
            hero_image_url="https://images.unsplash.com/photo-1602216056096-3b40cc0c9944?auto=format&fit=crop&w=1400&q=80",
            best_time_to_visit="September to March",
            tags=["backwaters", "tea_gardens", "ayurveda", "nature", "houseboats"],
            is_featured=True,
            latitude=10.8505,
            longitude=76.2711
        )

        d_rajasthan = Destination(
            id="dest-rajasthan-004",
            name="Rajasthan",
            slug="rajasthan",
            country="India",
            state_region="Rajasthan",
            description="Land of royal kings, golden Thar sand dunes, monumental hilltop fortresses, and opulent heritage palace hotels in Jaipur, Udaipur, and Jaisalmer.",
            hero_image_url="https://images.unsplash.com/photo-1599661046289-e31897846e41?auto=format&fit=crop&w=1400&q=80",
            best_time_to_visit="October to March",
            tags=["palaces", "desert", "forts", "royal_heritage", "culture"],
            is_featured=True,
            latitude=27.0238,
            longitude=74.2179
        )

        d_kashmir = Destination(
            id="dest-kashmir-005",
            name="Kashmir",
            slug="kashmir",
            country="India",
            state_region="Jammu & Kashmir",
            description="The Crown Jewel of the Himalayas, offering tranquil Dal Lake shikara rides, Gulmarg powder-snow skiing, and the wildflower meadows of Pahalgam.",
            hero_image_url="https://images.unsplash.com/photo-1595815771614-ade9d652a65d?auto=format&fit=crop&w=1400&q=80",
            best_time_to_visit="March to October (Snow: Dec-Feb)",
            tags=["dal_lake", "snow_skiing", "valleys", "houseboats", "nature"],
            is_featured=True,
            latitude=34.0837,
            longitude=74.7973
        )

        db.add_all([d_manali, d_goa, d_kerala, d_rajasthan, d_kashmir])
        db.flush()

        # ----------------------------------------------------
        # 4. Realistic Hotels for Manali & Others
        # ----------------------------------------------------
        h_manali_1 = Hotel(
            id="htl-manali-001",
            destination_id=d_manali.id,
            vendor_id=v_himalayan.id,
            name="The Himalayan Luxury Boutique Resort & Castle",
            category="luxury",
            price_per_night=18500.0,
            currency="INR",
            rating=4.9,
            address="Hadimba Road, Manali, Himachal Pradesh 175131",
            amenities=["Victorian Castle Architecture", "Heated Swimming Pool", "Panoramic Mountain Views", "Gourmet Fine Dining", "Fireplace Lounges", "Spa"],
            images=["https://images.unsplash.com/photo-1582719478250-c89cae4dc85b?auto=format&fit=crop&w=800&q=80"],
            description="A premier 19th-century Victorian Gothic castle stay set amongst apple and cherry orchards with dramatic views of snow peaks."
        )

        h_manali_2 = Hotel(
            id="htl-manali-002",
            destination_id=d_manali.id,
            vendor_id=v_himalayan.id,
            name="Larisa Mountain Resort & Organic Apple Orchard",
            category="boutique",
            price_per_night=12000.0,
            currency="INR",
            rating=4.8,
            address="Haripur, Manali, Himachal Pradesh 175136",
            amenities=["Private Stone Cottages", "Organic Farm-to-Table Dining", "Jacuzzi", "Bonfire Patios", "Forest Trails"],
            images=["https://images.unsplash.com/photo-1566073771259-6a8506099945?auto=format&fit=crop&w=800&q=80"],
            description="Exquisite stone and wood cottages nestled in fragrant apple orchards with bespoke butler service and starlit bonfire evenings."
        )

        h_manali_3 = Hotel(
            id="htl-manali-003",
            destination_id=d_manali.id,
            vendor_id=v_himalayan.id,
            name="Span Resort & Alpine Spa (Riverside)",
            category="luxury",
            price_per_night=16000.0,
            currency="INR",
            rating=4.7,
            address="Baragran Bihal, Manali Highway, HP 175129",
            amenities=["Direct Riverfront Access", "Helipad Access", "Ayurvedic Wellness Spa", "Fly Fishing", "Tennis Court"],
            images=["https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?auto=format&fit=crop&w=800&q=80"],
            description="Located directly along the banks of the rushing Beas River with private pine woods and bespoke riverside breakfast pavilions."
        )

        h_manali_4 = Hotel(
            id="htl-manali-004",
            destination_id=d_manali.id,
            vendor_id=v_himalayan.id,
            name="Zostel Plus Old Manali (Boutique Social Stay)",
            category="mid-range",
            price_per_night=3800.0,
            currency="INR",
            rating=4.6,
            address="Manu Temple Road, Old Manali 175131",
            amenities=["High-Speed Fiber WiFi", "Co-working Lounge", "Café Deck", "Mountain View Pods", "Acoustic Nights"],
            images=["https://images.unsplash.com/photo-1520250497591-112f2f40a3f4?auto=format&fit=crop&w=800&q=80"],
            description="Vibrant design-led social stay overlooking Old Manali valley, beloved by creators, digital nomads, and young explorers."
        )

        # Goa Hotel
        h_goa_1 = Hotel(
            id="htl-goa-001",
            destination_id=d_goa.id,
            name="Taj Exotica Resort & Spa (Benaulim)",
            category="luxury",
            price_per_night=24000.0,
            currency="INR",
            rating=4.9,
            address="Calwaddo, Benaulim, Goa 403716",
            amenities=["Private Beach Front", "Golf Course", "Jiva Ayurvedic Spa", "Infinity Pool"],
            images=["https://images.unsplash.com/photo-1571896349842-33c89424de2d?auto=format&fit=crop&w=800&q=80"],
            description="Mediterranean-style 56-acre coastal sanctuary overlooking the pristine Arabian Sea."
        )

        db.add_all([h_manali_1, h_manali_2, h_manali_3, h_manali_4, h_goa_1])
        db.flush()

        # ----------------------------------------------------
        # 5. Realistic Activities for Manali & Others
        # ----------------------------------------------------
        act_manali_1 = Activity(
            id="act-manali-001",
            destination_id=d_manali.id,
            vendor_id=v_adventure.id,
            title="Solang Valley High Altitude Paragliding & ATV Expedition",
            category="adventure",
            duration_hours=3.5,
            price_per_person=3500.0,
            currency="INR",
            difficulty_level="moderate",
            rating=4.9,
            capacity=12,
            images=["https://images.unsplash.com/photo-1506012787146-f92b2d7d6d96?auto=format&fit=crop&w=800&q=80"],
            description="Tandem flight over cedar canopies with certified instructors, followed by a rugged quad bike excursion along alpine streams.",
            meeting_point="Solang Adventure Base Camp"
        )

        act_manali_2 = Activity(
            id="act-manali-002",
            destination_id=d_manali.id,
            vendor_id=v_adventure.id,
            title="Rohtang Pass & Atal Tunnel Snow Glacier Tour",
            category="nature",
            duration_hours=6.0,
            price_per_person=4200.0,
            currency="INR",
            difficulty_level="moderate",
            rating=4.8,
            capacity=30,
            images=["https://images.unsplash.com/photo-1517824806704-9040b037703b?auto=format&fit=crop&w=800&q=80"],
            description="Journey through the engineering marvel of Atal Tunnel into Lahaul Valley and ascend to Rohtang Pass (3,978m) for pristine snow landscapes.",
            meeting_point="TourFlow Private Lounge, Manali Mall Road"
        )

        act_manali_3 = Activity(
            id="act-manali-003",
            destination_id=d_manali.id,
            vendor_id=v_adventure.id,
            title="Jogini Waterfalls & Vashisht Natural Sulphur Springs Trek",
            category="adventure",
            duration_hours=3.0,
            price_per_person=1500.0,
            currency="INR",
            difficulty_level="easy",
            rating=4.7,
            capacity=20,
            images=["https://images.unsplash.com/photo-1432821596592-e2c18b78144f?auto=format&fit=crop&w=800&q=80"],
            description="Tranquil guided forest trail passing ancient apple orchards to the cascading Jogini falls, culminating in ancient hot spring baths.",
            meeting_point="Vashisht Temple Square"
        )

        act_manali_4 = Activity(
            id="act-manali-004",
            destination_id=d_manali.id,
            vendor_id=v_himalayan.id,
            title="Old Manali Bohemian Artisan Café & Woodcarving Trail",
            category="culture",
            duration_hours=2.5,
            price_per_person=1200.0,
            currency="INR",
            difficulty_level="easy",
            rating=4.8,
            capacity=25,
            images=["https://images.unsplash.com/photo-1554118811-1e0d58224f24?auto=format&fit=crop&w=800&q=80"],
            description="Curated insider walk discovering woodcraft master studios, hidden rooftop cider cafés, and local live acoustic folklore.",
            meeting_point="Old Manali Bridge Gate"
        )

        act_manali_5 = Activity(
            id="act-manali-005",
            destination_id=d_manali.id,
            vendor_id=v_adventure.id,
            title="Beas River Grade-IV White Water Rafting & Zipline",
            category="adventure",
            duration_hours=2.5,
            price_per_person=2800.0,
            currency="INR",
            difficulty_level="challenging",
            rating=4.8,
            capacity=16,
            images=["https://images.unsplash.com/photo-1530866495561-507c9faab2ed?auto=format&fit=crop&w=800&q=80"],
            description="Adrenaline-pumping 14km rafting course through glacier-fed rapids in the Kullu-Manali stretch with safety kayaks.",
            meeting_point="Pirdi Rafting Point"
        )

        db.add_all([act_manali_1, act_manali_2, act_manali_3, act_manali_4, act_manali_5])
        db.flush()

        # ----------------------------------------------------
        # 6. Realistic Transport Options for Manali
        # ----------------------------------------------------
        tr_manali_1 = TransportOption(
            id="trn-manali-001",
            destination_id=d_manali.id,
            vendor_id=v_trans.id,
            type="private_cab",
            name="Chauffeur-Driven Toyota Fortuner 4x4 Alpine SUV",
            route_from="Chandigarh / Bhuntar Airport",
            route_to="Manali Valley Resorts",
            duration_hours=5.5,
            price=12500.0,
            currency="INR",
            capacity=4,
            features=["Heated Leather Seats", "Alpine Snow Chains", "WiFi Hotspot", "Chilled Spring Water", "Roof Carrier"]
        )

        tr_manali_2 = TransportOption(
            id="trn-manali-002",
            destination_id=d_manali.id,
            vendor_id=v_trans.id,
            type="volvo_bus",
            name="Mercedes-Benz Multi-Axle Luxury AC Sleeper",
            route_from="Delhi Kashmere Gate ISBT",
            route_to="Manali Private Bus Terminal",
            duration_hours=12.0,
            price=1950.0,
            currency="INR",
            capacity=32,
            features=["Full Flat Sleeper Pods", "Individual Entertainment Screens", "Air Suspension", "Blankets & Water"]
        )

        tr_manali_3 = TransportOption(
            id="trn-manali-003",
            destination_id=d_manali.id,
            vendor_id=v_trans.id,
            type="self_drive",
            name="Mahindra Thar 4x4 Hardtop Adventure Rental",
            route_from="Manali Town Hub",
            route_to="Solang & Sissu Valley Exploration",
            duration_hours=24.0,
            price=5500.0,
            currency="INR",
            capacity=4,
            features=["4-Wheel Drive Low Range", "All-Terrain Tyres", "GPS Navigation", "Zero-Dep Insurance"]
        )

        db.add_all([tr_manali_1, tr_manali_2, tr_manali_3])
        db.flush()

        # ----------------------------------------------------
        # 6b. Catalog Inventory for Goa, Kashmir, Kerala, Rajasthan,
        # Udaipur (activities + transport + stays so every place the
        # user picks validates and generates like Manali trips)
        # ----------------------------------------------------
        for _builder in _all_catalog_builders():
            for _model, _rows in _builder().items():
                db.add_all(_rows)
        db.flush()
        ensure_catalog_coordinates(db)

        # ----------------------------------------------------
        # 7. Central Entity Demo: Seed Trip with Complete Sub-Entities
        # ----------------------------------------------------
        demo_trip = Trip(
            id="trp-manali-alpine-demo-001",
            user_id=primary_user.id,
            destination_id=d_manali.id,
            title="Winter Escape: Curated Manali Alpine Explorer",
            status="confirmed",
            start_date=datetime.utcnow() + timedelta(days=14),
            end_date=datetime.utcnow() + timedelta(days=18),
            duration_days=4,
            total_budget=75000.0,
            currency="INR",
            traveler_count=2,
            pace="balanced"
        )
        db.add(demo_trip)
        db.flush()

        # Preferences
        demo_pref = TripPreference(
            id="pref-manali-001",
            trip_id=demo_trip.id,
            budget_tier="luxury",
            interests=["snow", "mountains", "paragliding", "artisan_cafes", "heritage"],
            travel_companions="couple",
            accommodation_types=["boutique", "mountain_view_resort"],
            transport_preferences=["private_suv", "chauffeur"],
            dietary_requirements=["vegetarian"],
            special_requests="High floor mountain-facing room with fireplace access and private photography stops."
        )
        db.add(demo_pref)

        # Itinerary Items (Day 1 to 4)
        iti_1 = ItineraryItem(
            id="iti-001",
            trip_id=demo_trip.id,
            day_number=1,
            order_index=1,
            item_type="transport",
            title="Private Luxury Fortuner SUV Transfer from Chandigarh",
            description="Scenic mountain highway transfer with riverside tea stop at Pandoh Dam.",
            start_time="08:00 AM",
            end_time="01:30 PM",
            cost=12500.0,
            status="confirmed",
            transport_id=tr_manali_1.id,
            location="Chandigarh to Manali"
        )
        iti_2 = ItineraryItem(
            id="iti-002",
            trip_id=demo_trip.id,
            day_number=1,
            order_index=2,
            item_type="hotel",
            title="Check-in at The Himalayan Luxury Boutique Castle",
            description="Welcome Himalayan herbal tea, luggage unpacking, and mountain terrace relaxation.",
            start_time="02:00 PM",
            end_time="04:00 PM",
            cost=18500.0,
            status="confirmed",
            hotel_id=h_manali_1.id,
            location=h_manali_1.address
        )
        iti_3 = ItineraryItem(
            id="iti-003",
            trip_id=demo_trip.id,
            day_number=1,
            order_index=3,
            item_type="activity",
            title="Old Manali Bohemian Artisan Café Walk",
            description="Evening stroll exploring woodcraft workshops, local cider, and acoustic violin performance.",
            start_time="05:00 PM",
            end_time="07:30 PM",
            cost=2400.0,
            status="confirmed",
            activity_id=act_manali_4.id,
            location="Old Manali"
        )
        iti_4 = ItineraryItem(
            id="iti-004",
            trip_id=demo_trip.id,
            day_number=2,
            order_index=1,
            item_type="activity",
            title="Solang Valley High Altitude Paragliding & ATV Expedition",
            description="Morning tandem paragliding glide over snow pine trees followed by quad trail ride.",
            start_time="09:00 AM",
            end_time="01:00 PM",
            cost=7000.0,
            status="confirmed",
            activity_id=act_manali_1.id,
            location="Solang Valley"
        )
        iti_5 = ItineraryItem(
            id="iti-005",
            trip_id=demo_trip.id,
            day_number=3,
            order_index=1,
            item_type="activity",
            title="Rohtang Pass & Atal Tunnel Snow Expedition",
            description="Full day panoramic drive through Atal Tunnel to North Portal and Rohtang glaciers.",
            start_time="08:30 AM",
            end_time="03:30 PM",
            cost=8400.0,
            status="proposed",
            activity_id=act_manali_2.id,
            location="Rohtang Pass"
        )
        db.add_all([iti_1, iti_2, iti_3, iti_4, iti_5])

        # Booking
        demo_booking = Booking(
            id="bkg-001",
            trip_id=demo_trip.id,
            vendor_id=v_himalayan.id,
            booking_reference="TF-MANALI-7782",
            item_type="hotel",
            item_id=h_manali_1.id,
            amount=37000.0,
            currency="INR",
            status="confirmed",
            payment_status="paid"
        )
        db.add(demo_booking)

        # Alert
        demo_alert = Alert(
            id="alt-001",
            trip_id=demo_trip.id,
            alert_type="weather",
            severity="info",
            title="Fresh Snowfall Forecast at Rohtang Pass",
            description="Expect 8-12 inches of fresh powder snow at Rohtang Pass on Day 3. 4x4 SUV equipped with tire chains is pre-arranged.",
            is_resolved=False
        )
        db.add(demo_alert)

        # Notification
        demo_notif = Notification(
            id="notif-001",
            trip_id=demo_trip.id,
            user_id=primary_user.id,
            title="Your Alpine Itinerary is Ready",
            message="TourFlow AI has optimized your 4-day Manali winter escape with private transfers and top-rated stays.",
            type="success",
            is_read=False
        )
        db.add(demo_notif)

        # Change History
        ch_1 = ChangeHistory(
            id="chg-001",
            trip_id=demo_trip.id,
            changed_by="ai",
            action="itinerary_optimized",
            field_changed="day_2_timing",
            old_value="11:00 AM",
            new_value="09:00 AM",
            reason="Shifted paragliding to early morning for optimal alpine thermal wind conditions"
        )
        ch_2 = ChangeHistory(
            id="chg-002",
            trip_id=demo_trip.id,
            changed_by="user",
            action="room_upgrade",
            field_changed="hotel_category",
            old_value="boutique",
            new_value="luxury_castle",
            reason="Traveler upgraded to The Himalayan Victorian Castle"
        )
        db.add_all([ch_1, ch_2])

        # Review
        demo_review = Review(
            id="rev-001",
            trip_id=demo_trip.id,
            user_id=primary_user.id,
            rating=5.0,
            title="Flawless Alpine Planning Experience",
            comment="TourFlow AI dynamic recommendations and live disruption alerts made our mountain trip completely seamless.",
            destination_rating=5.0,
            ai_planning_rating=5.0
        )
        db.add(demo_review)

        # Dispatch fleet inventory
        if not db.query(Vehicle).first():
            db.add_all([
                Vehicle(id="veh-001", name="Mahindra Thar 4x4", registration_number="HP01-TRAN-1001", vehicle_type="private_cab", capacity=4, is_active=True),
                Vehicle(id="veh-002", name="Toyota Innova Crysta", registration_number="HP01-TRAN-1002", vehicle_type="private_cab", capacity=6, is_active=True),
                Vehicle(id="veh-003", name="Volvo 9400 Coach", registration_number="HP01-TRAN-2001", vehicle_type="volvo_bus", capacity=40, is_active=True),
            ])
        if not db.query(Driver).first():
            db.add_all([
                Driver(id="drv-001", name="Tenzin Norbu", phone="+91 98160 10001", license_number="HP-DL-20180001", is_active=True),
                Driver(id="drv-002", name="Amit Thakur", phone="+91 98160 10002", license_number="HP-DL-20190002", is_active=True),
                Driver(id="drv-003", name="Rajesh Kumar", phone="+91 98160 10003", license_number="HP-DL-20200003", is_active=True),
            ])

        db.commit()
        logger.info("Deterministic database seed completed successfully!")

    except Exception as e:
        db.rollback()
        logger.error(f"Error during seeding: {e}")
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    run_seed()
