"""SceneSeen commercial taxonomy (what is worth finding, not everything that can be found).

Two kinds of concepts, deliberately separate:

* OBJECTS      things with a location in the frame (phone, watch, sofa). Found by an
               open-vocabulary object detector from the `prompts` below.
* CONTEXTS     properties of a whole scene (restaurant, office, indoor). These have no bounding
               box; they are classified from the CLIP embeddings Phase 1 already caches.

Nothing here names brands or products. "sneakers" is a category of thing; matching it to a
specific product is Phase 2B (product catalogue), which consumes these detections.

`base_relevance` is the commercial value of the object type on a 0-1 scale (rule-based, V1):
how likely a brand would pay for, or a viewer would shop for, this kind of thing.
Bump TAXONOMY_VERSION whenever labels, prompts or relevance values change.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

TAXONOMY_VERSION = "2026.10-4"


@dataclass(frozen=True)
class Category:
    id: str
    name: str
    icon: str


CATEGORIES: dict[str, Category] = {c.id: c for c in [
    Category("fashion", "Fashion", "👕"),
    Category("accessories", "Accessories", "⌚"),
    Category("electronics", "Electronics", "📱"),
    Category("automotive", "Automotive", "🚗"),
    Category("food_beverage", "Food & Beverage", "🥤"),
    Category("furniture", "Furniture & Home", "🛋️"),
    Category("beauty", "Beauty & Personal Care", "💄"),
    Category("locations", "Locations & Venues", "📍"),
    Category("real_estate", "Real Estate & Environment", "🏠"),
]}


@dataclass(frozen=True)
class ObjectType:
    id: str
    label: str                      # shown to users
    category: str
    prompts: tuple[str, ...]        # detector queries (first = canonical)
    base_relevance: float
    icon: str = ""
    min_confidence: float | None = None   # display threshold for this type (see the tiers below)
    group: str = ""                 # types in the same group compete for one box (sneakers vs shoes)
    describe_color: bool = False    # "white sneakers": only where colour is a meaningful attribute


# Display thresholds on detector confidence, in tiers. Calibrated on a blind review of 177
# detections from 4 videos (docs/COMMERCIAL.md): precision was 15 % at 0.2-0.3, 39 % at 0.3-0.4,
# 75 % at 0.4-0.6 and 94 % above 0.6, and it depends strongly on the kind of object.
TIER_LARGE = 0.40      # large, distinctive things (clothing, furniture, vehicles): reliable early
TIER_DEFAULT = 0.50    # hand-held / medium objects
TIER_SMALL = 0.55      # small accessories (watch, jewelry): often confused below this
TIER_RARE = 0.65       # types that were almost never right in review (camera, headphones, cosmetics ...)


def _o(id, label, category, prompts, rel, icon="", tier=TIER_DEFAULT, **kw) -> ObjectType:
    return ObjectType(id, label, category, tuple(prompts), rel, icon, min_confidence=tier, **kw)


OBJECTS: dict[str, ObjectType] = {o.id: o for o in [
    # ---- Fashion
    _o("shirt", "Shirt", "fashion", ["shirt"], 0.70, "👔", group="top", describe_color=True, tier=TIER_LARGE),
    _o("tshirt", "T-shirt", "fashion", ["t-shirt"], 0.70, "👕", group="top", describe_color=True, tier=TIER_LARGE),
    _o("jacket", "Jacket", "fashion", ["jacket"], 0.80, "🧥", group="top", describe_color=True, tier=TIER_LARGE),
    _o("suit", "Suit", "fashion", ["suit jacket"], 0.80, "🤵", group="top", describe_color=True, tier=TIER_LARGE),
    _o("dress", "Dress", "fashion", ["dress"], 0.85, "👗", group="top", describe_color=True, tier=TIER_LARGE),
    _o("trousers", "Trousers", "fashion", ["trousers"], 0.65, "👖", describe_color=True, tier=TIER_LARGE),
    _o("sneakers", "Sneakers", "fashion", ["sneakers"], 0.90, "👟", group="footwear", describe_color=True),
    _o("shoes", "Shoes", "fashion", ["shoes"], 0.80, "👞", group="footwear", describe_color=True),
    _o("hat", "Hat / cap", "fashion", ["hat", "baseball cap"], 0.70, "🧢", describe_color=True, tier=TIER_LARGE),
    # ---- Accessories
    _o("watch", "Watch", "accessories", ["wristwatch"], 0.95, "⌚", tier=TIER_SMALL),
    _o("sunglasses", "Sunglasses", "accessories", ["sunglasses"], 0.90, "🕶️", group="eyewear"),
    _o("eyeglasses", "Eyeglasses", "accessories", ["eyeglasses"], 0.70, "👓", group="eyewear"),
    _o("handbag", "Handbag", "accessories", ["handbag"], 0.95, "👜", group="bag", describe_color=True),
    _o("backpack", "Backpack", "accessories", ["backpack"], 0.80, "🎒", group="bag", describe_color=True),
    _o("jewelry", "Jewelry", "accessories", ["necklace", "earrings", "bracelet"], 0.85, "💍", tier=TIER_SMALL),
    # ---- Electronics
    _o("smartphone", "Smartphone", "electronics", ["smartphone", "mobile phone"], 0.95, "📱", group="screen_small"),
    _o("laptop", "Laptop", "electronics", ["laptop computer"], 0.90, "💻", group="screen_mid"),
    _o("tablet", "Tablet", "electronics", ["tablet computer"], 0.85, "📲", group="screen_small", tier=TIER_RARE),
    _o("headphones", "Headphones", "electronics", ["headphones"], 0.90, "🎧", tier=TIER_RARE),
    _o("television", "Television", "electronics", ["television"], 0.85, "📺", group="screen_mid", tier=TIER_RARE),
    _o("camera", "Camera", "electronics", ["camera"], 0.80, "📷", tier=TIER_RARE),
    # ---- Automotive
    _o("car", "Car", "automotive", ["car"], 0.95, "🚗", group="vehicle", describe_color=True, tier=TIER_LARGE),
    _o("motorcycle", "Motorcycle", "automotive", ["motorcycle"], 0.90, "🏍️", group="vehicle", describe_color=True, tier=TIER_LARGE),
    # ---- Food & Beverage
    _o("beverage_bottle", "Beverage bottle", "food_beverage", ["bottle"], 0.80, "🍾", group="container"),
    _o("soft_drink", "Soft drink can", "food_beverage", ["soda can"], 0.90, "🥤", group="container", tier=TIER_RARE),
    _o("coffee", "Coffee cup", "food_beverage", ["coffee cup", "mug"], 0.80, "☕", group="cup", tier=TIER_RARE),
    _o("glass", "Drinking glass", "food_beverage", ["drinking glass"], 0.55, "🥛", group="cup"),
    _o("food_packaging", "Food packaging", "food_beverage", ["food package", "snack bag"], 0.85, "🍫", tier=TIER_RARE),
    # ---- Furniture / Home
    _o("sofa", "Sofa", "furniture", ["sofa"], 0.75, "🛋️", group="seat", tier=TIER_LARGE),
    _o("armchair", "Armchair", "furniture", ["armchair"], 0.70, "💺", group="seat", tier=TIER_LARGE),
    _o("chair", "Chair", "furniture", ["chair"], 0.50, "🪑", group="seat", tier=TIER_LARGE),
    _o("table", "Table", "furniture", ["table"], 0.50, "🪵", tier=TIER_LARGE),
    _o("bed", "Bed", "furniture", ["bed"], 0.70, "🛏️", tier=TIER_LARGE),
    _o("lamp", "Lamp", "furniture", ["lamp"], 0.65, "💡"),
    _o("appliance", "Home appliance", "furniture", ["refrigerator", "washing machine", "microwave oven"], 0.80, "🧊", tier=TIER_RARE),
    # ---- Beauty / Personal care
    _o("cosmetics", "Cosmetics", "beauty", ["lipstick", "makeup product"], 0.90, "💄", tier=TIER_RARE),
    _o("perfume", "Perfume", "beauty", ["perfume bottle"], 0.90, "🧴", group="container", tier=TIER_RARE),
    _o("skincare", "Skincare product", "beauty", ["skincare product", "cosmetic jar"], 0.85, "🧴", tier=TIER_RARE),
]}


@dataclass(frozen=True)
class Context:
    id: str
    label: str
    prompts: tuple[str, ...]        # CLIP text prompts
    category: str | None            # commercial category it maps to; None = not commercially interesting
    icon: str = ""
    show: bool = True               # False: "cannot tell" classes; if one wins, the venue is unknown


def _c(id, label, prompts, category, icon="", show=True) -> Context:
    return Context(id, label, tuple(prompts), category, icon, show)


# Venue / setting of a scene. Non-commercial settings are included on purpose: a classifier that
# may only answer "restaurant or hotel or gym" will call a hospital corridor a hotel. Prompts name
# what is distinctive about a venue (a first version with "a scene in a hotel room" labelled almost
# every furnished interior a hotel in review).
VENUES: dict[str, Context] = {c.id: c for c in [
    _c("restaurant", "Restaurant", ["people eating at tables inside a restaurant"], "locations", "🍽️"),
    _c("cafe", "Café", ["the inside of a coffee shop with a counter and small tables"], "locations", "☕"),
    _c("hotel", "Hotel", ["a hotel reception desk in a hotel lobby", "a hotel corridor with numbered room doors"],
       "locations", "🏨"),
    _c("gym", "Gym", ["the inside of a gym with fitness machines and weights"], "locations", "🏋️"),
    _c("retail_store", "Retail store", ["the inside of a shop with products on shelves", "a market stall selling goods",
                                        "the aisles of a supermarket"], "locations", "🛍️"),
    _c("bar", "Bar / nightlife", ["the inside of a bar or nightclub with a bar counter"], "locations", "🍸"),
    _c("office", "Office", ["an office with desks, computers and office chairs", "a manager's office with a large desk"],
       "real_estate", "🏢"),
    _c("apartment", "Home interior", ["the living room of a home with a sofa", "a bedroom in a home with a bed",
                                      "the kitchen of a home", "the hallway or entrance of a home"], "real_estate", "🏠"),
    _c("villa", "Villa / house exterior", ["the exterior of a villa or large house with a garden"], "real_estate", "🏡"),
    _c("compound", "Residential compound", ["a residential compound with apartment buildings and landscaping"],
       "real_estate", "🏘️"),
    # ---- other settings (reported as context, never as a commercial location)
    _c("street", "Street", ["a city street with buildings and pavement"], None),
    _c("vehicle_interior", "Inside a vehicle", ["people sitting inside a car", "a driver behind a steering wheel"], None),
    _c("hospital", "Hospital", ["a hospital room with a patient bed", "a hospital corridor"], None),
    _c("school", "School / classroom", ["a classroom with students and desks"], None),
    _c("nature", "Outdoors / nature", ["an outdoor scene in a desert or countryside", "a beach by the sea",
                                       "a river or the open sea with boats"], None),
    _c("stadium", "Sports venue", ["a football pitch or sports stadium"], None),
    # police fired on any dark room or any scene with an officer in review -> treated as "cannot tell"
    _c("police", "Police / interrogation room", ["a police station", "a bare interrogation room with a table"], None,
       show=False),
    _c("stairwell", "Corridor / stairwell", ["an empty corridor or stairwell of a building"], None),
    _c("studio", "TV studio / stage", ["a television news studio", "a theatre stage"], None),
    _c("night_exterior", "Outdoors at night", ["an outdoor scene at night in the dark"], None),
    # ---- "cannot tell": if one of these wins, no venue is reported
    _c("closeup", "Close-up", ["a close-up of a person's face", "a close-up of two people talking"], None, show=False),
    _c("text", "Titles / graphics", ["a title card with text", "a black screen", "computer graphics or animation"],
       None, show=False),
]}

ENVIRONMENTS: dict[str, Context] = {c.id: c for c in [
    _c("indoor", "Indoor", ["an indoor scene"], None),
    _c("outdoor", "Outdoor", ["an outdoor scene"], None),
]}


def detector_queries() -> list[tuple[str, str]]:
    """[(prompt, object_type_id)] in a stable order; this list (hashed) is part of the cache key."""
    return [(p, o.id) for o in OBJECTS.values() for p in o.prompts]


def prompts_hash() -> str:
    """Changes only when the detector would be asked different questions."""
    return hashlib.sha1(json.dumps(detector_queries()).encode()).hexdigest()[:10]


def validate() -> None:
    """Internal consistency (run by the tests)."""
    for o in OBJECTS.values():
        assert o.category in CATEGORIES, f"{o.id}: unknown category {o.category}"
        assert 0.0 <= o.base_relevance <= 1.0 and o.prompts, o.id
    for c in [*VENUES.values(), *ENVIRONMENTS.values()]:
        assert c.category is None or c.category in CATEGORIES, c.id
        assert c.prompts, c.id
    prompts = [p for p, _ in detector_queries()]
    assert len(prompts) == len(set(prompts)), "duplicate detector prompt"
