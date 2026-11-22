"""Data classes from the SRS class diagram (3.9), as API models.

These mirror the planned Supabase tables in db/schema.sql. Embeddings are kept out of the
API responses; they live only in the store.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


def new_id() -> str:
    return uuid4().hex


def now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------- users

class Role(str, Enum):
    viewer = "viewer"
    brand_partner = "brand_partner"
    stylist = "stylist"
    admin = "admin"


class User(BaseModel):
    id: str
    role: Role = Role.viewer
    brand_id: str | None = None          # BrandPartner only: the one brand they manage


# ---------------------------------------------------------------- shows

class Show(BaseModel):
    id: str = Field(default_factory=new_id)
    title: str
    type: str = "series"                 # series | movie


class Episode(BaseModel):
    id: str = Field(default_factory=new_id)
    show_id: str
    season: int
    number: int


# ---------------------------------------------------------------- catalogue

class Brand(BaseModel):
    id: str = Field(default_factory=new_id)
    name: str
    price_tier: str | None = None        # budget | mid | premium


class Product(BaseModel):
    id: str = Field(default_factory=new_id)
    brand_id: str
    title: str
    category: str                        # from the product taxonomy
    price: float | None = None           # EGP
    product_url: str
    image_url: str | None = None
    colour: str | None = None


# ---------------------------------------------------------------- scans

class ScanInputKind(str, Enum):
    camera = "camera"                    # photo of a TV / laptop screen (main method)
    screenshot = "screenshot"


class TagSource(str, Enum):
    user = "user"
    suggested = "suggested"
    none = "none"


class Box(BaseModel):
    x: float                             # fractions of the image, 0..1
    y: float
    w: float
    h: float


class DetectedItem(BaseModel):
    id: str = Field(default_factory=new_id)
    category: str
    colour: str | None = None
    confidence: float
    box: Box


class MatchLabel(str, Enum):
    exact = "Exact"
    similar = "Similar"


class MatchResult(BaseModel):
    """ExactMatch / SimilarMatch from the class diagram, told apart by `label`."""
    id: str = Field(default_factory=new_id)
    rank: int
    similarity: float
    label: MatchLabel
    product: Product
    link: str                            # /r/{id}: tracked redirect to the brand's page


class ItemResults(BaseModel):
    item: DetectedItem
    matches: list[MatchResult]
    message: str | None = None           # "no close match yet" when nothing passes the cutoff


class Scan(BaseModel):
    id: str = Field(default_factory=new_id)
    user_id: str | None = None           # scanning needs no account (FR1)
    input_kind: ScanInputKind
    episode_id: str | None = None
    tag_source: TagSource = TagSource.none
    flagged: bool = False
    created_at: datetime = Field(default_factory=now)
    processing_ms: int | None = None     # logged for NFR1 (scan < 5 s)


class ScanCreated(BaseModel):
    scan_id: str
    suggested_episode: Episode | None = None


class ScanResults(BaseModel):
    scan: Scan
    items: list[ItemResults]


class TagRequest(BaseModel):
    episode_id: str | None = None        # None = user skipped


# ---------------------------------------------------------------- confirmations and engagement

class VerifiedStatus(str, Enum):
    pending = "pending"
    approved = "approved"


class VerifiedItemCreate(BaseModel):
    episode_id: str
    product_id: str
    character: str | None = None
    scene_time: str | None = None        # e.g. "00:12:30"


class VerifiedItem(VerifiedItemCreate):
    id: str = Field(default_factory=new_id)
    confirmed_by: str
    status: VerifiedStatus = VerifiedStatus.pending
    approved_by: str | None = None


class Click(BaseModel):
    id: str = Field(default_factory=new_id)
    match_id: str
    product_id: str
    user_id: str | None = None
    utm_source: str = "sceneseen"
    clicked_at: datetime = Field(default_factory=now)


class SavedItem(BaseModel):
    user_id: str
    product_id: str
    saved_at: datetime = Field(default_factory=now)


# ---------------------------------------------------------------- reports

class ImportReport(BaseModel):
    brand_id: str
    format: str
    imported: int
    skipped_rows: list[int]              # unreadable rows (1-based, header excluded)


class ProductStats(BaseModel):
    product_id: str
    title: str
    views: int
    clicks: int


class BrandReport(BaseModel):
    brand_id: str
    period_days: int
    views: int
    clicks: int
    products: list[ProductStats]         # aggregated only, no user ids (NFR12)
