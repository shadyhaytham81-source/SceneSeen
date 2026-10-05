"""Catalogue database: schema and sessions.

Local development uses one SQLite file (no server, no cloud account). Every column type used
here is portable, so production can point `database_url` at PostgreSQL without code changes;
the embedding table is laid out so it can become a pgvector column there (docs/CATALOG.md).
"""
from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, LargeBinary, Numeric, String, Text,
                        UniqueConstraint, create_engine, event)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

SCHEMA_VERSION = 1
IMAGE_ROLES = ("front", "back", "side", "detail", "lifestyle", "other")
AVAILABILITY = ("unknown", "in_stock", "out_of_stock", "discontinued")


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class Meta(Base):
    __tablename__ = "meta"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(255))


class Brand(Base):
    __tablename__ = "brands"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    website: Mapped[str | None] = mapped_column(String(500))
    description: Mapped[str | None] = mapped_column(Text)
    extra: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    products: Mapped[list["Product"]] = relationship(back_populates="brand")


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (UniqueConstraint("brand_id", "name", name="uq_product_brand_name"),
                      UniqueConstraint("brand_id", "sku", name="uq_product_brand_sku"),
                      Index("ix_product_category_type", "category", "object_type"))
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    brand_id: Mapped[int] = mapped_column(ForeignKey("brands.id"), index=True)
    name: Mapped[str] = mapped_column(String(300), index=True)
    category: Mapped[str] = mapped_column(String(40), index=True)          # commercial taxonomy category id
    object_type: Mapped[str | None] = mapped_column(String(40))            # commercial taxonomy object type id
    description: Mapped[str | None] = mapped_column(Text)
    sku: Mapped[str | None] = mapped_column(String(120), index=True)
    external_id: Mapped[str | None] = mapped_column(String(200), index=True)
    url: Mapped[str | None] = mapped_column(String(1000))
    price: Mapped[float | None] = mapped_column(Numeric(12, 2))
    currency: Mapped[str | None] = mapped_column(String(3))
    availability: Mapped[str] = mapped_column(String(20), default="unknown")
    extra: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    brand: Mapped[Brand] = relationship(back_populates="products")
    variants: Mapped[list["ProductVariant"]] = relationship(back_populates="product", cascade="all, delete-orphan",
                                                            order_by="ProductVariant.id")
    images: Mapped[list["ProductImage"]] = relationship(back_populates="product", cascade="all, delete-orphan",
                                                        order_by="ProductImage.position, ProductImage.id")


class ProductVariant(Base):
    __tablename__ = "product_variants"
    __table_args__ = (UniqueConstraint("product_id", "name", name="uq_variant_product_name"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))                         # e.g. "White / 42"
    color: Mapped[str | None] = mapped_column(String(60))
    size: Mapped[str | None] = mapped_column(String(60))
    sku: Mapped[str | None] = mapped_column(String(120), index=True)
    external_id: Mapped[str | None] = mapped_column(String(200))
    url: Mapped[str | None] = mapped_column(String(1000))
    price: Mapped[float | None] = mapped_column(Numeric(12, 2))
    currency: Mapped[str | None] = mapped_column(String(3))
    availability: Mapped[str] = mapped_column(String(20), default="unknown")
    extra: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    product: Mapped[Product] = relationship(back_populates="variants")


class ProductImage(Base):
    __tablename__ = "product_images"
    __table_args__ = (UniqueConstraint("product_id", "sha256", name="uq_image_product_sha"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"), index=True)
    variant_id: Mapped[int | None] = mapped_column(ForeignKey("product_variants.id", ondelete="SET NULL"))
    role: Mapped[str] = mapped_column(String(20), default="front")
    sha256: Mapped[str] = mapped_column(String(64), index=True)            # content hash = file name in the image store
    ext: Mapped[str] = mapped_column(String(8))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    size_bytes: Mapped[int] = mapped_column(Integer)
    original_filename: Mapped[str | None] = mapped_column(String(300))
    position: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    product: Mapped[Product] = relationship(back_populates="images")


class ImageEmbedding(Base):
    """One embedding per (image content, embedding model). Unchanged images are never re-embedded;
    a new model (or model version) gets its own rows. In PostgreSQL `vector` becomes pgvector."""
    __tablename__ = "image_embeddings"
    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_key: Mapped[str] = mapped_column(String(120), primary_key=True)
    dim: Mapped[int] = mapped_column(Integer)
    vector: Mapped[bytes] = mapped_column(LargeBinary)                     # float32, L2-normalised
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Verification(Base):
    """Current human decision for one detected object (one row per video + candidate)."""
    __tablename__ = "verifications"
    __table_args__ = (UniqueConstraint("video_id", "candidate_key", name="uq_verification_candidate"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[str] = mapped_column(String(32), index=True)
    video_name: Mapped[str | None] = mapped_column(String(500))
    candidate_key: Mapped[str] = mapped_column(String(32))
    scene_id: Mapped[int | None] = mapped_column(Integer)
    type_id: Mapped[str] = mapped_column(String(40))
    label: Mapped[str | None] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(20))                        # confirmed | no_match
    product_id: Mapped[int | None] = mapped_column(ForeignKey("products.id"))
    variant_id: Mapped[int | None] = mapped_column(ForeignKey("product_variants.id"))
    brand_id: Mapped[int | None] = mapped_column(ForeignKey("brands.id"))
    source: Mapped[str | None] = mapped_column(String(20))                 # suggestion | search
    suggestion_rank: Mapped[int | None] = mapped_column(Integer)
    match_score: Mapped[float | None] = mapped_column(Float)
    match_confidence: Mapped[float | None] = mapped_column(Float)
    match_state: Mapped[str | None] = mapped_column(String(20))
    detection_confidence: Mapped[float | None] = mapped_column(Float)
    commercial_relevance: Mapped[float | None] = mapped_column(Float)
    models: Mapped[dict] = mapped_column(JSON, default=dict)               # detector / embedder / taxonomy versions
    snapshot: Mapped[dict] = mapped_column(JSON, default=dict)             # what was on screen when decided
    decided_by: Mapped[str] = mapped_column(String(120))
    decided_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    product: Mapped[Product | None] = relationship()


class VerificationEvent(Base):
    """Append-only audit trail: every confirm / reject / change / clear, never updated or deleted."""
    __tablename__ = "verification_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[str] = mapped_column(String(32), index=True)
    candidate_key: Mapped[str] = mapped_column(String(32), index=True)
    action: Mapped[str] = mapped_column(String(20))                        # confirm | no_match | change | clear
    product_id: Mapped[int | None] = mapped_column(Integer)
    previous_product_id: Mapped[int | None] = mapped_column(Integer)
    previous_status: Mapped[str | None] = mapped_column(String(20))
    actor: Mapped[str] = mapped_column(String(120))
    at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)


class Database:
    """Engine + session factory for one database URL."""

    def __init__(self, url: str):
        self.url = url
        if url.startswith("sqlite:///"):
            Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, future=True, connect_args=connect_args)
        if url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def _pragmas(dbapi_conn, _):   # enforce foreign keys; WAL lets the UI read while a job writes
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA foreign_keys=ON")
                cur.execute("PRAGMA journal_mode=WAL")
                cur.close()
        self._Session = sessionmaker(self.engine, expire_on_commit=False, future=True)
        Base.metadata.create_all(self.engine)
        with self.session() as s:
            if s.get(Meta, "schema_version") is None:
                s.add(Meta(key="schema_version", value=str(SCHEMA_VERSION)))

    @contextmanager
    def session(self):
        s = self._Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def dispose(self) -> None:
        self.engine.dispose()
