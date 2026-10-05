"""Catalogue database: schema and sessions.

Local development uses one SQLite file (no server, no cloud account). Every column type used
here is portable, so production can point `database_url` at PostgreSQL without
code changes (docs/CATALOG.md). `image_embeddings` is only used by the disabled experimental matcher.
"""
from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import (JSON, Boolean, DateTime, ForeignKey, Index, Integer, LargeBinary, Numeric, String, Text,
                        UniqueConstraint, create_engine, event)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

SCHEMA_VERSION = 2          # 2: human identifications replace the (experimental) match verifications
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


IDENT_STATUSES = ("brand_identified", "product_identified", "confirmed", "unknown_product", "no_product", "not_commercial")


class Identification(Base):
    """What a PERSON says a detected commercial object is (one current row per video + object).

    Kept apart from the detection data on purpose: detections live in the per-video analysis
    cache and are produced by a model; this table only ever holds human statements. An object
    without a row is simply "unidentified".
    """
    __tablename__ = "identifications"
    __table_args__ = (UniqueConstraint("video_id", "object_key", name="uq_identification_object"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[str] = mapped_column(String(32), index=True)
    video_name: Mapped[str | None] = mapped_column(String(500))
    object_key: Mapped[str] = mapped_column(String(32))                    # the grouped object (all its occurrences)
    scene_id: Mapped[int | None] = mapped_column(Integer)
    type_id: Mapped[str] = mapped_column(String(40))                       # generic label at the time, for reports
    label: Mapped[str | None] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(24))                        # one of IDENT_STATUSES
    brand_id: Mapped[int | None] = mapped_column(ForeignKey("brands.id"), index=True)
    product_id: Mapped[int | None] = mapped_column(ForeignKey("products.id"), index=True)
    variant_id: Mapped[int | None] = mapped_column(ForeignKey("product_variants.id"))
    notes: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(20), default="human")       # always "human" today
    identified_by: Mapped[str] = mapped_column(String(120))
    identified_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    brand: Mapped[Brand | None] = relationship()
    product: Mapped[Product | None] = relationship()
    variant: Mapped[ProductVariant | None] = relationship()


class IdentificationEvent(Base):
    """Append-only history: every identification, change and removal, with the value before it.
    Rows are never updated or deleted, so no identification is ever silently overwritten."""
    __tablename__ = "identification_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[str] = mapped_column(String(32), index=True)
    object_key: Mapped[str] = mapped_column(String(32), index=True)
    action: Mapped[str] = mapped_column(String(20))                        # identify | change | clear
    actor: Mapped[str] = mapped_column(String(120))
    at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    before: Mapped[dict | None] = mapped_column(JSON)                      # {status, brand_id, brand, product_id, ...}
    after: Mapped[dict | None] = mapped_column(JSON)
    notes: Mapped[str | None] = mapped_column(Text)


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
            row = s.get(Meta, "schema_version")
            if row is None:
                s.add(Meta(key="schema_version", value=str(SCHEMA_VERSION)))
            elif row.value != str(SCHEMA_VERSION):      # only additive changes so far: create_all added the new tables
                row.value = str(SCHEMA_VERSION)

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
