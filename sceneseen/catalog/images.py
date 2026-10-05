"""Content-addressed store for product reference images.

A file is stored once under its SHA-256 (`ab/abcdef….jpg`); the database only keeps the hash,
so nothing depends on where the project folder lives, identical uploads are de-duplicated, and
an embedding computed for a hash stays valid for as long as that content exists.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

MAX_BYTES = 25 * 1024 * 1024
FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}


class ImageError(ValueError):
    """Upload is not a usable image."""


class ImageStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, sha256: str, ext: str) -> Path:
        return self.root / sha256[:2] / f"{sha256}.{ext}"

    def save(self, data: bytes) -> dict:
        """Validate and store image bytes. Returns {sha256, ext, width, height, size_bytes}."""
        from PIL import Image, UnidentifiedImageError

        if not data:
            raise ImageError("the image file is empty")
        if len(data) > MAX_BYTES:
            raise ImageError(f"the image is larger than {MAX_BYTES // (1024 * 1024)} MB")
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
            with Image.open(io.BytesIO(data)) as im:
                fmt, (w, h) = im.format, im.size
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
            raise ImageError("the file is not a readable image (use JPEG, PNG or WebP)") from e
        if fmt not in FORMATS:
            raise ImageError(f"unsupported image format {fmt} (use JPEG, PNG or WebP)")
        if min(w, h) < 32:
            raise ImageError("the image is too small to be a useful reference (under 32 px)")
        sha = hashlib.sha256(data).hexdigest()
        p = self.path(sha, FORMATS[fmt])
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(p)
        return {"sha256": sha, "ext": FORMATS[fmt], "width": w, "height": h, "size_bytes": len(data)}

    def load_rgb(self, sha256: str, ext: str):
        """RGB numpy array, or raises ImageError when the file is missing / unreadable."""
        import numpy as np
        from PIL import Image, UnidentifiedImageError

        p = self.path(sha256, ext)
        if not p.exists():
            raise ImageError(f"image file {sha256[:12]}… is missing from the image store")
        try:
            with Image.open(p) as im:
                return np.asarray(im.convert("RGB"))
        except (UnidentifiedImageError, OSError) as e:
            raise ImageError(f"image file {sha256[:12]}… is unreadable") from e
