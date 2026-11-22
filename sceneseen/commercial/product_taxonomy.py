"""Product taxonomy: normalise catalogue products and detector labels to product types,
umbrella categories and colours, using the rules in product_taxonomy.json.

Deterministic and conservative: a field that cannot be read with confidence stays None
("unknown"). Nothing here looks at images; it only decides WHICH catalogue products a
detected object may be compared with (see matching.py).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

TAXONOMY_PATH = Path(__file__).with_name("product_taxonomy.json")


def _tokens(text: str) -> list[str]:
    text = (text or "").lower().replace("&", " and ")
    return re.findall(r"[a-z0-9]+", text)


def _phrase(words: str) -> tuple[str, ...]:
    return tuple(_tokens(words))


def _find(tokens: list[str], phrase: tuple[str, ...]) -> int:
    """Index where `phrase` occurs in `tokens` (a text word may be the phrase word + 's'/'es'), else -1."""
    n = len(phrase)
    for i in range(len(tokens) - n + 1):
        if all(t == p or t == p + "s" or t == p + "es" for t, p in zip(tokens[i:i + n], phrase)):
            return i
    return -1


@dataclass(frozen=True)
class ProductInfo:
    """Normalised view of one catalogue product. The original name is kept untouched."""
    product_type: str | None          # e.g. "hoodie"; None = unknown
    umbrella: str | None              # e.g. "upper_layers"
    colors: tuple[str, ...] = ()      # colour families, e.g. ("black",) or ("green", "pink")
    patterned: bool = False           # striped / plaid / printed …
    type_source: str = ""             # "product_type" | "name" | ""

    @property
    def multicolor(self) -> bool:
        return len(self.colors) > 1


@dataclass(frozen=True)
class DetectionQuery:
    """Normalised detector output: what to look for in the catalogue."""
    label: str                        # original label, e.g. "Black jacket"
    detection_type: str | None        # detector type id, e.g. "jacket"
    strict_types: tuple[str, ...] = ()
    broad_types: tuple[str, ...] = ()
    color: str | None = None          # colour family, None = unknown
    umbrella: str | None = None       # umbrella of the first strict type (for display)
    notes: tuple[str, ...] = field(default_factory=tuple)


class ProductTaxonomy:
    def __init__(self, data: dict):
        self.data = data
        self.version: str = data["version"]
        self.types: dict[str, dict] = data["product_types"]
        self.umbrellas: dict[str, str] = data["umbrellas"]
        self.order: list[str] = data["type_order"]
        missing = [t for t in self.types if t not in self.order]
        unknown = [t for t in self.order if t not in self.types]
        if missing or unknown:
            raise ValueError(f"product_taxonomy.json: type_order must list every product type once "
                             f"(missing {missing}, unknown {unknown})")
        for t, spec in self.types.items():
            if spec["umbrella"] not in self.umbrellas:
                raise ValueError(f"product type '{t}' has unknown umbrella '{spec['umbrella']}'")
        self._type_phrases = [(t, sorted({_phrase(k) for k in self.types[t]["keywords"]}, key=len, reverse=True))
                              for t in self.order]
        self.detections: dict[str, dict] = data["detections"]
        for det, spec in self.detections.items():
            for t in list(spec.get("strict", [])) + list(spec.get("broad", [])):
                if t not in self.types:
                    raise ValueError(f"detection '{det}' refers to unknown product type '{t}'")
        self._det_synonyms = sorted(((_phrase(k), v) for k, v in data.get("detection_label_synonyms", {}).items()),
                                    key=lambda kv: len(kv[0]), reverse=True)
        self._color_phrases = sorted(((_phrase(w), fam) for fam, words in data["colors"].items() for w in words),
                                     key=lambda kv: len(kv[0]), reverse=True)
        self.near_colors: dict[str, list[str]] = data.get("near_colors", {})
        self._patterns = [_phrase(w) for w in data.get("pattern_words", [])]

    # ---- products
    def product_type_of(self, text: str) -> str | None:
        toks = _tokens(text)
        if not toks:
            return None
        for t, phrases in self._type_phrases:
            if any(_find(toks, p) >= 0 for p in phrases):
                return t
        return None

    def colors_of(self, text: str) -> tuple[str, ...]:
        """Colour families named in `text`, in order of appearance, longest phrase first ("off white" ≠ "white")."""
        toks = _tokens(text)
        used = [False] * len(toks)
        found: list[tuple[int, str]] = []
        for phrase, fam in self._color_phrases:
            i = _find(toks, phrase)
            while i >= 0:
                if not any(used[i:i + len(phrase)]):
                    for j in range(i, i + len(phrase)):
                        used[j] = True
                    found.append((i, fam))
                nxt = _find(toks[i + 1:], phrase)
                i = -1 if nxt < 0 else i + 1 + nxt
        out: list[str] = []
        for _, fam in sorted(found):
            if fam not in out:
                out.append(fam)
        return tuple(out)

    def describe_product(self, name: str, product_type: str = "", color_options: list[str] | None = None) -> ProductInfo:
        """Product type from the shop's product_type first (most reliable), the name second."""
        t, src = self.product_type_of(product_type), "product_type"
        if t is None:
            t, src = self.product_type_of(name), "name"
        colors = list(self.colors_of(name))
        for opt in color_options or []:
            for c in self.colors_of(opt):
                if c not in colors:
                    colors.append(c)
        toks = _tokens(name)
        patterned = any(_find(toks, p) >= 0 for p in self._patterns)
        return ProductInfo(t, self.types[t]["umbrella"] if t else None, tuple(colors), patterned, src if t else "")

    # ---- detections
    def detection_query(self, label: str, detection_type: str | None = None, color: str | None = None) -> DetectionQuery:
        """Normalise a detector output ("Black jacket", type "jacket", colour "black")."""
        notes = []
        det = detection_type if detection_type in self.detections else None
        if det is None:                        # free-text label (other detectors, tests): use synonyms
            toks = _tokens(label)
            for phrase, target in self._det_synonyms:
                if _find(toks, phrase) >= 0:
                    det = target
                    notes.append(f"label '{label}' read as '{target}'")
                    break
        if color is None:
            cs = self.colors_of(label)
            color = cs[0] if len(cs) == 1 else None
        if det is None:                        # a label that names a product type directly ("hoodie")
            t = self.product_type_of(label)
            if t is not None:
                umb = self.types[t]["umbrella"]
                same = tuple(x for x in self.order if x != t and self.types[x]["umbrella"] == umb)
                notes.append(f"label '{label}' read as product type '{t}'")
                return DetectionQuery(label, detection_type, (t,), same, color, umb, tuple(notes))
            return DetectionQuery(label, detection_type, (), (), color, None,
                                  tuple(notes + ["no catalogue product type corresponds to this kind of object"]))
        spec = self.detections[det]
        strict, broad = tuple(spec.get("strict", [])), tuple(spec.get("broad", []))
        umbrella = self.types[strict[0]]["umbrella"] if strict else None
        return DetectionQuery(label, det, strict, broad, color, umbrella, tuple(notes))

    # ---- colour signal
    def color_relation(self, detected: str | None, product: ProductInfo) -> str:
        """same | near | different | unknown. 'unknown' whenever either side is not clearly known."""
        if not detected or not product.colors:
            return "unknown"
        if detected in product.colors:
            return "same"
        near = set(self.near_colors.get(detected, []))
        if near & set(product.colors):
            return "near"
        return "different"


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> ProductTaxonomy:
    return ProductTaxonomy(json.loads(Path(path).read_text(encoding="utf-8")))


def load_taxonomy(path: Path | str | None = None) -> ProductTaxonomy:
    """Loaded once and re-read automatically when the JSON file changes."""
    p = Path(path or TAXONOMY_PATH)
    return _load(str(p), p.stat().st_mtime)
