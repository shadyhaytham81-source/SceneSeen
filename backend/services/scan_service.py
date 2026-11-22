"""ScanService: one scan end to end (FR3-FR8), within 5 s (NFR1).

    preprocess -> embed frame -> suggest episode -> detect -> embed crops -> match -> store

The orchestration is final; Detector and Embedder are stubs (see their modules).
"""
from __future__ import annotations

import logging
import time

from ..schemas import ItemResults, MatchLabel, Scan, ScanCreated, ScanInputKind, TagSource, User
from ..store import Store
from .analytics import AnalyticsService
from .detector import Detector
from .embedder import Embedder
from .matcher import NO_MATCH, Matcher
from .scene_recogniser import SceneRecogniser

log = logging.getLogger("sceneseen.backend.scan")
MAX_BYTES = 8 * 1024 * 1024


def preprocess(image: bytes, kind: ScanInputKind) -> bytes:
    """STUB: milestone 3 resizes/compresses, and for camera scans corrects the screen's perspective."""
    return image


class ScanService:
    def __init__(self, store: Store, detector: Detector | None = None, embedder: Embedder | None = None,
                 matcher: Matcher | None = None):
        self.store = store
        self.detector = detector or Detector()
        self.embedder = embedder or Embedder()
        self.matcher = matcher or Matcher(store)
        self.recogniser = SceneRecogniser(store)
        self.analytics = AnalyticsService(store)

    def process(self, image: bytes, kind: ScanInputKind, user: User | None) -> ScanCreated:
        t0 = time.perf_counter()
        image = preprocess(image, kind)
        frame_vec = self.embedder.embed(image)
        suggested = self.recogniser.suggest(frame_vec)

        scan = Scan(user_id=user.id if user else None, input_kind=kind)
        if suggested:
            scan.episode_id, scan.tag_source = suggested.id, TagSource.suggested
        self.store.scans[scan.id] = scan
        self.store.scan_vectors[scan.id] = frame_vec
        self.store.scan_results[scan.id] = self._match(scan, image)

        scan.processing_ms = int((time.perf_counter() - t0) * 1000)
        log.info("scan %s: %d items in %d ms", scan.id, len(self.store.scan_results[scan.id]), scan.processing_ms)
        return ScanCreated(scan_id=scan.id, suggested_episode=suggested)

    def _match(self, scan: Scan, image: bytes) -> list[ItemResults]:
        out = []
        for item in self.detector.detect(image):
            vec = self.embedder.embed(image)          # STUB: milestone 3 embeds the item's crop, not the frame
            matches = self.matcher.find_similar(item, vec, scan.episode_id)
            for m in matches:
                self.store.matches[m.id] = m
                self.analytics.record_view(m.product.id, scan.id)
            out.append(ItemResults(item=item, matches=matches, message=None if matches else NO_MATCH))
        return out

    def tag(self, scan: Scan, episode_id: str | None) -> Scan:
        """FR4.3-FR4.4. Re-labels results, since Exact depends on the episode."""
        scan.episode_id = episode_id
        scan.tag_source = TagSource.user if episode_id else TagSource.none
        for res in self.store.scan_results.get(scan.id, []):
            exact = self.store.approved_products(episode_id)
            for m in res.matches:
                m.label = MatchLabel.exact if m.product.id in exact else MatchLabel.similar
        return scan
