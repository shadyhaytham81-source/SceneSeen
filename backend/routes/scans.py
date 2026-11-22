"""POST /scans, PATCH /scans/{id}, GET /scans/{id}/results (FR3-FR8)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..auth import current_user
from ..deps import get_scan_service, get_store
from ..schemas import Scan, ScanCreated, ScanInputKind, ScanResults, TagRequest, User
from ..services.scan_service import MAX_BYTES, ScanService
from ..store import Store

router = APIRouter(prefix="/scans", tags=["scans"])


def _scan(store: Store, scan_id: str, user: User | None) -> Scan:
    scan = store.scans.get(scan_id)
    if not scan:
        raise HTTPException(404, "unknown scan")
    if scan.user_id and (not user or user.id != scan.user_id):
        raise HTTPException(403, "not your scan")
    return scan


@router.post("", response_model=ScanCreated, status_code=201)
async def create_scan(image: UploadFile = File(...), kind: ScanInputKind = Form(ScanInputKind.camera),
                      user: User | None = Depends(current_user), svc: ScanService = Depends(get_scan_service)):
    data = await image.read()
    if not data:
        raise HTTPException(400, "empty image")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image too large")
    return svc.process(data, kind, user)


@router.patch("/{scan_id}", response_model=Scan)
def tag_scan(scan_id: str, body: TagRequest, user: User | None = Depends(current_user),
             store: Store = Depends(get_store), svc: ScanService = Depends(get_scan_service)):
    scan = _scan(store, scan_id, user)
    if body.episode_id and body.episode_id not in store.episodes:
        raise HTTPException(404, "unknown episode")
    return svc.tag(scan, body.episode_id)


@router.get("/{scan_id}/results", response_model=ScanResults)
def scan_results(scan_id: str, user: User | None = Depends(current_user), store: Store = Depends(get_store)):
    scan = _scan(store, scan_id, user)
    return ScanResults(scan=scan, items=store.scan_results.get(scan_id, []))
