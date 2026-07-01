"""GET/POST /clients."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.db.models import Client, ClientIn
from app.deps import get_mongo

router = APIRouter()


@router.get("", response_model=list[Client])
async def list_clients(mongo=Depends(get_mongo)) -> list[Client]:
    cursor = mongo.clients.find({}, sort=[("created_at", -1)])
    out: list[Client] = []
    async for doc in cursor:
        out.append(_to_model(doc))
    return out


@router.post("", response_model=Client, status_code=201)
async def create_client(body: ClientIn, mongo=Depends(get_mongo)) -> Client:
    cid = str(uuid.uuid4())
    doc = {
        "_id": cid,
        "name": body.name,
        "industry": body.industry,
        "notes": body.notes,
        "policies": [],
        "created_at": datetime.now(tz=timezone.utc),
    }
    await mongo.clients.insert_one(doc)
    return _to_model(doc)


@router.get("/{client_id}", response_model=Client)
async def get_client(client_id: str, mongo=Depends(get_mongo)) -> Client:
    doc = await mongo.clients.find_one({"_id": client_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="client not found")
    return _to_model(doc)


def _to_model(doc: dict) -> Client:
    return Client(
        _id=doc["_id"],
        name=doc.get("name", ""),
        industry=doc.get("industry"),
        notes=doc.get("notes"),
        created_at=doc.get("created_at") or datetime.now(tz=timezone.utc),
        policies=doc.get("policies", []),
    )
