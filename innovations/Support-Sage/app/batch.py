"""Batch prediction endpoint for processing multiple tickets at once."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import TicketRequest

router = APIRouter(prefix="/api/v1", tags=["batch"])


@router.post("/predict/batch", summary="Batch triage multiple tickets")
async def batch_predict(tickets: list[TicketRequest], db: Session = Depends(get_db)) -> dict[str, Any]:
    """Predict priority for a batch of up to 50 tickets."""
    if len(tickets) > 50:
        return {"error": "Batch size exceeds limit of 50"}
    from app.main import get_model
    from app.model import predict
    from app.monitoring import log_prediction

    model, encoder, regressor = get_model()
    results = []
    for ticket in tickets:
        result = predict(ticket.subject, ticket.description, model, encoder, regressor)
        result["ticket_id"] = ticket.ticket_id
        log_prediction(db, ticket.ticket_id, ticket.subject, ticket.description, result)
        results.append(result)
    return {"results": results, "count": len(results)}
