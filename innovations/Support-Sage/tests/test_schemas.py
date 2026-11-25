"""Tests for Pydantic request/response schemas."""
import pytest
from pydantic import ValidationError

from app.schemas import TicketRequest


def test_ticket_request_valid():
    t = TicketRequest(subject="VPN issue", description="Cannot connect to VPN from home")
    assert t.subject == "VPN issue"

def test_ticket_request_strips_whitespace():
    t = TicketRequest(subject="  VPN   ", description="  Some long description here  ")
    assert t.subject == "VPN"
    assert not t.description.startswith(" ")

def test_ticket_request_too_short_subject():
    with pytest.raises(ValidationError):
        TicketRequest(subject="A", description="Some long enough description")

def test_ticket_request_too_short_description():
    with pytest.raises(ValidationError):
        TicketRequest(subject="Valid subject", description="Too short")

def test_ticket_request_auto_id():
    t = TicketRequest(subject="Test subject", description="Test description that is long enough")
    assert len(t.ticket_id) > 0
