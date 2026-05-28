import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from app.db.database import get_db
from app.db.models import Meeting, ActionItem
from app.core.storage import download_file
from app.core.notifier import send_summary_email
from app.core.extractor import MeetingReport, ActionItemSchema
import tempfile
import os

log = logging.getLogger("meetings_api")
router = APIRouter(prefix="/meetings", tags=["meetings"])
action_router = APIRouter(prefix="/action-items", tags=["action-items"])


class ActionItemUpdate(BaseModel):
    status: str  # open | done | cancelled


# ── Meetings ──────────────────────────────────────────────────────────────

@router.get("")
async def list_meetings(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * page_size
    result = await db.execute(
        select(Meeting)
        .order_by(Meeting.date.desc())
        .offset(offset)
        .limit(page_size)
    )
    meetings = result.scalars().all()

    total_result = await db.execute(select(func.count(Meeting.id)))
    total = total_result.scalar_one()

    return {
        "page":      page,
        "page_size": page_size,
        "total":     total,
        "items": [
            {
                "id":                str(m.id),
                "call_id":           m.call_id,
                "title":             m.title,
                "date":              m.date.isoformat() if m.date else None,
                "duration_minutes":  m.duration_minutes,
                "participant_count": m.participant_count,
                "status":            m.status,
            }
            for m in meetings
        ],
    }


@router.get("/{meeting_id}")
async def get_meeting(meeting_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Meeting)
        .options(selectinload(Meeting.action_items))
        .where(Meeting.id == uuid.UUID(meeting_id))
    )
    meeting = result.scalar_one_or_none()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    return {
        "id":                str(meeting.id),
        "call_id":           meeting.call_id,
        "title":             meeting.title,
        "date":              meeting.date.isoformat() if meeting.date else None,
        "duration_minutes":  meeting.duration_minutes,
        "participant_count": meeting.participant_count,
        "summary":           meeting.summary,
        "decisions":         meeting.decisions,
        "open_questions":    meeting.open_questions,
        "status":            meeting.status,
        "audio_path":        meeting.audio_path,
        "action_items": [
            {
                "id":             str(ai.id),
                "task":           ai.task,
                "owner":          ai.owner,
                "deadline":       str(ai.deadline) if ai.deadline else None,
                "priority":       ai.priority,
                "status":         ai.status,
                "jira_ticket_id": ai.jira_ticket_id,
            }
            for ai in meeting.action_items
        ],
    }


@router.get("/{meeting_id}/transcript")
async def get_transcript(meeting_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Meeting).where(Meeting.id == uuid.UUID(meeting_id))
    )
    meeting = result.scalar_one_or_none()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    if not meeting.transcript_path:
        raise HTTPException(status_code=404, detail="Transcript not available")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, "transcript.txt")
        try:
            await download_file(meeting.transcript_path, local_path)
            with open(local_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not fetch transcript: {e}")

    return PlainTextResponse(content=content)


@router.post("/{meeting_id}/resend-email")
async def resend_email(
    meeting_id: str,
    to: list[str] = Query(default=[]),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Meeting)
        .options(selectinload(Meeting.action_items))
        .where(Meeting.id == uuid.UUID(meeting_id))
    )
    meeting = result.scalar_one_or_none()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    report = MeetingReport(
        summary=meeting.summary or "",
        decisions=meeting.decisions or [],
        action_items=[
            ActionItemSchema(
                task=ai.task,
                owner=ai.owner or "Unassigned",
                deadline=str(ai.deadline) if ai.deadline else None,
                priority=ai.priority or "medium",
            )
            for ai in meeting.action_items
        ],
        open_questions=meeting.open_questions or [],
        duration_minutes=meeting.duration_minutes,
        participant_count=meeting.participant_count,
    )

    recipients = to or []
    if not recipients:
        raise HTTPException(status_code=400, detail="Provide at least one recipient via ?to=email")

    try:
        await send_summary_email(
            to_addresses=recipients,
            report=report,
            meeting_title=meeting.title,
            meeting_date=meeting.date.strftime("%Y-%m-%d %H:%M UTC") if meeting.date else "",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email failed: {e}")

    return {"status": "sent", "recipients": recipients}


# ── Action Items ──────────────────────────────────────────────────────────

@action_router.patch("/{item_id}")
async def update_action_item(
    item_id: str,
    body: ActionItemUpdate,
    db: AsyncSession = Depends(get_db),
):
    valid_statuses = {"open", "done", "cancelled"}
    if body.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"status must be one of {valid_statuses}")

    result = await db.execute(
        select(ActionItem).where(ActionItem.id == uuid.UUID(item_id))
    )
    item = result.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Action item not found")

    item.status = body.status
    await db.commit()

    return {"id": item_id, "status": item.status}
