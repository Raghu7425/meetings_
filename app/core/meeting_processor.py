import asyncio
import logging
import os
import uuid
import tempfile
from datetime import datetime, timezone

from sqlalchemy import select, update
from app.db.database import AsyncSessionLocal
from app.db.models import Meeting, ActionItem
from app.core.graph_client import get_recording_download_url, download_recording, get_attendees, get_call_record
from app.core.transcriber import transcribe
from app.core.extractor import extract_insights
from app.core.storage import upload_file
from app.core.notifier import send_summary_email, send_failure_email
from app.core.scheduler import schedule_reminders
from app.config import JIRA_ENABLED

log = logging.getLogger("meeting_processor")


async def _get_or_create_meeting(call_id: str) -> tuple[Meeting, bool]:
    """Return (meeting, created). Inserts placeholder row so we hold the call_id."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Meeting).where(Meeting.call_id == call_id))
        existing = result.scalar_one_or_none()
        if existing:
            return existing, False

        meeting = Meeting(call_id=call_id, status="processing")
        session.add(meeting)
        await session.commit()
        await session.refresh(meeting)
        return meeting, True


async def _mark_failed(call_id: str, error: str) -> None:
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(Meeting)
            .where(Meeting.call_id == call_id)
            .values(status="failed")
        )
        await session.commit()
    log.error(f"[processor] call_id={call_id} marked as failed: {error}")


async def process_meeting(call_id: str) -> None:
    log.info(f"[processor] Starting pipeline for call_id={call_id}")

    try:
        meeting, created = await _get_or_create_meeting(call_id)
        meeting_uuid = str(meeting.id)

        # ── 1. Fetch call record metadata ─────────────────────────────────
        try:
            record = await get_call_record(call_id)
        except Exception as e:
            raise RuntimeError(f"get_call_record failed: {e}") from e

        # Extract title / date
        start_dt = record.get("startDateTime")
        if start_dt:
            try:
                dt = datetime.fromisoformat(start_dt.replace("Z", "+00:00"))
            except ValueError:
                dt = datetime.now(timezone.utc)
        else:
            dt = datetime.now(timezone.utc)

        join_info = record.get("joinWebUrl", "")
        title = record.get("subject") or f"Meeting {dt.strftime('%Y-%m-%d %H:%M')}"

        # ── 2. Get attendees ───────────────────────────────────────────────
        try:
            attendees = await get_attendees(call_id)
        except Exception as e:
            log.warning(f"[processor] Could not fetch attendees: {e}")
            attendees = []

        # ── 3. Download recording ──────────────────────────────────────────
        try:
            download_url = await get_recording_download_url(call_id)
        except Exception as e:
            raise RuntimeError(f"get_recording_download_url failed: {e}") from e

        with tempfile.TemporaryDirectory() as tmpdir:
            mp4_path = os.path.join(tmpdir, f"{call_id}.mp4")
            try:
                await download_recording(download_url, mp4_path)
            except Exception as e:
                raise RuntimeError(f"download_recording failed: {e}") from e

            # ── 4. Upload raw .mp4 to MinIO ────────────────────────────────
            try:
                audio_object = f"recordings/{call_id}/{call_id}.mp4"
                audio_minio_url = await upload_file(mp4_path, audio_object)
            except Exception as e:
                log.warning(f"[processor] MinIO upload failed (non-fatal): {e}")
                audio_minio_url = ""

            # ── 5. Transcribe ──────────────────────────────────────────────
            try:
                segments, transcript_text = await transcribe(mp4_path)
            except Exception as e:
                raise RuntimeError(f"transcription failed: {e}") from e

            # ── 6. Upload transcript to MinIO ──────────────────────────────
            txt_path = os.path.join(tmpdir, f"{call_id}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)

            try:
                txt_object = f"recordings/{call_id}/{call_id}.txt"
                await upload_file(txt_path, txt_object)
            except Exception as e:
                log.warning(f"[processor] Transcript MinIO upload failed (non-fatal): {e}")
                txt_object = ""

        # ── 7. Extract insights ────────────────────────────────────────────
        try:
            report = await extract_insights(transcript_text)
        except Exception as e:
            raise RuntimeError(f"extract_insights failed: {e}") from e

        # ── 8. Persist to DB ───────────────────────────────────────────────
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(Meeting)
                .where(Meeting.call_id == call_id)
                .values(
                    title=report.meeting_metadata.meeting_title or title,
                    date=dt,
                    duration_minutes=report.duration_minutes,
                    participant_count=report.participant_count or len(attendees),
                    summary=report.summary.short_summary,
                    decisions=[d.decision for d in report.decisions],
                    open_questions=report.open_questions,
                    audio_path=audio_object if audio_minio_url else None,
                    transcript_path=txt_object or None,
                    structured_data=report.model_dump(),
                    status="done",
                )
            )

            # Insert action items
            for ai in report.action_items:
                action = ActionItem(
                    meeting_id=meeting.id,
                    task=ai.task,
                    owner=ai.owner,
                    deadline=ai.deadline,
                    priority=ai.priority,
                    status="open",
                )
                session.add(action)

            await session.commit()

        log.info(f"[processor] DB persisted for call_id={call_id}")

        # ── 9. Send summary email ──────────────────────────────────────────
        try:
            await send_summary_email(
                to_addresses=attendees,
                report=report,
                meeting_title=title,
                meeting_date=dt.strftime("%Y-%m-%d %H:%M UTC"),
            )
        except Exception as e:
            log.error(f"[processor] Email send failed (non-fatal): {e}")

        # ── 10. Schedule reminders ─────────────────────────────────────────
        try:
            schedule_reminders(report.action_items, meeting_title=title)
        except Exception as e:
            log.warning(f"[processor] Reminder scheduling failed (non-fatal): {e}")

        # ── 11. Jira integration (optional) ───────────────────────────────
        if JIRA_ENABLED:
            try:
                from app.core.jira_client import create_tickets_for_action_items
                ticket_pairs = await create_tickets_for_action_items(
                    report.action_items,
                    meeting_title=title,
                    meeting_id=meeting_uuid,
                )
                # Persist Jira ticket IDs
                async with AsyncSessionLocal() as session:
                    result = await session.execute(
                        select(ActionItem).where(ActionItem.meeting_id == meeting.id)
                    )
                    db_items = result.scalars().all()
                    for idx, ticket_key in ticket_pairs:
                        if idx < len(db_items):
                            db_items[idx].jira_ticket_id = ticket_key
                    await session.commit()
            except Exception as e:
                log.error(f"[processor] Jira integration failed (non-fatal): {e}")

        log.info(f"[processor] Pipeline complete for call_id={call_id}")

    except Exception as e:
        log.exception(f"[processor] Fatal error for call_id={call_id}: {e}")
        await _mark_failed(call_id, str(e))
        await send_failure_email(call_id, str(e))
