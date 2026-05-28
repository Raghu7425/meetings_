import logging
from fastapi import APIRouter, BackgroundTasks, Request, Query, Response
from fastapi.responses import PlainTextResponse, JSONResponse
from app.core.meeting_processor import process_meeting
from app.config import TEAMS_WEBHOOK_SECRET

log = logging.getLogger("webhook")
router = APIRouter(prefix="/webhook", tags=["webhook"])


@router.get("/teams")
async def teams_webhook_validation(
    validationToken: str = Query(default=None),
) -> Response:
    """Microsoft Graph API sends a GET with ?validationToken=xxx to verify the endpoint.
    Must respond 200 with the token as plain text within a few seconds.
    """
    if validationToken:
        log.info("[webhook] Validation handshake received")
        return PlainTextResponse(content=validationToken, status_code=200)

    return JSONResponse({"status": "webhook active"})


@router.post("/teams")
async def teams_webhook_event(
    request: Request,
    background_tasks: BackgroundTasks,
) -> Response:
    """Receive Graph API change notifications for call records.
    Must respond 202 within 3 seconds; heavy processing runs in background.
    """
    try:
        body = await request.json()
    except Exception:
        log.warning("[webhook] Could not parse request body")
        return Response(status_code=202)

    notifications = body.get("value", [])
    if not notifications:
        log.debug("[webhook] Empty notification batch")
        return Response(status_code=202)

    for notification in notifications:
        # Verify clientState matches our secret
        client_state = notification.get("clientState", "")
        if client_state != TEAMS_WEBHOOK_SECRET:
            log.warning(
                f"[webhook] clientState mismatch — got {client_state!r}, expected secret"
            )
            continue

        resource_data = notification.get("resourceData", {})
        call_id = (
            resource_data.get("id")
            or notification.get("resourceData", {}).get("callId")
        )

        # Fall back to parsing resource URL like /communications/callRecords/{id}
        if not call_id:
            resource = notification.get("resource", "")
            parts = resource.rstrip("/").split("/")
            if parts:
                call_id = parts[-1]

        if not call_id:
            log.warning("[webhook] Could not extract call_id from notification — skipping")
            continue

        log.info(f"[webhook] Queuing pipeline for call_id={call_id}")
        background_tasks.add_task(process_meeting, call_id)

    return Response(status_code=202)
