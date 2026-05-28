import logging
import asyncio
import httpx
from app.config import AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET

log = logging.getLogger("graph_client")

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
TOKEN_URL  = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"

_token_cache: dict = {"token": None, "expires_at": 0.0}


async def get_access_token() -> str:
    import time

    now = time.monotonic()
    if _token_cache["token"] and _token_cache["expires_at"] > now + 60:
        return _token_cache["token"]

    url = TOKEN_URL.format(tenant=AZURE_TENANT_ID)
    data = {
        "grant_type":    "client_credentials",
        "client_id":     AZURE_CLIENT_ID,
        "client_secret": AZURE_CLIENT_SECRET,
        "scope":         "https://graph.microsoft.com/.default",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, data=data)
        resp.raise_for_status()
        body = resp.json()

    token = body["access_token"]
    expires_in = int(body.get("expires_in", 3600))
    _token_cache["token"] = token
    _token_cache["expires_at"] = now + expires_in
    log.info("[graph] Access token acquired")
    return token


async def _get(path: str, **kwargs) -> dict:
    token = await get_access_token()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{GRAPH_BASE}{path}", headers=headers, **kwargs)
        resp.raise_for_status()
        return resp.json()


async def get_call_record(call_id: str) -> dict:
    """Return callRecord including participants and session metadata."""
    return await _get(f"/communications/callRecords/{call_id}?$expand=sessions($expand=segments)")


async def get_recording_download_url(call_id: str) -> str:
    """Return a direct download URL for the meeting recording."""
    record = await get_call_record(call_id)

    # Walk sessions → segments → recordings
    sessions = record.get("sessions", [])
    for session in sessions:
        for segment in session.get("segments", []):
            recordings = segment.get("recordings", [])
            for rec in recordings:
                url = (
                    rec.get("content")
                    or rec.get("downloadUrl")
                    or rec.get("@microsoft.graph.downloadUrl")
                )
                if url:
                    log.info(f"[graph] Recording URL found for call {call_id}")
                    return url

    # Fallback: try the onlineMeeting recordings endpoint
    # (requires the meeting's online meeting ID, which may be in the call record)
    raise ValueError(f"No recording URL found in callRecord for call_id={call_id}")


async def download_recording(url: str, dest_path: str) -> str:
    """Stream a recording file to dest_path; returns dest_path."""
    token = await get_access_token()
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=600, follow_redirects=True) as client:
        async with client.stream("GET", url, headers=headers) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as fh:
                async for chunk in resp.aiter_bytes(chunk_size=1 << 20):  # 1 MB
                    fh.write(chunk)

    log.info(f"[graph] Recording saved to {dest_path}")
    return dest_path


async def get_attendees(call_id: str) -> list[str]:
    """Return list of attendee email addresses from the call record."""
    record = await get_call_record(call_id)
    emails: list[str] = []

    participants = record.get("participants", [])
    for p in participants:
        identity = p.get("identity", {})
        user = identity.get("user", {})
        email = user.get("userPrincipalName") or user.get("mail")
        if email:
            emails.append(email)

    log.info(f"[graph] Found {len(emails)} attendees for call {call_id}")
    return emails


async def renew_subscription(subscription_id: str, expiry_datetime: str) -> dict:
    """Renew an existing Graph API webhook subscription."""
    token = await get_access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    body = {"expirationDateTime": expiry_datetime}

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.patch(
            f"{GRAPH_BASE}/subscriptions/{subscription_id}",
            headers=headers,
            json=body,
        )
        resp.raise_for_status()
        return resp.json()


async def create_subscription(notification_url: str) -> dict:
    """Create a new callRecords webhook subscription valid for 3 days."""
    from datetime import datetime, timedelta, timezone

    expiry = (datetime.now(timezone.utc) + timedelta(days=2, hours=23)).isoformat()
    token = await get_access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    body = {
        "changeType":           "created,updated",
        "notificationUrl":      notification_url,
        "resource":             "/communications/callRecords",
        "expirationDateTime":   expiry,
        "clientState":          "TeamsWebhookSecret",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{GRAPH_BASE}/subscriptions", headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

    log.info(f"[graph] Subscription created: {data.get('id')}")
    return data
