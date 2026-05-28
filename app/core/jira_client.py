import asyncio
import logging
import httpx
from app.config import JIRA_URL, JIRA_USER, JIRA_API_TOKEN, JIRA_PROJECT_KEY

log = logging.getLogger("jira_client")

_PRIORITY_MAP = {
    "high":   "High",
    "medium": "Medium",
    "low":    "Low",
}


def _build_auth() -> tuple[str, str]:
    return (JIRA_USER, JIRA_API_TOKEN)


async def create_ticket(
    summary:     str,
    description: str,
    assignee:    str | None = None,
    due_date:    str | None = None,
    project_key: str | None = None,
    priority:    str = "medium",
) -> str:
    key = project_key or JIRA_PROJECT_KEY

    fields: dict = {
        "project":     {"key": key},
        "summary":     summary[:255],
        "description": {
            "type":    "doc",
            "version": 1,
            "content": [
                {
                    "type":    "paragraph",
                    "content": [{"type": "text", "text": description}],
                }
            ],
        },
        "issuetype":   {"name": "Task"},
        "priority":    {"name": _PRIORITY_MAP.get(priority.lower(), "Medium")},
    }

    if assignee:
        fields["assignee"] = {"name": assignee}

    if due_date:
        fields["duedate"] = due_date

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{JIRA_URL}/rest/api/3/issue",
            auth=_build_auth(),
            json={"fields": fields},
        )
        resp.raise_for_status()
        data = resp.json()

    ticket_id = data.get("key", "")
    log.info(f"[jira] Created ticket {ticket_id}: {summary[:50]}")
    return ticket_id


async def create_tickets_for_action_items(
    action_items: list,
    meeting_title: str,
    meeting_id: str,
) -> list[tuple[int, str]]:
    """Create one Jira ticket per action item.
    Returns list of (index, ticket_key) pairs for successfully created tickets.
    """
    results: list[tuple[int, str]] = []

    for i, item in enumerate(action_items):
        task     = item.task     if hasattr(item, "task")     else item.get("task", "")
        owner    = item.owner    if hasattr(item, "owner")    else item.get("owner", "")
        deadline = item.deadline if hasattr(item, "deadline") else item.get("deadline")
        priority = item.priority if hasattr(item, "priority") else item.get("priority", "medium")

        desc = (
            f"Action item from meeting: {meeting_title}\n"
            f"Meeting ID: {meeting_id}\n\n"
            f"Task: {task}\n"
            f"Owner: {owner}"
        )

        try:
            ticket_key = await create_ticket(
                summary=task[:200],
                description=desc,
                assignee=owner or None,
                due_date=deadline,
                priority=priority,
            )
            results.append((i, ticket_key))
        except Exception as e:
            log.error(f"[jira] Failed to create ticket for action item {i}: {e}")

    return results
