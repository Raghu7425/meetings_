import logging
import asyncio
import uuid
from datetime import datetime, date, timedelta, timezone
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from app.config import WEBHOOK_BASE_URL

log = logging.getLogger("scheduler")

_scheduler: AsyncIOScheduler | None = None
_graph_subscription_id: str | None  = None


def get_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler(
            jobstores={"default": MemoryJobStore()},
            job_defaults={"coalesce": True, "max_instances": 1},
            timezone="UTC",
        )
    return _scheduler


def start_scheduler() -> None:
    sched = get_scheduler()
    if not sched.running:
        sched.start()
        log.info("[scheduler] APScheduler started")
        # Schedule Graph API subscription renewal every 2 days
        sched.add_job(
            _renew_graph_subscription,
            "interval",
            days=2,
            id="graph_sub_renewal",
            replace_existing=True,
        )


def stop_scheduler() -> None:
    sched = get_scheduler()
    if sched.running:
        sched.shutdown(wait=False)
        log.info("[scheduler] APScheduler stopped")


def schedule_once(func, run_date: datetime, job_id: str | None = None, **kwargs) -> str:
    sched = get_scheduler()
    jid = job_id or str(uuid.uuid4())
    sched.add_job(
        func,
        "date",
        run_date=run_date,
        id=jid,
        replace_existing=True,
        kwargs=kwargs,
    )
    log.info(f"[scheduler] Scheduled job {jid} at {run_date.isoformat()}")
    return jid


def cancel_job(job_id: str) -> None:
    sched = get_scheduler()
    try:
        sched.remove_job(job_id)
        log.info(f"[scheduler] Cancelled job {job_id}")
    except Exception:
        pass


def schedule_reminders(action_items: list, meeting_title: str = "Teams Meeting") -> dict[str, str]:
    """Schedule a 24-hour-before reminder email for each action item with a deadline.
    Returns mapping of action_item index → job_id."""
    from app.core.notifier import send_reminder_email

    job_ids: dict[str, str] = {}

    for item in action_items:
        deadline_str = item.deadline if hasattr(item, "deadline") else item.get("deadline")
        if not deadline_str:
            continue

        try:
            deadline_date = date.fromisoformat(deadline_str)
            reminder_dt   = datetime(
                deadline_date.year, deadline_date.month, deadline_date.day,
                9, 0, 0, tzinfo=timezone.utc
            ) - timedelta(days=1)

            if reminder_dt <= datetime.now(timezone.utc):
                continue  # deadline already passed or within 24h

            owner = item.owner if hasattr(item, "owner") else item.get("owner", "")
            task  = item.task  if hasattr(item, "task")  else item.get("task", "")

            # We notify SMTP_FROM address; real deployments would look up owner's email
            jid = schedule_once(
                send_reminder_email,
                run_date=reminder_dt,
                to_address=None,  # resolved at fire time via owner lookup
                task=task,
                owner=owner,
                deadline=deadline_str,
                meeting_title=meeting_title,
            )
            task_key = f"{task[:30]}_{owner}"
            job_ids[task_key] = jid

        except (ValueError, AttributeError) as e:
            log.warning(f"[scheduler] Could not schedule reminder: {e}")

    return job_ids


async def _renew_graph_subscription() -> None:
    global _graph_subscription_id

    if not _graph_subscription_id:
        log.info("[scheduler] No subscription ID stored — creating new subscription")
        try:
            from app.core.graph_client import create_subscription
            notification_url = f"{WEBHOOK_BASE_URL}/webhook/teams"
            data = await create_subscription(notification_url)
            _graph_subscription_id = data.get("id")
        except Exception as e:
            log.error(f"[scheduler] Could not create Graph subscription: {e}")
        return

    try:
        from datetime import timezone as tz
        from app.core.graph_client import renew_subscription
        new_expiry = (datetime.now(tz.utc) + timedelta(days=2, hours=23)).isoformat()
        await renew_subscription(_graph_subscription_id, new_expiry)
        log.info(f"[scheduler] Graph subscription {_graph_subscription_id} renewed")
    except Exception as e:
        log.error(f"[scheduler] Subscription renewal failed: {e}")
        _graph_subscription_id = None  # Force re-create on next run


def set_graph_subscription_id(sub_id: str) -> None:
    global _graph_subscription_id
    _graph_subscription_id = sub_id
