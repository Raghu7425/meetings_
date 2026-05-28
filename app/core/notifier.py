import asyncio
import logging
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, date, timedelta
from jinja2 import Environment, FileSystemLoader, select_autoescape
from app.core.extractor import MeetingReport
from app.config import (
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM,
    ADMIN_EMAIL, TEMPLATES_DIR,
)

log = logging.getLogger("notifier")

_jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(["html"]),
)


def _render_html(meeting_title: str, report: MeetingReport, meeting_date: str) -> str:
    template = _jinja_env.get_template("email_summary.html")
    return template.render(
        meeting_title=meeting_title,
        meeting_date=meeting_date,
        summary=report.summary,
        decisions=report.decisions,
        action_items=[a.model_dump() for a in report.action_items],
        open_questions=report.open_questions,
        duration_minutes=report.duration_minutes,
        participant_count=report.participant_count,
    )


def _render_plain(meeting_title: str, report: MeetingReport, meeting_date: str) -> str:
    lines = [
        f"Meeting Summary: {meeting_title}",
        f"Date: {meeting_date}",
        "",
        "SUMMARY",
        "-------",
        report.summary,
        "",
    ]

    if report.decisions:
        lines += ["DECISIONS", "---------"]
        for d in report.decisions:
            lines.append(f"• {d}")
        lines.append("")

    if report.action_items:
        lines += ["ACTION ITEMS", "------------"]
        for a in report.action_items:
            deadline = a.deadline or "No deadline"
            lines.append(f"• [{a.priority.upper()}] {a.task} — {a.owner} (due: {deadline})")
        lines.append("")

    if report.open_questions:
        lines += ["OPEN QUESTIONS", "--------------"]
        for q in report.open_questions:
            lines.append(f"• {q}")

    return "\n".join(lines)


def _send_email_sync(
    to_addresses: list[str],
    subject: str,
    html_body: str,
    plain_body: str,
) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = ", ".join(to_addresses)
    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        if SMTP_PORT == 587:
            server.starttls()
        if SMTP_USER:
            server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_FROM, to_addresses, msg.as_string())

    log.info(f"[notifier] Email sent to {to_addresses}")


async def send_summary_email(
    to_addresses: list[str],
    report: MeetingReport,
    meeting_title: str = "Teams Meeting",
    meeting_date: str | None = None,
) -> None:
    if not to_addresses:
        log.warning("[notifier] No recipients — skipping email")
        return

    if meeting_date is None:
        meeting_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html_body  = _render_html(meeting_title, report, meeting_date)
    plain_body = _render_plain(meeting_title, report, meeting_date)
    subject    = f"Meeting Summary: {meeting_title} — {meeting_date}"

    await asyncio.to_thread(_send_email_sync, to_addresses, subject, html_body, plain_body)


async def send_failure_email(call_id: str, error: str) -> None:
    subject = f"[Meeting Assistant] Processing failed for call {call_id}"
    plain   = f"The meeting pipeline failed for call_id={call_id}.\n\nError:\n{error}"
    html    = f"<p>The meeting pipeline failed for <b>{call_id}</b>.</p><pre>{error}</pre>"

    try:
        await asyncio.to_thread(
            _send_email_sync, [ADMIN_EMAIL], subject, html, plain
        )
    except Exception as e:
        log.error(f"[notifier] Could not send failure email: {e}")


async def send_reminder_email(
    to_address: str,
    task: str,
    owner: str,
    deadline: str,
    meeting_title: str,
) -> None:
    subject = f"[Reminder] Action item due tomorrow: {task[:60]}"
    plain = (
        f"Hi {owner},\n\n"
        f"This is a reminder that the following action item from '{meeting_title}' is due tomorrow ({deadline}):\n\n"
        f"  {task}\n\n"
        f"Please ensure this is completed on time."
    )
    html = (
        f"<p>Hi <b>{owner}</b>,</p>"
        f"<p>This is a reminder that the following action item from <b>{meeting_title}</b> "
        f"is due tomorrow (<b>{deadline}</b>):</p>"
        f"<blockquote>{task}</blockquote>"
        f"<p>Please ensure this is completed on time.</p>"
    )

    try:
        await asyncio.to_thread(_send_email_sync, [to_address], subject, html, plain)
        log.info(f"[notifier] Reminder sent to {to_address} for task: {task[:50]}")
    except Exception as e:
        log.error(f"[notifier] Failed to send reminder email: {e}")
