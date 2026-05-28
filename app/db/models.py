import uuid
from datetime import datetime
from sqlalchemy import String, Text, Integer, DateTime, Date, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.database import Base


class Meeting(Base):
    __tablename__ = "meetings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    call_id: Mapped[str] = mapped_column(String(256), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False, default="Untitled Meeting")
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now())
    duration_minutes: Mapped[int] = mapped_column(Integer, nullable=True)
    participant_count: Mapped[int] = mapped_column(Integer, nullable=True)
    summary: Mapped[str] = mapped_column(Text, nullable=True)
    decisions: Mapped[list] = mapped_column(JSONB, nullable=True, default=list)
    open_questions: Mapped[list] = mapped_column(JSONB, nullable=True, default=list)
    transcript_path: Mapped[str] = mapped_column(String(1024), nullable=True)
    audio_path: Mapped[str] = mapped_column(String(1024), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="processing")
    structured_data: Mapped[dict] = mapped_column(JSONB, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    action_items: Mapped[list["ActionItem"]] = relationship("ActionItem", back_populates="meeting", cascade="all, delete-orphan")


class ActionItem(Base):
    __tablename__ = "action_items"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("meetings.id"), nullable=False)
    task: Mapped[str] = mapped_column(Text, nullable=False)
    owner: Mapped[str] = mapped_column(String(256), nullable=True)
    deadline: Mapped[str] = mapped_column(Date, nullable=True)
    priority: Mapped[str] = mapped_column(String(32), nullable=True, default="medium")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    reminder_job_id: Mapped[str] = mapped_column(String(256), nullable=True)
    jira_ticket_id: Mapped[str] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    meeting: Mapped["Meeting"] = relationship("Meeting", back_populates="action_items")
