from datetime import datetime
from sqlmodel import Column, DateTime, Field, SQLModel, text


class BaseTimestampMixin(SQLModel):
    """Base mixin for tracking record creation and updates natively in SQLite.

    Attributes:
        created_at: The timestamp when the record was created. Defaults to the
            current system time via SQLite's CURRENT_TIMESTAMP.
        updated_at: The timestamp when the record was last modified.
            Automatically updated on every row modification.
    """

    created_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime,
            nullable=False,
            server_default=text("(datetime('now', 'localtime'))"),
        ),
    )
    updated_at: datetime | None = Field(
        default=None,
        sa_column=Column(
            DateTime,
            nullable=False,
            server_default=text("(datetime('now', 'localtime'))"),
            onupdate=datetime.now,
        ),
    )
