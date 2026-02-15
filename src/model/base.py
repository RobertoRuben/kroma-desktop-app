from datetime import datetime

from sqlalchemy import DateTime, text
from sqlmodel import Field, SQLModel


class BaseTimestampMixin(SQLModel):
    """Base mixin for tracking record creation and updates natively in SQLite.

    Uses ``sa_type`` + ``sa_column_kwargs`` instead of ``sa_column=Column(...)``
    so that SQLModel creates a **new** Column instance for every subclass.
    This avoids the "Column already assigned to Table" error when multiple
    table-mapped classes inherit from this mixin.

    Attributes:
        created_at: The timestamp when the record was created.
        updated_at: The timestamp when the record was last modified.
    """

    created_at: datetime | None = Field(
        default=None,
        sa_type=DateTime,
        sa_column_kwargs={
            "server_default": text("(datetime('now', 'localtime'))"),
            "nullable": False,
        },
    )
    updated_at: datetime | None = Field(
        default=None,
        sa_type=DateTime,
        sa_column_kwargs={
            "server_default": text("(datetime('now', 'localtime'))"),
            "nullable": False,
            "onupdate": datetime.now,
        },
    )
