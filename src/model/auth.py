"""Local user model for offline authentication cache."""

from datetime import datetime

from sqlmodel import BIGINT, Column, DateTime, Field, SQLModel, String, Boolean, text


class CachedUser(SQLModel, table=True):
    """Stores authenticated user data locally for offline access.

    After a successful online login, the user's profile and tokens are
    cached here so the app can start without internet on subsequent launches.

    Attributes:
        id: The remote user ID (used as local PK — not auto-generated).
        username: The user's login name.
        is_active: Whether the account is active on the server.
        role_id: The user's role identifier.
        role_name: Human-readable role name.
        access_token: Last valid access token (for API calls when online).
        refresh_token: Last valid refresh token (for token renewal).
    """

    __tablename__ = "cached_users"

    id: int = Field(sa_column=Column(BIGINT, primary_key=True, autoincrement=False))
    username: str = Field(sa_column=Column(String(255), unique=True, nullable=False))
    is_active: bool = Field(default=True, sa_column=Column(Boolean, nullable=False))
    role_id: int = Field(sa_column=Column(BIGINT, nullable=False))
    role_name: str = Field(sa_column=Column(String(255), nullable=False))
    access_token: str = Field(sa_column=Column(String(2000), nullable=False))
    refresh_token: str = Field(sa_column=Column(String(2000), nullable=False))
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
        ),
    )
