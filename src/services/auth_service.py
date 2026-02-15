"""Authentication service — online login + offline cached user."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
from sqlmodel import select

from src.db import get_session
from src.model.auth import CachedUser
from src.services.api_client import ApiClient

logger = logging.getLogger(__name__)


@dataclass
class AuthResult:
    """Lightweight result object returned after login."""

    success: bool
    user: CachedUser | None = None
    error: str | None = None
    offline: bool = False


class AuthService:
    """Handles login against the remote API and local cache fallback.

    Flow
    ----
    1. **First login** (must be online):
       - POST ``/auth/login`` with username + password (OAuth2 form).
       - GET ``/auth/me`` to retrieve user profile.
       - Persist user + tokens in ``cached_users`` table.
    2. **Subsequent launches** (offline-first):
       - If a ``CachedUser`` row exists, return it immediately.
       - Optionally attempt token refresh in the background when online.
    """

    def __init__(self) -> None:
        self._api = ApiClient()

    # ── public ──────────────────────────────────────────────

    def login_online(self, username: str, password: str) -> AuthResult:
        """Authenticate against the remote API and cache locally."""
        try:
            # 1. Login — OAuth2 password flow expects form-encoded data
            token_data = self._api.post(
                "/auth/login",
                data={"username": username, "password": password},
                headers_extra={"Content-Type": "application/x-www-form-urlencoded"},
            )

            access_token: str = token_data["access_token"]
            refresh_token: str = token_data["refresh_token"]

            # 2. Get user profile
            self._api.set_token(access_token)
            user_data = self._api.get("/auth/me")

            # 3. Cache locally
            cached = self._upsert_cached_user(
                user_data=user_data,
                access_token=access_token,
                refresh_token=refresh_token,
            )

            return AuthResult(success=True, user=cached)

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 401:
                return AuthResult(success=False, error="Credenciales incorrectas.")
            return AuthResult(
                success=False,
                error=f"Error del servidor ({status}).",
            )
        except (httpx.ConnectError, httpx.TimeoutException):
            return AuthResult(
                success=False,
                error="Sin conexion a internet. Inicie sesion al menos una vez con conexion.",
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Unexpected login error")
            return AuthResult(success=False, error=str(exc))

    def login_offline(self) -> AuthResult:
        """Return the locally cached user (if any)."""
        with get_session() as session:
            stmt = select(CachedUser).limit(1)
            user = session.exec(stmt).first()
            if user:
                return AuthResult(success=True, user=user, offline=True)
            return AuthResult(
                success=False,
                error="No hay sesion guardada. Necesita conectarse a internet la primera vez.",
            )

    def try_refresh_token(self, cached: CachedUser) -> AuthResult:
        """Attempt to refresh the access token using the stored refresh token."""
        try:
            self._api.set_token(cached.refresh_token)
            token_data = self._api.post("/auth/refresh")
            new_access = token_data["access_token"]
            new_refresh = token_data.get("refresh_token", cached.refresh_token)

            with get_session() as session:
                user = session.get(CachedUser, cached.id)
                if user:
                    user.access_token = new_access
                    user.refresh_token = new_refresh
                    session.add(user)
                    session.commit()
                    session.refresh(user)
                    return AuthResult(success=True, user=user)
            return AuthResult(success=False, error="Usuario no encontrado en cache.")
        except Exception as exc:
            logger.warning("Token refresh failed: %s", exc)
            return AuthResult(success=False, error=str(exc))

    def get_cached_user(self) -> CachedUser | None:
        """Return the cached user row or None."""
        with get_session() as session:
            return session.exec(select(CachedUser).limit(1)).first()

    def logout(self) -> None:
        """Remove the cached user (forces re-login)."""
        with get_session() as session:
            users = session.exec(select(CachedUser)).all()
            for u in users:
                session.delete(u)
            session.commit()

    # ── private ─────────────────────────────────────────────

    def _upsert_cached_user(
        self,
        user_data: dict,
        access_token: str,
        refresh_token: str,
    ) -> CachedUser:
        """Insert or update the cached user row."""
        with get_session() as session:
            user_id = user_data["id"]
            existing = session.get(CachedUser, user_id)

            if existing:
                existing.username = user_data["username"]
                existing.is_active = user_data.get("is_active", True)
                existing.role_id = user_data.get("role_id", 0)
                existing.role_name = user_data.get("role_name", "")
                existing.access_token = access_token
                existing.refresh_token = refresh_token
                session.add(existing)
                session.commit()
                session.refresh(existing)
                return existing

            new_user = CachedUser(
                id=user_id,
                username=user_data["username"],
                is_active=user_data.get("is_active", True),
                role_id=user_data.get("role_id", 0),
                role_name=user_data.get("role_name", ""),
                access_token=access_token,
                refresh_token=refresh_token,
            )
            session.add(new_user)
            session.commit()
            session.refresh(new_user)
            return new_user
