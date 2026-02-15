"""Centralised HTTP client for the Kroma REST API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.config.settings import API_BASE_URL, API_TIMEOUT

logger = logging.getLogger(__name__)


class ApiClient:
    """Thin wrapper around *httpx* that handles auth headers and base URL.

    Usage::

        client = ApiClient(access_token="eyJ...")
        data = client.get("/agricultural-units")
    """

    def __init__(self, access_token: str | None = None) -> None:
        self._base_url = API_BASE_URL.rstrip("/")
        self._timeout = API_TIMEOUT
        self._access_token = access_token

    # ── helpers ─────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        return headers

    def set_token(self, token: str) -> None:
        self._access_token = token

    # ── public verbs ────────────────────────────────────────

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Perform a GET request and return parsed JSON."""
        url = f"{self._base_url}{path}"
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(url, headers=self._headers(), params=params)
            resp.raise_for_status()
            return resp.json()

    def post(
        self,
        path: str,
        *,
        json: Any | None = None,
        data: dict[str, Any] | None = None,
        headers_extra: dict[str, str] | None = None,
    ) -> Any:
        """Perform a POST request and return parsed JSON."""
        url = f"{self._base_url}{path}"
        hdrs = self._headers()
        if headers_extra:
            hdrs.update(headers_extra)
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=hdrs, json=json, data=data)
            resp.raise_for_status()
            return resp.json()

    # ── connectivity check ──────────────────────────────────

    @staticmethod
    def is_online() -> bool:
        """Return *True* if the API is reachable (quick HEAD/GET check)."""
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(f"{API_BASE_URL.rstrip('/')}/health")
                return resp.status_code < 500
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
            return False
