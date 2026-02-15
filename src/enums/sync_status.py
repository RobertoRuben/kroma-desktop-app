from enum import Enum


class SyncStatus(str, Enum):
    """Status for future API synchronization."""
    PENDING = "pending"
    SYNCED = "synced"
    FAILED = "failed"
