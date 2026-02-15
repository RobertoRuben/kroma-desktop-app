import os

from sqlmodel import SQLModel, create_engine, Session, text
from sqlalchemy import event
from sqlalchemy.engine import Engine

from src.config.settings import BASE_DIR

# Database configuration — store next to the executable / project root
_DB_PATH = os.path.join(BASE_DIR, "kroma_desktop.db")
SQLITE_URL = f"sqlite:///{_DB_PATH}"

# SQLModel.create_engine is a direct proxy to SQLAlchemy's create_engine
engine = create_engine(SQLITE_URL, echo=False)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record) -> None:
    """Enables SQLite foreign key support using a low-level event listener.

    This is necessary because SQLite disables foreign keys by default.
    SQLModel doesn't have a high-level function for this, so we use SQLAlchemy events.

    Args:
        dbapi_connection: The raw connection to the SQLite database.
        connection_record: Metadata about the connection.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_db_and_tables() -> None:
    """Initializes the database by creating all registered tables.

    This must be called at app startup. It imports all model modules
    to ensure they are registered in the metadata.
    """
    # Import models so SQLModel registers them in metadata
    import src.model.catalogs  # noqa: F401
    import src.model.production  # noqa: F401
    import src.model.analysis  # noqa: F401
    import src.model.auth  # noqa: F401

    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    """Returns a new SQLModel session for database operations."""
    return Session(engine)
