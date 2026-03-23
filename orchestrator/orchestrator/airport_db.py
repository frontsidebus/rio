"""Local SQLite airport database for offline airport lookups.

Replaces the external aviationapi.com HTTP dependency with a local SQLite
database built from OurAirports public-domain data.  Provides richer data
(runways, frequencies) than the cloud API ever did.

Usage::

    db = AirportDB()                     # auto-finds data/airports.db
    info = db.lookup("KJFK")             # exact ICAO/FAA lookup
    results = db.search("Kennedy")       # fuzzy search by name or city
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default path: <project_root>/data/airports.db
_DEFAULT_DB_PATH: Path = Path(__file__).resolve().parents[2] / "data" / "airports.db"


class AirportDB:
    """Thread-safe wrapper around the local airports SQLite database."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._conn: sqlite3.Connection | None = None

        if not self._db_path.exists():
            logger.warning(
                "Airport database not found at %s. "
                "Run 'python tools/build_airport_db.py' to create it. "
                "Lookups will return None until the database is available.",
                self._db_path,
            )
            return

        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA query_only=ON")
        count = self._conn.execute("SELECT COUNT(*) FROM airports").fetchone()[0]
        logger.info("AirportDB loaded: %d airports from %s", count, self._db_path)

    @property
    def available(self) -> bool:
        """Return True if the database is loaded and ready for queries."""
        return self._conn is not None

    def lookup(self, identifier: str) -> dict[str, Any] | None:
        """Look up an airport by ICAO or FAA identifier.

        Auto-prefixes 3-letter codes with 'K' to match US ICAO conventions
        (e.g., "JFK" -> "KJFK").  Returns None if not found or DB unavailable.
        """
        if not self._conn:
            return None

        identifier = identifier.strip().upper()
        if not identifier.startswith("K") and len(identifier) == 3:
            identifier = f"K{identifier}"

        row = self._conn.execute(
            "SELECT * FROM airports WHERE ident = ?", (identifier,)
        ).fetchone()

        if row is None:
            return None

        result: dict[str, Any] = {
            "identifier": row["ident"],
            "name": row["name"] or "Unknown",
            "city": row["city"] or row["municipality"] or "",
            "state": row["state"] or "",
            "elevation": str(row["elevation_ft"] or ""),
            "latitude": str(row["latitude"] or ""),
            "longitude": str(row["longitude"] or ""),
            "runways": self._get_runways(identifier),
            "frequencies": self._get_frequencies(identifier),
        }
        return result

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search airports by name or city (case-insensitive substring match).

        Returns a list of airport dicts, ordered by relevance (exact ident match
        first, then name matches, then city matches).
        """
        if not self._conn:
            return []

        query = query.strip()
        if not query:
            return []

        pattern = f"%{query}%"

        # Try exact ident match first
        rows = self._conn.execute(
            "SELECT * FROM airports WHERE ident = ? LIMIT 1",
            (query.upper(),),
        ).fetchall()

        # Then name/city substring matches
        rows += self._conn.execute(
            """
            SELECT * FROM airports
            WHERE (name LIKE ? OR city LIKE ? OR municipality LIKE ?)
              AND ident != ?
            ORDER BY
                CASE
                    WHEN name LIKE ? THEN 0
                    WHEN city LIKE ? THEN 1
                    ELSE 2
                END,
                name
            LIMIT ?
            """,
            (pattern, pattern, pattern, query.upper(), pattern, pattern, limit),
        ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows[:limit]:
            results.append(
                {
                    "identifier": row["ident"],
                    "name": row["name"] or "Unknown",
                    "city": row["city"] or row["municipality"] or "",
                    "state": row["state"] or "",
                    "elevation": str(row["elevation_ft"] or ""),
                    "latitude": str(row["latitude"] or ""),
                    "longitude": str(row["longitude"] or ""),
                }
            )
        return results

    def _get_runways(self, airport_ident: str) -> list[dict[str, Any]]:
        """Fetch all runways for a given airport identifier."""
        if not self._conn:
            return []
        rows = self._conn.execute(
            """
            SELECT le_ident, he_ident, length_ft, width_ft, surface, lighted,
                   le_heading, he_heading
            FROM runways
            WHERE airport_ident = ?
            ORDER BY length_ft DESC
            """,
            (airport_ident,),
        ).fetchall()
        return [
            {
                "designator": (
                    f"{r['le_ident']}/{r['he_ident']}"
                    if r["le_ident"] and r["he_ident"]
                    else r["le_ident"] or r["he_ident"] or ""
                ),
                "length_ft": r["length_ft"],
                "width_ft": r["width_ft"],
                "surface": r["surface"] or "",
                "lighted": bool(r["lighted"]),
                "le_heading": r["le_heading"],
                "he_heading": r["he_heading"],
            }
            for r in rows
        ]

    def _get_frequencies(self, airport_ident: str) -> list[dict[str, Any]]:
        """Fetch all published frequencies for a given airport identifier."""
        if not self._conn:
            return []
        rows = self._conn.execute(
            """
            SELECT type, description, frequency_mhz
            FROM frequencies
            WHERE airport_ident = ?
            ORDER BY type, frequency_mhz
            """,
            (airport_ident,),
        ).fetchall()
        return [
            {
                "type": r["type"] or "",
                "description": r["description"] or "",
                "frequency_mhz": r["frequency_mhz"] or "",
            }
            for r in rows
        ]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
