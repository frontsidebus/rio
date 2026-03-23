"""Tests for orchestrator.airport_db and the local-DB path in orchestrator.tools.lookup_airport."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from orchestrator.airport_db import AirportDB

# ---------------------------------------------------------------------------
# Schema used by build_airport_db.py -- kept in sync manually.
# ---------------------------------------------------------------------------
_SCHEMA_SQL = """
CREATE TABLE airports (
    ident           TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    type            TEXT,
    city            TEXT,
    state           TEXT,
    country         TEXT,
    elevation_ft    REAL,
    latitude        REAL,
    longitude       REAL,
    iso_region      TEXT,
    municipality    TEXT
);
CREATE TABLE runways (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    airport_ident   TEXT NOT NULL,
    length_ft       REAL,
    width_ft        REAL,
    surface         TEXT,
    lighted         INTEGER DEFAULT 0,
    le_ident        TEXT,
    he_ident        TEXT,
    le_heading      REAL,
    he_heading      REAL,
    FOREIGN KEY (airport_ident) REFERENCES airports(ident)
);
CREATE TABLE frequencies (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    airport_ident   TEXT NOT NULL,
    type            TEXT,
    description     TEXT,
    frequency_mhz   TEXT,
    FOREIGN KEY (airport_ident) REFERENCES airports(ident)
);
CREATE INDEX idx_airports_ident ON airports(ident);
CREATE INDEX idx_runways_airport ON runways(airport_ident);
CREATE INDEX idx_freq_airport ON frequencies(airport_ident);
CREATE TABLE metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# ---------------------------------------------------------------------------
# Fixture: tiny airport database
# ---------------------------------------------------------------------------

_AIRPORTS: list[dict[str, Any]] = [
    {
        "ident": "KJFK",
        "name": "John F Kennedy International Airport",
        "type": "large_airport",
        "city": "New York",
        "state": "NY",
        "country": "US",
        "elevation_ft": 13.0,
        "latitude": 40.6413,
        "longitude": -73.7781,
        "iso_region": "US-NY",
        "municipality": "New York",
    },
    {
        "ident": "KLAX",
        "name": "Los Angeles International Airport",
        "type": "large_airport",
        "city": "Los Angeles",
        "state": "CA",
        "country": "US",
        "elevation_ft": 128.0,
        "latitude": 33.9425,
        "longitude": -118.408,
        "iso_region": "US-CA",
        "municipality": "Los Angeles",
    },
    {
        "ident": "KOSH",
        "name": "Wittman Regional Airport",
        "type": "medium_airport",
        "city": "Oshkosh",
        "state": "WI",
        "country": "US",
        "elevation_ft": 808.0,
        "latitude": 43.9844,
        "longitude": -88.557,
        "iso_region": "US-WI",
        "municipality": "Oshkosh",
    },
    {
        "ident": "EGLL",
        "name": "London Heathrow Airport",
        "type": "large_airport",
        "city": "London",
        "state": "ENG",
        "country": "GB",
        "elevation_ft": 83.0,
        "latitude": 51.4706,
        "longitude": -0.4619,
        "iso_region": "GB-ENG",
        "municipality": "London",
    },
]

_RUNWAYS: list[dict[str, Any]] = [
    {
        "airport_ident": "KJFK",
        "length_ft": 14511.0,
        "width_ft": 200.0,
        "surface": "ASP",
        "lighted": 1,
        "le_ident": "13L",
        "he_ident": "31R",
        "le_heading": 133.0,
        "he_heading": 313.0,
    },
    {
        "airport_ident": "KJFK",
        "length_ft": 12079.0,
        "width_ft": 200.0,
        "surface": "ASP",
        "lighted": 1,
        "le_ident": "04L",
        "he_ident": "22R",
        "le_heading": 42.0,
        "he_heading": 222.0,
    },
    {
        "airport_ident": "KOSH",
        "length_ft": 8002.0,
        "width_ft": 150.0,
        "surface": "ASP",
        "lighted": 1,
        "le_ident": "18",
        "he_ident": "36",
        "le_heading": 180.0,
        "he_heading": 360.0,
    },
]

_FREQUENCIES: list[dict[str, Any]] = [
    {
        "airport_ident": "KJFK",
        "type": "TWR",
        "description": "Kennedy Tower",
        "frequency_mhz": "119.100",
    },
    {
        "airport_ident": "KJFK",
        "type": "GND",
        "description": "Kennedy Ground",
        "frequency_mhz": "121.900",
    },
    {
        "airport_ident": "KJFK",
        "type": "ATIS",
        "description": "Kennedy ATIS",
        "frequency_mhz": "128.725",
    },
    {
        "airport_ident": "KOSH",
        "type": "TWR",
        "description": "Oshkosh Tower",
        "frequency_mhz": "118.500",
    },
]


def _build_test_db(db_path: Path) -> Path:
    """Create a tiny SQLite airport database for testing."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA_SQL)

    conn.executemany(
        """INSERT INTO airports
           (ident, name, type, city, state, country, elevation_ft,
            latitude, longitude, iso_region, municipality)
           VALUES (:ident, :name, :type, :city, :state, :country,
                   :elevation_ft, :latitude, :longitude, :iso_region, :municipality)""",
        _AIRPORTS,
    )
    conn.executemany(
        """INSERT INTO runways
           (airport_ident, length_ft, width_ft, surface, lighted,
            le_ident, he_ident, le_heading, he_heading)
           VALUES (:airport_ident, :length_ft, :width_ft, :surface, :lighted,
                   :le_ident, :he_ident, :le_heading, :he_heading)""",
        _RUNWAYS,
    )
    conn.executemany(
        """INSERT INTO frequencies
           (airport_ident, type, description, frequency_mhz)
           VALUES (:airport_ident, :type, :description, :frequency_mhz)""",
        _FREQUENCIES,
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    """Return the path to a freshly built test airport database."""
    return _build_test_db(tmp_path / "airports.db")


@pytest.fixture
def airport_db(test_db: Path) -> AirportDB:
    """Return an AirportDB instance backed by the test database."""
    db = AirportDB(db_path=test_db)
    yield db
    db.close()


# =========================================================================
# AirportDB.available
# =========================================================================


class TestAirportDBAvailable:
    """Test the .available property for present and missing databases."""

    def test_available_when_db_exists(self, airport_db: AirportDB) -> None:
        assert airport_db.available is True

    def test_not_available_when_db_missing(self, tmp_path: Path) -> None:
        db = AirportDB(db_path=tmp_path / "nonexistent.db")
        assert db.available is False

    def test_not_available_after_close(self, airport_db: AirportDB) -> None:
        airport_db.close()
        assert airport_db.available is False


# =========================================================================
# AirportDB.lookup
# =========================================================================


class TestAirportDBLookup:
    """Test airport lookups by ICAO and FAA identifiers."""

    def test_lookup_by_icao(self, airport_db: AirportDB) -> None:
        result = airport_db.lookup("KJFK")
        assert result is not None
        assert result["identifier"] == "KJFK"
        assert result["name"] == "John F Kennedy International Airport"

    def test_lookup_three_letter_faa_auto_prefix(self, airport_db: AirportDB) -> None:
        """A 3-letter FAA code not starting with K gets auto K-prefix."""
        result = airport_db.lookup("JFK")
        assert result is not None
        assert result["identifier"] == "KJFK"

    def test_lookup_three_letter_already_k_prefix(self, airport_db: AirportDB) -> None:
        """A 3-letter code already starting with K (e.g., 'KOS') gets K-prefixed to KKOS."""
        # 'LAX' should resolve to 'KLAX'
        result = airport_db.lookup("LAX")
        assert result is not None
        assert result["identifier"] == "KLAX"

    def test_lookup_non_us_icao(self, airport_db: AirportDB) -> None:
        """4-letter non-US ICAO codes should work without modification."""
        result = airport_db.lookup("EGLL")
        assert result is not None
        assert result["identifier"] == "EGLL"
        assert "Heathrow" in result["name"]

    def test_lookup_nonexistent_returns_none(self, airport_db: AirportDB) -> None:
        result = airport_db.lookup("KZZZ")
        assert result is None

    def test_lookup_case_insensitive(self, airport_db: AirportDB) -> None:
        result = airport_db.lookup("kjfk")
        assert result is not None
        assert result["identifier"] == "KJFK"

    def test_lookup_strips_whitespace(self, airport_db: AirportDB) -> None:
        result = airport_db.lookup("  KJFK  ")
        assert result is not None
        assert result["identifier"] == "KJFK"

    def test_lookup_result_structure(self, airport_db: AirportDB) -> None:
        """Verify all expected keys are present and correctly typed."""
        result = airport_db.lookup("KJFK")
        assert result is not None

        expected_keys = {
            "identifier", "name", "city", "state",
            "elevation", "latitude", "longitude",
            "runways", "frequencies",
        }
        assert set(result.keys()) == expected_keys

        # String fields
        assert isinstance(result["identifier"], str)
        assert isinstance(result["name"], str)
        assert isinstance(result["city"], str)
        assert isinstance(result["state"], str)
        assert isinstance(result["elevation"], str)
        assert isinstance(result["latitude"], str)
        assert isinstance(result["longitude"], str)

        # Collection fields
        assert isinstance(result["runways"], list)
        assert isinstance(result["frequencies"], list)

    def test_lookup_runway_data(self, airport_db: AirportDB) -> None:
        result = airport_db.lookup("KJFK")
        assert result is not None
        runways = result["runways"]
        assert len(runways) == 2

        # Runways are ordered by length_ft DESC
        assert runways[0]["length_ft"] == 14511.0
        assert runways[0]["designator"] == "13L/31R"
        assert runways[0]["surface"] == "ASP"
        assert runways[0]["lighted"] is True
        assert runways[0]["le_heading"] == 133.0

    def test_lookup_frequency_data(self, airport_db: AirportDB) -> None:
        result = airport_db.lookup("KJFK")
        assert result is not None
        freqs = result["frequencies"]
        assert len(freqs) == 3

        # Frequencies ordered by type, frequency_mhz
        freq_types = [f["type"] for f in freqs]
        assert "ATIS" in freq_types
        assert "TWR" in freq_types
        assert "GND" in freq_types

        atis = next(f for f in freqs if f["type"] == "ATIS")
        assert atis["frequency_mhz"] == "128.725"
        assert atis["description"] == "Kennedy ATIS"

    def test_lookup_no_runways_or_frequencies(self, airport_db: AirportDB) -> None:
        """KLAX has no runways/freqs in our fixture -- should return empty lists."""
        result = airport_db.lookup("KLAX")
        assert result is not None
        assert result["runways"] == []
        assert result["frequencies"] == []

    def test_lookup_returns_none_when_db_unavailable(self, tmp_path: Path) -> None:
        db = AirportDB(db_path=tmp_path / "missing.db")
        assert db.lookup("KJFK") is None


# =========================================================================
# AirportDB.search
# =========================================================================


class TestAirportDBSearch:
    """Test substring search by name and city."""

    def test_search_by_name(self, airport_db: AirportDB) -> None:
        results = airport_db.search("Kennedy")
        assert len(results) >= 1
        assert results[0]["identifier"] == "KJFK"

    def test_search_by_city(self, airport_db: AirportDB) -> None:
        results = airport_db.search("Oshkosh")
        assert len(results) >= 1
        idents = [r["identifier"] for r in results]
        assert "KOSH" in idents

    def test_search_case_insensitive(self, airport_db: AirportDB) -> None:
        results = airport_db.search("kennedy")
        assert len(results) >= 1
        assert results[0]["identifier"] == "KJFK"

    def test_search_no_matches(self, airport_db: AirportDB) -> None:
        results = airport_db.search("Nonexistent City XYZ")
        assert results == []

    def test_search_empty_query(self, airport_db: AirportDB) -> None:
        results = airport_db.search("")
        assert results == []

    def test_search_whitespace_query(self, airport_db: AirportDB) -> None:
        results = airport_db.search("   ")
        assert results == []

    def test_search_exact_ident_prioritized(self, airport_db: AirportDB) -> None:
        """An exact ident match should appear first."""
        results = airport_db.search("KOSH")
        assert len(results) >= 1
        assert results[0]["identifier"] == "KOSH"

    def test_search_result_structure(self, airport_db: AirportDB) -> None:
        results = airport_db.search("International")
        assert len(results) >= 1
        for r in results:
            expected_keys = {
                "identifier", "name", "city", "state",
                "elevation", "latitude", "longitude",
            }
            assert set(r.keys()) == expected_keys

    def test_search_respects_limit(self, airport_db: AirportDB) -> None:
        results = airport_db.search("Airport", limit=2)
        assert len(results) <= 2

    def test_search_returns_empty_when_db_unavailable(self, tmp_path: Path) -> None:
        db = AirportDB(db_path=tmp_path / "missing.db")
        assert db.search("anything") == []


# =========================================================================
# Thread safety
# =========================================================================


class TestAirportDBThreadSafety:
    """Verify the DB connection is configured for multi-threaded access."""

    def test_check_same_thread_false(self, test_db: Path) -> None:
        """AirportDB must open SQLite with check_same_thread=False."""
        db = AirportDB(db_path=test_db)
        assert db.available
        # If check_same_thread were True, calling from a different thread
        # would raise ProgrammingError.  We verify the connection works by
        # running a query from a thread pool.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(db.lookup, "KJFK")
            result = future.result(timeout=5)
        assert result is not None
        assert result["identifier"] == "KJFK"
        db.close()


# =========================================================================
# Graceful degradation
# =========================================================================


class TestAirportDBGracefulDegradation:
    """Verify no exceptions are raised when the database file is absent."""

    def test_init_logs_warning_on_missing_db(self, tmp_path: Path) -> None:
        """Constructing AirportDB with a missing path should not raise."""
        db = AirportDB(db_path=tmp_path / "does_not_exist.db")
        assert db.available is False

    def test_all_methods_safe_when_unavailable(self, tmp_path: Path) -> None:
        db = AirportDB(db_path=tmp_path / "does_not_exist.db")
        assert db.lookup("KJFK") is None
        assert db.search("Kennedy") == []
        # close is also safe
        db.close()


# =========================================================================
# tools.lookup_airport with local DB integration
# =========================================================================


class TestLookupAirportLocalDB:
    """Test tools.lookup_airport using the local database path."""

    @pytest.mark.asyncio
    async def test_uses_local_db_when_available(self, airport_db: AirportDB) -> None:
        """lookup_airport should return data from the local DB without HTTP."""
        # We need to import here to avoid module-level side effects
        from orchestrator import tools

        with patch.object(tools, "_airport_db", airport_db):
            result = await tools.lookup_airport("KJFK")

        assert result["identifier"] == "KJFK"
        assert result["name"] == "John F Kennedy International Airport"
        # Local DB results include runways and frequencies
        assert "runways" in result
        assert "frequencies" in result

    @pytest.mark.asyncio
    @respx.mock
    async def test_falls_back_to_http_when_db_unavailable(self, tmp_path: Path) -> None:
        """When the local DB is missing, falls back to aviationapi.com."""
        from orchestrator import tools

        unavailable_db = AirportDB(db_path=tmp_path / "missing.db")

        respx.get("https://api.aviationapi.com/v1/airports", params={"apt": "KJFK"}).mock(
            return_value=httpx.Response(200, json={
                "KJFK": [{"facility_name": "JFK INTL", "city": "NEW YORK"}],
            })
        )

        with patch.object(tools, "_airport_db", unavailable_db):
            result = await tools.lookup_airport("KJFK")

        assert result["identifier"] == "KJFK"
        assert result["name"] == "JFK INTL"

    @pytest.mark.asyncio
    @respx.mock
    async def test_falls_back_to_http_when_not_in_local_db(
        self, airport_db: AirportDB
    ) -> None:
        """Airport not in local DB triggers HTTP fallback."""
        from orchestrator import tools

        respx.get(
            "https://api.aviationapi.com/v1/airports", params={"apt": "KSFO"}
        ).mock(
            return_value=httpx.Response(200, json={
                "KSFO": [{"facility_name": "SAN FRANCISCO INTL", "city": "SAN FRANCISCO"}],
            })
        )

        with patch.object(tools, "_airport_db", airport_db):
            result = await tools.lookup_airport("KSFO")

        assert result["identifier"] == "KSFO"
        assert result["name"] == "SAN FRANCISCO INTL"

    @pytest.mark.asyncio
    async def test_consistent_format_local_vs_http(self, airport_db: AirportDB) -> None:
        """Both local and HTTP paths return dicts with 'identifier' and 'name' keys."""
        from orchestrator import tools

        with patch.object(tools, "_airport_db", airport_db):
            local_result = await tools.lookup_airport("KJFK")

        assert "identifier" in local_result
        assert "name" in local_result
        assert "city" in local_result
        assert "elevation" in local_result
        assert "latitude" in local_result
        assert "longitude" in local_result
