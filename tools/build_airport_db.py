#!/usr/bin/env python3
"""Build the local airport SQLite database from OurAirports CSV data.

Downloads airport, runway, and frequency data from the OurAirports public-domain
data set and builds a compact SQLite database for offline airport lookups by the
MERLIN orchestrator.  This replaces the external aviationapi.com HTTP dependency.

Usage:
    python build_airport_db.py                   # download + build
    python build_airport_db.py --refresh         # re-download & rebuild
    python build_airport_db.py --db ./custom.db  # custom output path
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import httpx

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("merlin.airport_db")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "airports.db",
)

OURAIRPORTS_AIRPORTS_URL: str = (
    "https://davidmegginson.github.io/ourairports-data/airports.csv"
)
OURAIRPORTS_RUNWAYS_URL: str = (
    "https://davidmegginson.github.io/ourairports-data/runways.csv"
)
OURAIRPORTS_FREQUENCIES_URL: str = (
    "https://davidmegginson.github.io/ourairports-data/airport-frequencies.csv"
)

CACHE_DIR: str = os.path.join(tempfile.gettempdir(), "merlin_airport_cache")
HTTP_TIMEOUT: int = 120


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS airports (
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

CREATE TABLE IF NOT EXISTS runways (
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

CREATE TABLE IF NOT EXISTS frequencies (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    airport_ident   TEXT NOT NULL,
    type            TEXT,
    description     TEXT,
    frequency_mhz   TEXT,
    FOREIGN KEY (airport_ident) REFERENCES airports(ident)
);

CREATE INDEX IF NOT EXISTS idx_airports_ident ON airports(ident);
CREATE INDEX IF NOT EXISTS idx_runways_airport ON runways(airport_ident);
CREATE INDEX IF NOT EXISTS idx_freq_airport ON frequencies(airport_ident);

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _download_cached(url: str, cache_dir: str = CACHE_DIR) -> str:
    """Download a URL, caching the result on disk. Returns the text content."""
    filename = url.rsplit("/", 1)[-1]
    cache_path = Path(cache_dir) / filename
    if cache_path.exists():
        log.info("  Using cached %s", cache_path.name)
        return cache_path.read_text(encoding="utf-8", errors="replace")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("  Downloading %s ...", url)
    with httpx.stream("GET", url, follow_redirects=True, timeout=HTTP_TIMEOUT) as resp:
        resp.raise_for_status()
        with open(cache_path, "wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=65536):
                fh.write(chunk)
    size_mb = cache_path.stat().st_size / 1_048_576
    log.info("  Saved %s (%.1f MB)", cache_path.name, size_mb)
    return cache_path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def _safe_float(val: str | None) -> float | None:
    """Convert a string to float, returning None on failure."""
    if val is None:
        return None
    val = val.strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def parse_airports(text: str) -> list[dict]:
    """Parse the OurAirports airports.csv into a list of row dicts."""
    reader = csv.DictReader(io.StringIO(text))
    results: list[dict] = []
    for row in reader:
        ident = (row.get("ident") or "").strip()
        if not ident or len(ident) < 2:
            continue
        atype = (row.get("type") or "").strip()
        if atype not in ("large_airport", "medium_airport", "small_airport", "seaplane_base"):
            continue
        iso_region = (row.get("iso_region") or "").strip()
        state = iso_region.replace("US-", "") if iso_region.startswith("US-") else iso_region
        results.append(
            {
                "ident": ident,
                "name": (row.get("name") or "").strip(),
                "type": atype,
                "city": (row.get("municipality") or "").strip(),
                "state": state,
                "country": (row.get("iso_country") or "").strip(),
                "elevation_ft": _safe_float(row.get("elevation_ft")),
                "latitude": _safe_float(row.get("latitude_deg")),
                "longitude": _safe_float(row.get("longitude_deg")),
                "iso_region": iso_region,
                "municipality": (row.get("municipality") or "").strip(),
            }
        )
    return results


def parse_runways(text: str) -> list[dict]:
    """Parse the OurAirports runways.csv into a list of row dicts."""
    reader = csv.DictReader(io.StringIO(text))
    results: list[dict] = []
    for row in reader:
        airport_ident = (row.get("airport_ident") or "").strip()
        if not airport_ident:
            continue
        results.append(
            {
                "airport_ident": airport_ident,
                "length_ft": _safe_float(row.get("length_ft")),
                "width_ft": _safe_float(row.get("width_ft")),
                "surface": (row.get("surface") or "").strip(),
                "lighted": 1 if row.get("lighted") == "1" else 0,
                "le_ident": (row.get("le_ident") or "").strip(),
                "he_ident": (row.get("he_ident") or "").strip(),
                "le_heading": _safe_float(row.get("le_heading_degT")),
                "he_heading": _safe_float(row.get("he_heading_degT")),
            }
        )
    return results


def parse_frequencies(text: str) -> list[dict]:
    """Parse the OurAirports airport-frequencies.csv into a list of row dicts."""
    reader = csv.DictReader(io.StringIO(text))
    results: list[dict] = []
    for row in reader:
        airport_ident = (row.get("airport_ident") or "").strip()
        if not airport_ident:
            continue
        results.append(
            {
                "airport_ident": airport_ident,
                "type": (row.get("type") or "").strip(),
                "description": (row.get("description") or "").strip(),
                "frequency_mhz": (row.get("frequency_mhz") or "").strip(),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------
def init_db(db_path: str) -> sqlite3.Connection:
    """Create the SQLite database file and apply the schema."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def bulk_insert_airports(conn: sqlite3.Connection, airports: list[dict]) -> int:
    """Insert or replace airport rows. Returns the count inserted."""
    sql = """
        INSERT OR REPLACE INTO airports
            (ident, name, type, city, state, country, elevation_ft,
             latitude, longitude, iso_region, municipality)
        VALUES
            (:ident, :name, :type, :city, :state, :country, :elevation_ft,
             :latitude, :longitude, :iso_region, :municipality)
    """
    conn.executemany(sql, airports)
    conn.commit()
    return len(airports)


def bulk_insert_runways(conn: sqlite3.Connection, runways: list[dict]) -> int:
    """Clear and re-insert all runway rows. Returns the count inserted."""
    conn.execute("DELETE FROM runways")
    sql = """
        INSERT INTO runways
            (airport_ident, length_ft, width_ft, surface, lighted,
             le_ident, he_ident, le_heading, he_heading)
        VALUES
            (:airport_ident, :length_ft, :width_ft, :surface, :lighted,
             :le_ident, :he_ident, :le_heading, :he_heading)
    """
    conn.executemany(sql, runways)
    conn.commit()
    return len(runways)


def bulk_insert_frequencies(conn: sqlite3.Connection, freqs: list[dict]) -> int:
    """Clear and re-insert all frequency rows. Returns the count inserted."""
    conn.execute("DELETE FROM frequencies")
    sql = """
        INSERT INTO frequencies (airport_ident, type, description, frequency_mhz)
        VALUES (:airport_ident, :type, :description, :frequency_mhz)
    """
    conn.executemany(sql, freqs)
    conn.commit()
    return len(freqs)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def download_and_build(db_path: str, refresh: bool = False) -> None:
    """Download OurAirports CSVs and build the airports.db SQLite database."""
    if refresh:
        cache = Path(CACHE_DIR)
        if cache.exists():
            for f in cache.iterdir():
                f.unlink()
            log.info("Cache cleared.")

    log.info("Step 1/4: Downloading airport data ...")
    airports_csv = _download_cached(OURAIRPORTS_AIRPORTS_URL)
    airports = parse_airports(airports_csv)
    log.info("  Parsed %d airports.", len(airports))

    log.info("Step 2/4: Downloading runway data ...")
    runways_csv = _download_cached(OURAIRPORTS_RUNWAYS_URL)
    runways = parse_runways(runways_csv)
    log.info("  Parsed %d runways.", len(runways))

    log.info("Step 3/4: Downloading frequency data ...")
    freq_csv = _download_cached(OURAIRPORTS_FREQUENCIES_URL)
    freqs = parse_frequencies(freq_csv)
    log.info("  Parsed %d frequencies.", len(freqs))

    log.info("Step 4/4: Building SQLite database at %s ...", db_path)
    conn = init_db(db_path)
    n_apt = bulk_insert_airports(conn, airports)
    n_rwy = bulk_insert_runways(conn, runways)
    n_frq = bulk_insert_frequencies(conn, freqs)

    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("last_updated", datetime.now(timezone.utc).isoformat()),
    )
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("source", "OurAirports (davidmegginson.github.io/ourairports-data)"),
    )
    conn.commit()
    conn.close()

    log.info(
        "Done: %d airports, %d runways, %d frequencies -> %s",
        n_apt,
        n_rwy,
        n_frq,
        db_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="build_airport_db",
        description="Download OurAirports data and build the MERLIN airport database.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to the output SQLite database (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download of all source CSVs (bust cache).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    download_and_build(args.db, refresh=args.refresh)


if __name__ == "__main__":
    main()
