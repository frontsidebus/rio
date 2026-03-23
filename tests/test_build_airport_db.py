"""Tests for tools/build_airport_db.py -- CSV parsing, schema creation, and metadata."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# The build script lives outside any installed package, so we add it to the
# import path dynamically.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from build_airport_db import (  # noqa: E402
    SCHEMA_SQL,
    bulk_insert_airports,
    bulk_insert_frequencies,
    bulk_insert_runways,
    init_db,
    parse_airports,
    parse_frequencies,
    parse_runways,
)

# ---------------------------------------------------------------------------
# Sample CSV data (minimal valid rows from OurAirports format)
# ---------------------------------------------------------------------------

AIRPORTS_CSV = """\
id,ident,type,name,latitude_deg,longitude_deg,elevation_ft,continent,iso_country,iso_region,municipality,scheduled_service,gps_code,iata_code,local_code,home_link,wikipedia_link,keywords
3632,"KJFK","large_airport","John F Kennedy International Airport",40.6413,-73.7781,13,"NA","US","US-NY","New York","yes","KJFK","JFK","JFK","","",""
3484,"KLAX","large_airport","Los Angeles International Airport",33.9425,-118.408,128,"NA","US","US-CA","Los Angeles","yes","KLAX","LAX","LAX","","",""
3453,"KOSH","medium_airport","Wittman Regional Airport",43.9844,-88.557,808,"NA","US","US-WI","Oshkosh","no","KOSH","OSH","OSH","","",""
99999,"0FAKE","heliport","Some Heliport",0,0,0,"NA","US","US-TX","Nowhere","no","","","","","",""
"""

AIRPORTS_CSV_MALFORMED = """\
id,ident,type,name,latitude_deg,longitude_deg,elevation_ft,continent,iso_country,iso_region,municipality,scheduled_service,gps_code,iata_code,local_code,home_link,wikipedia_link,keywords
,"","large_airport","No Ident",,,,,,,,,,,,,,
3632,"X","large_airport","Single Char Ident",,,,,,,,,,,,,,
3632,"KJFK","large_airport","JFK",bad_lat,bad_lon,bad_elev,,,,,,,,,,,
"""

RUNWAYS_CSV = """\
id,airport_ref,airport_ident,length_ft,width_ft,surface,lighted,closed,le_ident,le_latitude_deg,le_longitude_deg,le_elevation_ft,le_heading_degT,le_displaced_threshold_ft,he_ident,he_latitude_deg,he_longitude_deg,he_elevation_ft,he_heading_degT,he_displaced_threshold_ft
1,"3632","KJFK",14511,200,"ASP",1,0,"13L",,,13,133,,"31R",,,13,313,
2,"3632","KJFK",12079,200,"ASP",1,0,"04L",,,13,42,,"22R",,,13,222,
3,"3453","KOSH",8002,150,"ASP",1,0,"18",,,808,180,,"36",,,808,360,
"""

RUNWAYS_CSV_MALFORMED = """\
id,airport_ref,airport_ident,length_ft,width_ft,surface,lighted,closed,le_ident,le_latitude_deg,le_longitude_deg,le_elevation_ft,le_heading_degT,le_displaced_threshold_ft,he_ident,he_latitude_deg,he_longitude_deg,he_elevation_ft,he_heading_degT,he_displaced_threshold_ft
1,,"",bad_len,bad_width,"ASP",1,0,"09",,,0,90,,"27",,,0,270,
"""

FREQUENCIES_CSV = """\
id,airport_ref,airport_ident,type,description,frequency_mhz
1,"3632","KJFK","TWR","Kennedy Tower","119.100"
2,"3632","KJFK","GND","Kennedy Ground","121.900"
3,"3453","KOSH","TWR","Oshkosh Tower","118.500"
"""

FREQUENCIES_CSV_MALFORMED = """\
id,airport_ref,airport_ident,type,description,frequency_mhz
1,,"","TWR","Ghost Tower","000.000"
"""


# ---------------------------------------------------------------------------
# CSV parsing tests
# ---------------------------------------------------------------------------


class TestParseAirports:
    """Test parse_airports against sample and malformed CSV data."""

    def test_parses_valid_airports(self) -> None:
        airports = parse_airports(AIRPORTS_CSV)
        assert len(airports) == 3  # heliport is filtered out
        idents = [a["ident"] for a in airports]
        assert "KJFK" in idents
        assert "KLAX" in idents
        assert "KOSH" in idents

    def test_filters_out_non_airport_types(self) -> None:
        airports = parse_airports(AIRPORTS_CSV)
        idents = [a["ident"] for a in airports]
        assert "0FAKE" not in idents

    def test_field_values(self) -> None:
        airports = parse_airports(AIRPORTS_CSV)
        jfk = next(a for a in airports if a["ident"] == "KJFK")
        assert jfk["name"] == "John F Kennedy International Airport"
        assert jfk["type"] == "large_airport"
        assert jfk["city"] == "New York"
        assert jfk["state"] == "NY"
        assert jfk["country"] == "US"
        assert jfk["elevation_ft"] == pytest.approx(13.0)
        assert jfk["latitude"] == pytest.approx(40.6413)
        assert jfk["longitude"] == pytest.approx(-73.7781)
        assert jfk["iso_region"] == "US-NY"

    def test_us_state_extraction(self) -> None:
        """US iso_region 'US-NY' should produce state 'NY'."""
        airports = parse_airports(AIRPORTS_CSV)
        jfk = next(a for a in airports if a["ident"] == "KJFK")
        assert jfk["state"] == "NY"

    def test_skips_empty_ident(self) -> None:
        airports = parse_airports(AIRPORTS_CSV_MALFORMED)
        idents = [a["ident"] for a in airports]
        assert "" not in idents

    def test_skips_short_ident(self) -> None:
        airports = parse_airports(AIRPORTS_CSV_MALFORMED)
        idents = [a["ident"] for a in airports]
        assert "X" not in idents

    def test_bad_numeric_fields_become_none(self) -> None:
        airports = parse_airports(AIRPORTS_CSV_MALFORMED)
        # KJFK row has bad_lat, bad_lon, bad_elev
        jfk = next((a for a in airports if a["ident"] == "KJFK"), None)
        assert jfk is not None
        assert jfk["latitude"] is None
        assert jfk["longitude"] is None
        assert jfk["elevation_ft"] is None

    def test_empty_csv(self) -> None:
        header_only = "id,ident,type,name,latitude_deg,longitude_deg,elevation_ft,continent,iso_country,iso_region,municipality,scheduled_service,gps_code,iata_code,local_code,home_link,wikipedia_link,keywords\n"
        assert parse_airports(header_only) == []


class TestParseRunways:
    """Test parse_runways against sample and malformed CSV data."""

    def test_parses_valid_runways(self) -> None:
        runways = parse_runways(RUNWAYS_CSV)
        assert len(runways) == 3

    def test_field_values(self) -> None:
        runways = parse_runways(RUNWAYS_CSV)
        jfk_13l = next(r for r in runways if r["le_ident"] == "13L")
        assert jfk_13l["airport_ident"] == "KJFK"
        assert jfk_13l["length_ft"] == pytest.approx(14511.0)
        assert jfk_13l["width_ft"] == pytest.approx(200.0)
        assert jfk_13l["surface"] == "ASP"
        assert jfk_13l["lighted"] == 1
        assert jfk_13l["he_ident"] == "31R"
        assert jfk_13l["le_heading"] == pytest.approx(133.0)
        assert jfk_13l["he_heading"] == pytest.approx(313.0)

    def test_skips_empty_airport_ident(self) -> None:
        runways = parse_runways(RUNWAYS_CSV_MALFORMED)
        idents = [r["airport_ident"] for r in runways]
        assert "" not in idents

    def test_bad_numeric_fields_become_none(self) -> None:
        runways = parse_runways(RUNWAYS_CSV_MALFORMED)
        # The malformed row has empty airport_ident and is skipped entirely
        assert len(runways) == 0

    def test_empty_csv(self) -> None:
        header_only = "id,airport_ref,airport_ident,length_ft,width_ft,surface,lighted,closed,le_ident,le_latitude_deg,le_longitude_deg,le_elevation_ft,le_heading_degT,le_displaced_threshold_ft,he_ident,he_latitude_deg,he_longitude_deg,he_elevation_ft,he_heading_degT,he_displaced_threshold_ft\n"
        assert parse_runways(header_only) == []


class TestParseFrequencies:
    """Test parse_frequencies against sample and malformed CSV data."""

    def test_parses_valid_frequencies(self) -> None:
        freqs = parse_frequencies(FREQUENCIES_CSV)
        assert len(freqs) == 3

    def test_field_values(self) -> None:
        freqs = parse_frequencies(FREQUENCIES_CSV)
        twr = next(f for f in freqs if f["description"] == "Kennedy Tower")
        assert twr["airport_ident"] == "KJFK"
        assert twr["type"] == "TWR"
        assert twr["frequency_mhz"] == "119.100"

    def test_skips_empty_airport_ident(self) -> None:
        freqs = parse_frequencies(FREQUENCIES_CSV_MALFORMED)
        idents = [f["airport_ident"] for f in freqs]
        assert "" not in idents

    def test_empty_csv(self) -> None:
        header_only = "id,airport_ref,airport_ident,type,description,frequency_mhz\n"
        assert parse_frequencies(header_only) == []


# ---------------------------------------------------------------------------
# Database creation tests
# ---------------------------------------------------------------------------


class TestInitDB:
    """Test SQLite database creation and schema application."""

    def test_creates_database_file(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        conn = init_db(db_path)
        assert Path(db_path).exists()
        conn.close()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "nested" / "dir" / "test.db")
        conn = init_db(db_path)
        assert Path(db_path).exists()
        conn.close()

    def test_airports_table_exists(self, tmp_path: Path) -> None:
        conn = init_db(str(tmp_path / "test.db"))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "airports" in table_names
        assert "runways" in table_names
        assert "frequencies" in table_names
        assert "metadata" in table_names
        conn.close()

    def test_indexes_created(self, tmp_path: Path) -> None:
        conn = init_db(str(tmp_path / "test.db"))
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ).fetchall()
        index_names = [i[0] for i in indexes]
        assert "idx_airports_ident" in index_names
        assert "idx_runways_airport" in index_names
        assert "idx_freq_airport" in index_names
        conn.close()

    def test_wal_journal_mode(self, tmp_path: Path) -> None:
        conn = init_db(str(tmp_path / "test.db"))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
        conn.close()


# ---------------------------------------------------------------------------
# Bulk insert tests
# ---------------------------------------------------------------------------


class TestBulkInserts:
    """Test bulk insert operations for airports, runways, and frequencies."""

    def test_bulk_insert_airports(self, tmp_path: Path) -> None:
        conn = init_db(str(tmp_path / "test.db"))
        airports = parse_airports(AIRPORTS_CSV)
        count = bulk_insert_airports(conn, airports)
        assert count == 3

        rows = conn.execute("SELECT COUNT(*) FROM airports").fetchone()[0]
        assert rows == 3
        conn.close()

    def test_bulk_insert_airports_replace(self, tmp_path: Path) -> None:
        """INSERT OR REPLACE should update existing rows, not duplicate."""
        conn = init_db(str(tmp_path / "test.db"))
        airports = parse_airports(AIRPORTS_CSV)
        bulk_insert_airports(conn, airports)
        # Insert again -- should replace, not duplicate
        bulk_insert_airports(conn, airports)
        rows = conn.execute("SELECT COUNT(*) FROM airports").fetchone()[0]
        assert rows == 3
        conn.close()

    def test_bulk_insert_runways(self, tmp_path: Path) -> None:
        conn = init_db(str(tmp_path / "test.db"))
        runways = parse_runways(RUNWAYS_CSV)
        count = bulk_insert_runways(conn, runways)
        assert count == 3

        rows = conn.execute("SELECT COUNT(*) FROM runways").fetchone()[0]
        assert rows == 3
        conn.close()

    def test_bulk_insert_runways_clears_previous(self, tmp_path: Path) -> None:
        """bulk_insert_runways DELETEs before inserting."""
        conn = init_db(str(tmp_path / "test.db"))
        runways = parse_runways(RUNWAYS_CSV)
        bulk_insert_runways(conn, runways)
        # Insert again -- should clear + re-insert, not accumulate
        bulk_insert_runways(conn, runways)
        rows = conn.execute("SELECT COUNT(*) FROM runways").fetchone()[0]
        assert rows == 3
        conn.close()

    def test_bulk_insert_frequencies(self, tmp_path: Path) -> None:
        conn = init_db(str(tmp_path / "test.db"))
        freqs = parse_frequencies(FREQUENCIES_CSV)
        count = bulk_insert_frequencies(conn, freqs)
        assert count == 3

        rows = conn.execute("SELECT COUNT(*) FROM frequencies").fetchone()[0]
        assert rows == 3
        conn.close()

    def test_bulk_insert_frequencies_clears_previous(self, tmp_path: Path) -> None:
        conn = init_db(str(tmp_path / "test.db"))
        freqs = parse_frequencies(FREQUENCIES_CSV)
        bulk_insert_frequencies(conn, freqs)
        bulk_insert_frequencies(conn, freqs)
        rows = conn.execute("SELECT COUNT(*) FROM frequencies").fetchone()[0]
        assert rows == 3
        conn.close()


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------


class TestMetadata:
    """Test that the metadata table is populated by the build pipeline."""

    def test_metadata_populated(self, tmp_path: Path) -> None:
        """Simulate a mini-build and verify metadata rows."""
        from build_airport_db import download_and_build

        db_path = str(tmp_path / "test.db")

        # Mock _download_cached to return our sample CSVs
        def fake_download(url: str, cache_dir: str = "") -> str:
            if "airports.csv" in url:
                return AIRPORTS_CSV
            elif "runways.csv" in url:
                return RUNWAYS_CSV
            elif "airport-frequencies.csv" in url:
                return FREQUENCIES_CSV
            raise ValueError(f"Unexpected URL: {url}")

        with patch("build_airport_db._download_cached", side_effect=fake_download):
            download_and_build(db_path)

        conn = sqlite3.connect(db_path)
        meta = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
        assert "last_updated" in meta
        assert "source" in meta
        assert "OurAirports" in meta["source"]
        conn.close()

    def test_full_pipeline_row_counts(self, tmp_path: Path) -> None:
        """End-to-end: mock downloads, build DB, verify row counts."""
        from build_airport_db import download_and_build

        db_path = str(tmp_path / "test.db")

        def fake_download(url: str, cache_dir: str = "") -> str:
            if "airports.csv" in url:
                return AIRPORTS_CSV
            elif "runways.csv" in url:
                return RUNWAYS_CSV
            elif "airport-frequencies.csv" in url:
                return FREQUENCIES_CSV
            raise ValueError(f"Unexpected URL: {url}")

        with patch("build_airport_db._download_cached", side_effect=fake_download):
            download_and_build(db_path)

        conn = sqlite3.connect(db_path)
        assert conn.execute("SELECT COUNT(*) FROM airports").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM runways").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM frequencies").fetchone()[0] == 3
        conn.close()
