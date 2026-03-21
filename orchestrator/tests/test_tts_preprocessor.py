"""Tests for the aviation TTS preprocessor."""

from __future__ import annotations

import pytest

from orchestrator.tts_preprocessor import preprocess_for_tts


# ---------------------------------------------------------------------------
# Flight levels
# ---------------------------------------------------------------------------


class TestFlightLevels:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("Climb to FL350", "Climb to flight level tree fife zero"),
            ("Descend FL180", "Descend flight level one eight zero"),
            ("Maintain FL045", "Maintain flight level zero four fife"),
            ("Cross FL240", "Cross flight level two four zero"),
            ("At FL410", "At flight level four one zero"),
        ],
    )
    def test_flight_level(self, input_text: str, expected: str) -> None:
        assert preprocess_for_tts(input_text) == expected


# ---------------------------------------------------------------------------
# Altitudes
# ---------------------------------------------------------------------------


class TestAltitudes:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("Descend to 3500ft", "Descend to three thousand five hundred feet"),
            (
                "Maintain 3,500 feet",
                "Maintain three thousand five hundred feet",
            ),
            ("Climb to 10000ft", "Climb to ten thousand feet"),
            ("At 500ft", "At five hundred feet"),
            ("Level at 8000ft", "Level at eight thousand feet"),
        ],
    )
    def test_altitude(self, input_text: str, expected: str) -> None:
        assert preprocess_for_tts(input_text) == expected


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------


class TestHeadings:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("Turn HDG 270", "Turn heading two seven zero"),
            ("heading 090", "heading zero niner zero"),
            ("Fly HDG 360", "Fly heading tree six zero"),
            ("HDG 180", "heading one eight zero"),
        ],
    )
    def test_heading(self, input_text: str, expected: str) -> None:
        assert preprocess_for_tts(input_text) == expected


# ---------------------------------------------------------------------------
# Frequencies
# ---------------------------------------------------------------------------


class TestFrequencies:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            (
                "Contact tower on 118.30",
                "Contact tower on one one eight point tree zero",
            ),
            (
                "Monitor frequency 121.5",
                "Monitor frequency one two one point fife",
            ),
            (
                "Tune 119.10",
                "Tune one one niner point one zero",
            ),
            (
                "Contact approach 124.35",
                "Contact approach one two four point tree fife",
            ),
            (
                "ATIS 127.85",
                "ATIS one two seven point eight fife",
            ),
        ],
    )
    def test_frequency(self, input_text: str, expected: str) -> None:
        assert preprocess_for_tts(input_text) == expected


# ---------------------------------------------------------------------------
# Runway designators
# ---------------------------------------------------------------------------


class TestRunways:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            (
                "Cleared to land RWY 27L",
                "Cleared to land runway two seven left",
            ),
            ("Taxi to RWY 09R", "Taxi to runway zero niner right"),
            ("Departing runway 36C", "Departing runway tree six center"),
            ("RWY 18", "runway one eight"),
            ("Line up RWY 04", "Line up runway zero four"),
        ],
    )
    def test_runway(self, input_text: str, expected: str) -> None:
        assert preprocess_for_tts(input_text) == expected


# ---------------------------------------------------------------------------
# Squawk codes
# ---------------------------------------------------------------------------


class TestSquawk:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("Squawk 7700", "squawk seven seven zero zero"),
            ("squawk 1200", "squawk one two zero zero"),
            ("Squawk 0420", "squawk zero four two zero"),
            ("squawk 7500", "squawk seven fife zero zero"),
        ],
    )
    def test_squawk(self, input_text: str, expected: str) -> None:
        assert preprocess_for_tts(input_text) == expected


# ---------------------------------------------------------------------------
# Speeds
# ---------------------------------------------------------------------------


class TestSpeeds:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("Reduce to 250kt", "Reduce to two hundred fifty knots"),
            ("Speed 180kts", "Speed one hundred eighty knots"),
            ("V1", "V one"),
            ("Rotate at Vr", "Rotate at V R"),
            ("V2 plus 10", "V two plus 10"),
        ],
    )
    def test_speed(self, input_text: str, expected: str) -> None:
        assert preprocess_for_tts(input_text) == expected


# ---------------------------------------------------------------------------
# QNH / Altimeter
# ---------------------------------------------------------------------------


class TestAltimeter:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("QNH 1013", "Q N H one zero one tree"),
            ("QNH 0992", "Q N H zero niner niner two"),
            ("Set 29.92 inHg", "Set two niner niner two inches"),
            ("Altimeter 30.12 inHg", "Altimeter tree zero one two inches"),
        ],
    )
    def test_altimeter(self, input_text: str, expected: str) -> None:
        assert preprocess_for_tts(input_text) == expected


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------


class TestDistance:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("5nm from the field", "five nautical miles from the field"),
            ("1 NM out", "one nautical mile out"),
            ("DME 12.3", "D M E one two point tree"),
            ("DME 5", "D M E fife"),
            ("At 20nm", "At twenty nautical miles"),
        ],
    )
    def test_distance(self, input_text: str, expected: str) -> None:
        assert preprocess_for_tts(input_text) == expected


# ---------------------------------------------------------------------------
# Markdown stripping
# ---------------------------------------------------------------------------


class TestMarkdown:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("**bold text**", "bold text"),
            ("*italic text*", "italic text"),
            ("~~struck out~~", "struck out"),
            ("### Heading Three", "Heading Three"),
            ("`inline code`", "inline code"),
            ("[link text](http://example.com)", "link text"),
        ],
    )
    def test_markdown_stripping(self, input_text: str, expected: str) -> None:
        assert preprocess_for_tts(input_text) == expected


# ---------------------------------------------------------------------------
# Special characters
# ---------------------------------------------------------------------------


class TestSpecialChars:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("5\u00b0 nose up", "5 degrees nose up"),
            ("A \u2014 pause", "A , pause"),
            ("3 \u2013 5", "3 to 5"),
            ("~200ft", "approximately two hundred feet"),
            ("A & B", "A and B"),
        ],
    )
    def test_special_chars(self, input_text: str, expected: str) -> None:
        assert preprocess_for_tts(input_text) == expected


# ---------------------------------------------------------------------------
# Combined / integration scenarios
# ---------------------------------------------------------------------------


class TestCombined:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            (
                "Descend to FL350 and contact tower on 118.30",
                "Descend to flight level tree fife zero and contact tower on"
                " one one eight point tree zero",
            ),
            (
                "Turn HDG 270, descend to 3500ft, squawk 1200",
                "Turn heading two seven zero, descend to three thousand"
                " five hundred feet, squawk one two zero zero",
            ),
            (
                "Cleared ILS RWY 27L, maintain 250kt until 5nm",
                "Cleared I L S runway two seven left, maintain two hundred"
                " fifty knots until five nautical miles",
            ),
        ],
    )
    def test_combined_scenarios(self, input_text: str, expected: str) -> None:
        assert preprocess_for_tts(input_text) == expected


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_string(self) -> None:
        assert preprocess_for_tts("") == ""

    def test_plain_text_passthrough(self) -> None:
        text = "Check your six, we have traffic."
        assert preprocess_for_tts(text) == text

    def test_multiple_spaces_collapsed(self) -> None:
        assert preprocess_for_tts("too   many   spaces") == "too many spaces"

    def test_newlines_become_sentence_breaks(self) -> None:
        result = preprocess_for_tts("Line one\nLine two")
        assert result == "Line one. Line two"

    def test_bullet_list_conversion(self) -> None:
        text = "Items:\n- First\n- Second"
        result = preprocess_for_tts(text)
        # Bullets become sentence-break pauses; newlines also become periods
        assert "First" in result
        assert "Second" in result
        assert "-" not in result


# ---------------------------------------------------------------------------
# Aviation acronyms
# ---------------------------------------------------------------------------


class TestAviationAcronyms:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("Check the IFR chart", "Check the I F R chart"),
            ("File VFR", "File V F R"),
            ("Altitude 500 AGL", "Altitude 500 A G L"),
            ("Altitude 3000 MSL", "Altitude 3000 M S L"),
            ("PIREP reported", "pilot report reported"),
            ("Fly the RNAV approach", "Fly the R NAV approach"),
            ("TCAS alert", "T CAS alert"),
            ("GPWS warning", "G P W S warning"),
            ("Tune the VOR", "Tune the V O R"),
            ("Cleared ILS approach", "Cleared I L S approach"),
            ("GPS direct", "G P S direct"),
        ],
    )
    def test_acronym_expansion(self, input_text: str, expected: str) -> None:
        assert preprocess_for_tts(input_text) == expected

    def test_pronounceable_acronyms_unchanged(self) -> None:
        """Acronyms that are already pronounceable should pass through."""
        for word in ("NOTAM", "SIGMET", "ATIS", "UNICOM", "SID", "STAR", "METAR", "TAF"):
            result = preprocess_for_tts(f"Check the {word}")
            assert word in result
