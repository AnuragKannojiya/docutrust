"""Tests for the structural chunker."""
from __future__ import annotations

from app.ingestion.chunker import chunk_sections, make_chunk_id


def test_short_section_emits_one_chunk():
    sections = [("intro", "Hello world.")]
    out = chunk_sections(sections, target_tokens=100, overlap_tokens=10)
    assert len(out) == 1
    assert out[0][0] == "intro"
    assert out[0][1] == "Hello world."


def test_long_section_is_windowed_with_overlap():
    body = " ".join(["alpha beta gamma delta"] * 200)  # ~800 tokens
    sections = [("long", body)]
    out = chunk_sections(sections, target_tokens=200, overlap_tokens=40)
    assert len(out) >= 2
    # Each chunk's text should be smaller than the source.
    for path, text in out:
        assert text
        assert path.startswith("long::chunk_")
    # And the chunks together should cover substantially all of the body.
    combined = " ".join(t for _, t in out)
    assert "alpha" in combined and "delta" in combined


def test_empty_sections_skipped():
    sections = [("a", ""), ("b", "   "), ("c", "real content")]
    out = chunk_sections(sections)
    assert [p for p, _ in out] == ["c"]


def test_make_chunk_id_is_stable():
    a = make_chunk_id("doc_abc", "section_path/with/slashes", 7)
    b = make_chunk_id("doc_abc", "section_path/with/slashes", 7)
    assert a == b
    assert a.startswith("doc_abc::")
    assert a.endswith("::7")
