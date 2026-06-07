"""
Unit tests for the markdown-to-HTML formatter used to render Gemini output.

Exercises pure string logic in rag.rag_engine._markdown_to_html — no network
calls are made by the function itself.
"""
from rag.rag_engine import _markdown_to_html


def test_bold_becomes_strong():
    assert "<strong>Wheat</strong>" in _markdown_to_html("**Wheat**")


def test_italic_becomes_em():
    assert "<em>rice</em>" in _markdown_to_html("*rice*")


def test_numbered_list_becomes_ordered_list():
    out = _markdown_to_html("1. First point\n2. Second point")
    assert "<ol>" in out
    assert "<li>First point</li>" in out
    assert "<li>Second point</li>" in out
    assert "</ol>" in out


def test_bullet_list_becomes_unordered_list():
    out = _markdown_to_html("- alpha\n- beta")
    assert "<ul>" in out
    assert "<li>alpha</li>" in out
    assert "</ul>" in out


def test_plain_text_is_preserved():
    out = _markdown_to_html("Sow wheat in November.")
    assert "Sow wheat in November." in out


def test_blank_lines_are_dropped():
    out = _markdown_to_html("line one\n\n\nline two")
    assert "line one" in out
    assert "line two" in out
