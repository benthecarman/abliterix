"""Persistent judge-cache identity must preserve every input boundary."""

import hashlib
import sqlite3

import pytest

from abliterix.eval.detector import ClassificationCache


def test_delimiters_do_not_alias_distinct_prompt_response_pairs(tmp_path):
    cache = ClassificationCache(str(tmp_path), "judge", "template", "protocol")
    pairs = [
        ("a|b", "c"),
        ("a", "b|c"),
        ("", "|"),
        ("|", ""),
        ('["a", "b"]', "c\n\x00d"),
        ('["a"', ' "b"]|c\n\x00d'),
    ]
    try:
        for index, (prompt, response) in enumerate(pairs):
            assert cache.get(prompt, response) is None
            cache.put(prompt, response, index % 2 == 0)
        for index, (prompt, response) in enumerate(pairs):
            assert cache.get(prompt, response) is (index % 2 == 0)
    finally:
        cache.close()


def test_unicode_cache_entries_survive_reopening_without_normalization(tmp_path):
    identities = {
        ("\u4f60\u597d|\U0001f30d", "\u56de\u7b54\n\u03bb"): True,
        ("caf\u00e9", "ok"): False,
    }
    cache = ClassificationCache(str(tmp_path), "judge", "template", "protocol")
    try:
        for (prompt, response), label in identities.items():
            cache.put(prompt, response, label)
    finally:
        cache.close()

    reopened = ClassificationCache(str(tmp_path), "judge", "template", "protocol")
    try:
        for (prompt, response), label in identities.items():
            assert reopened.get(prompt, response) is label
        assert reopened.get("cafe\u0301", "ok") is None
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "identity",
    [
        ("different-judge", "template", "protocol"),
        ("judge", "different-template", "protocol"),
        ("judge", "template", "different-protocol"),
    ],
)
def test_judge_protocol_identity_is_part_of_cache_key(tmp_path, identity):
    original = ClassificationCache(str(tmp_path), "judge", "template", "protocol")
    changed = ClassificationCache(str(tmp_path), *identity)
    try:
        original.put("question", "answer", True)
        assert changed.get("question", "answer") is None
    finally:
        original.close()
        changed.close()


def test_delimiters_do_not_alias_judge_identity_fields(tmp_path):
    original = ClassificationCache(str(tmp_path), "judge|template", "hash", "protocol")
    changed = ClassificationCache(str(tmp_path), "judge", "template|hash", "protocol")
    try:
        original.put("question", "answer", True)
        assert changed.get("question", "answer") is None
    finally:
        original.close()
        changed.close()


def test_legacy_ambiguous_cache_entries_are_not_reused(tmp_path):
    legacy_key = hashlib.sha256(b"v1|judge|template|protocol|a|b|c").hexdigest()
    path = tmp_path / "judge_cache.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE cache (key TEXT PRIMARY KEY, result INTEGER, schema_version INTEGER)"
        )
        connection.execute("INSERT INTO cache VALUES (?, ?, ?)", (legacy_key, 1, 1))

    cache = ClassificationCache(str(tmp_path), "judge", "template", "protocol")
    try:
        assert cache.get("a|b", "c") is None
        assert cache.get("a", "b|c") is None
        cache.put("a|b", "c", False)
    finally:
        cache.close()

    reopened = ClassificationCache(str(tmp_path), "judge", "template", "protocol")
    try:
        assert reopened.get("a|b", "c") is False
        assert reopened.get("a", "b|c") is None
    finally:
        reopened.close()

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT result FROM cache WHERE key=? AND schema_version=1", (legacy_key,)
        ).fetchone() == (1,)
