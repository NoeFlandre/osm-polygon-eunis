"""Streaming geometry-first filtering primitives for the EUNIS census helper."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeGuard


class Assignment(Protocol):
    """The assignment fields needed to gate sentence retrieval."""

    code: str | None
    name: str | None
    overlap_percentage: float | None
    source_version: str | None


GeometryRow = Mapping[str, object]
TextFetcher = Callable[[Sequence[GeometryRow]], Mapping[str, str | None]]
Classifier = Callable[[object], Assignment | None]
MINIMUM_NATIVE_OVERLAP_PERCENTAGE = 80.0


@dataclass(slots=True)
class CensusCounts:
    """Bounded per-region counters emitted with a census checkpoint."""

    geometry_rows: int = 0
    native_eligible_rows: int = 0
    invalid_geometry_rows: int = 0
    outside_native_extent_rows: int = 0
    projection_invalid_rows: int = 0
    native_overlap_errors: int = 0
    ambiguous_native_rows: int = 0
    unassigned_native_rows: int = 0
    text_fetch_rows: int = 0
    missing_text_rows: int = 0
    text_hash_mismatches: int = 0
    verified_text_rows: int = 0


def candidate_order_digest(source: str, row: GeometryRow) -> str:
    """Return the same stable seed-42 ordering key used by cohort selection."""
    payload = "|".join(
        (
            "42",
            source,
            str(row["decision"]),
            str(row["stream"]),
            str(row["sentence_key"]),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def has_unique_native_coverage(result: Assignment | None) -> TypeGuard[Assignment]:
    """Require a unique production assignment meeting the inclusive 80% rule."""
    return (
        result is not None
        and result.code is not None
        and result.overlap_percentage is not None
        and result.overlap_percentage >= MINIMUM_NATIVE_OVERLAP_PERCENTAGE
    )


def iter_geometry_first_candidates(  # noqa: PLR0913
    geometry_rows: Iterable[GeometryRow],
    *,
    source: str,
    stream: str,
    classify: Classifier,
    fetch_texts: TextFetcher,
    batch_size: int = 256,
    counts: CensusCounts | None = None,
) -> Iterator[dict[str, object]]:
    """Yield hash-verified candidate rows, fetching text only after native assignment.

    ``geometry_rows`` must be ordered by ``candidate_order_digest``. ``fetch_texts``
    receives only rows whose geometry has one native EUNIS assignment at 80% or
    greater. It must return text keyed by ``sentence_key``; text is checked again
    against each pinned relevance label before a row is yielded.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    metrics = counts if counts is not None else CensusCounts()
    eligible_batch: list[tuple[GeometryRow, Assignment]] = []
    for row in geometry_rows:
        metrics.geometry_rows += 1
        assignment = classify(row["geometry"])
        if has_unique_native_coverage(assignment):
            metrics.native_eligible_rows += 1
            eligible_batch.append((row, assignment))
        if len(eligible_batch) >= batch_size:
            yield from _verified_batch(eligible_batch, source, stream, fetch_texts, metrics)
            eligible_batch = []
    if eligible_batch:
        yield from _verified_batch(eligible_batch, source, stream, fetch_texts, metrics)


def _verified_batch(
    rows: Sequence[tuple[GeometryRow, Assignment]],
    source: str,
    stream: str,
    fetch_texts: TextFetcher,
    counts: CensusCounts,
) -> Iterator[dict[str, object]]:
    counts.text_fetch_rows += len(rows)
    texts = fetch_texts([row for row, _ in rows])
    for row, assignment in rows:
        key = str(row["sentence_key"])
        sentence = texts.get(key)
        if sentence is None:
            counts.missing_text_rows += 1
            continue
        expected = str(row["text_sha256"]).lower()
        if _text_digest(sentence) != expected:
            counts.text_hash_mismatches += 1
            continue
        result = dict(row)
        result["raw_sentence"] = sentence
        result["source"] = source
        result["stream"] = stream
        result["eunis_code"] = assignment.code
        result["eunis_name"] = assignment.name
        result["eunis_overlap_percentage"] = assignment.overlap_percentage
        result["eunis_source_version"] = assignment.source_version
        counts.verified_text_rows += 1
        yield result


def _text_digest(sentence: str) -> str:
    return hashlib.sha256(sentence.encode("utf-8")).hexdigest()
