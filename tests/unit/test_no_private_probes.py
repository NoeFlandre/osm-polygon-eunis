"""Guard against tests that only probe for the existence of private names."""

import re
from pathlib import Path

_PROBES = (
    re.compile(r"assert\s+callable\("),
    re.compile(r"getattr\([^)]*,\s*[\"']_\w+[\"']"),
)


def test_tests_do_not_probe_for_private_names() -> None:
    root = Path(__file__).parents[1]
    offenders = [
        f"{path.relative_to(root)}:{number}"
        for path in sorted(root.rglob("*.py"))
        if path != Path(__file__)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if any(probe.search(line) for probe in _PROBES)
    ]

    assert offenders == []
