"""Guard against tests that only probe for the existence of private names."""

import ast
from pathlib import Path
from typing import cast

import pytest

_ATTRIBUTE_PROBES = frozenset({"getattr", "hasattr", "setattr", "delattr"})


def _is_private_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("_")
        and not (value.startswith("__") and value.endswith("__"))
    )


def _is_private_attribute_probe(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _ATTRIBUTE_PROBES
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and _is_private_name(node.args[1].value)
    )


def _is_assert_callable(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Assert)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Name)
        and node.test.func.id == "callable"
    )


def _probe_lines(source: str) -> list[int]:
    """Return line numbers of private-name probes and ``assert callable(...)``."""

    return sorted(
        cast(ast.stmt | ast.expr, node).lineno
        for node in ast.walk(ast.parse(source))
        if _is_private_attribute_probe(node) or _is_assert_callable(node)
    )


@pytest.mark.parametrize(
    "source",
    [
        'getattr(module, "_helper", None)',
        'getattr(\n    module,\n    "_helper",\n    None,\n)',
        'getattr((load("a", (1, 2))), "_helper")',
        'hasattr(module, "_helper")',
        "assert callable(parse)",
        'assert callable(\n    getattr(module, "public")\n), "message"',
    ],
)
def test_probe_detector_flags_private_probes(source: str) -> None:
    assert _probe_lines(source) != []


@pytest.mark.parametrize(
    "source",
    [
        'getattr(module, "public", None)',
        'getattr(module, "__name__")',
        "getattr(module, name)",
        "assert parse(value) == 1",
        "callable(parse)",
    ],
)
def test_probe_detector_accepts_other_code(source: str) -> None:
    assert _probe_lines(source) == []


def test_tests_do_not_probe_for_private_names() -> None:
    root = Path(__file__).parents[1]
    offenders = [
        f"{path.relative_to(root)}:{number}"
        for path in sorted(root.rglob("*.py"))
        if path != Path(__file__)
        for number in _probe_lines(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []
