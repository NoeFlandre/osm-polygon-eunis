from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "docs.yml"


def _load_workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_docs_workflow_builds_strictly_and_deploys_pages_with_minimal_permissions() -> None:
    workflow = _load_workflow()
    events = workflow.get("on", workflow.get(True))
    assert events["push"]["branches"] == ["main"]
    assert "workflow_dispatch" in events
    assert workflow["permissions"] == {"contents": "read"}

    build = workflow["jobs"]["build"]
    build_commands = [step.get("run", "") for step in build["steps"]]
    assert any("mkdocs build --strict --site-dir site" in command for command in build_commands)
    upload = next(
        step
        for step in build["steps"]
        if step.get("uses", "").startswith("actions/upload-pages-artifact@")
    )
    assert upload["with"]["path"] == "site"

    deploy = workflow["jobs"]["deploy"]
    assert deploy["needs"] == "build"
    assert deploy["permissions"] == {"pages": "write", "id-token": "write"}
    assert deploy["environment"]["name"] == "github-pages"
    assert deploy["environment"]["url"] == "${{ steps.deployment.outputs.page_url }}"
    assert any(
        step.get("uses", "").startswith("actions/deploy-pages@") and step.get("id") == "deployment"
        for step in deploy["steps"]
    )

    action_refs = [
        step["uses"]
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "uses" in step
    ]
    for reference in action_refs:
        assert re.search(r"@[0-9a-f]{40}(?:\s|$)", reference), reference


def test_contributor_quality_task_is_shared_by_pre_commit_and_ci() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    qa_workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "qa.yml").read_text())
    pre_commit = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hooks = [hook for repository in pre_commit["repos"] for hook in repository["hooks"]]

    assert (
        "quality: lint shellcheck format-check typecheck test architecture crap vulture smoke docs"
        in makefile
    )
    quality_steps = [
        step.get("run", "") for step in qa_workflow["jobs"]["deterministic-quality"]["steps"]
    ]
    assert "make quality" in quality_steps
    assert {hook["id"] for hook in hooks} == {
        "check-added-large-files",
        "check-yaml",
        "end-of-file-fixer",
        "ruff",
        "ruff-format",
        "ty",
    }
    assert (ROOT / "CONTRIBUTING.md").is_file()
    assert (ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml").is_file()
    assert (ROOT / ".github" / "ISSUE_TEMPLATE" / "feature_request.yml").is_file()
    assert (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").is_file()


def test_docker_runtime_installs_rasterio_shared_library_dependencies() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    runtime_stage = re.search(
        r"^FROM python:3\.12-slim-bookworm(?:@sha256:[0-9a-f]{64})? AS runtime$",
        dockerfile,
        flags=re.MULTILINE,
    )
    assert runtime_stage is not None
    runtime = dockerfile[runtime_stage.end() :]

    assert "apt-get update" in runtime
    assert "libexpat1" in runtime
    assert "rm -rf /var/lib/apt/lists/*" in runtime


def test_docker_ci_checks_non_root_user_and_secret_free_image_history() -> None:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "qa.yml").read_text())
    docker_steps = workflow["jobs"]["docker"]["steps"]
    commands = "\n".join(step.get("run", "") for step in docker_steps)

    assert "docker image inspect --format" in commands
    assert '[ "$image_user" = "0" ]' in commands
    assert '[ "$image_user" = "root" ]' in commands
    assert '[ "$image_user" = "0:0" ]' in commands
    assert "docker history --no-trunc" in commands
    assert "HF_TOKEN" in commands
    assert "HUGGINGFACE_HUB_TOKEN" in commands
    assert "hf_[A-Za-z0-9]" in commands


def test_declared_python_support_matches_the_ci_matrix() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "qa.yml").read_text())

    assert project["requires-python"] == ">=3.12,<3.15"
    assert workflow["jobs"]["deterministic-quality"]["strategy"]["matrix"]["python-version"] == [
        "3.12",
        "3.13",
        "3.14",
    ]
