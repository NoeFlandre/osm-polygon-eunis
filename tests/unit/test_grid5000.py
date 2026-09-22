from pathlib import Path

import pytest

from osm_polygon_eunis.grid5000 import (
    Grid5000Config,
    build_cancel_command,
    build_oarsub_command,
    build_policy_command,
    build_rsync_command,
    build_ssh_command,
    build_status_command,
    parse_job_id,
    validate_persistent_root,
)


def test_default_profile_requests_one_lille_cpu_host() -> None:
    config = Grid5000Config(frontend="flille", persistent_root="/home/u/eunis")

    assert build_oarsub_command(
        config,
        "/home/u/eunis/source/scripts/grid5000/description-release.sh",
    ) == (
        "oarsub",
        "-q",
        "default",
        "-t",
        "night",
        "-p",
        "chuc",
        "-l",
        "host=1/core=16,walltime=12:00:00",
        "-S",
        "/home/u/eunis/source/scripts/grid5000/description-release.sh",
    )


def test_config_rejects_invalid_resource_values() -> None:
    with pytest.raises(ValueError, match="frontend"):
        Grid5000Config(frontend="", persistent_root="/home/u/eunis")
    with pytest.raises(ValueError, match="cores"):
        Grid5000Config(frontend="flille", persistent_root="/home/u/eunis", cores=0)
    with pytest.raises(ValueError, match="workers"):
        Grid5000Config(frontend="flille", persistent_root="/home/u/eunis", workers=0)
    with pytest.raises(ValueError, match="walltime"):
        Grid5000Config(frontend="flille", persistent_root="/home/u/eunis", walltime="forever")
    with pytest.raises(ValueError, match="persistent"):
        Grid5000Config(frontend="flille", persistent_root="relative/eunis")


def test_persistent_root_rejects_ephemeral_or_mac_paths() -> None:
    for path in (
        "/tmp/eunis",
        "/private/tmp/eunis",
        "/Volumes/Seagate/eunis",
        "/Users/noe/eunis",
    ):
        with pytest.raises(ValueError):
            validate_persistent_root(path)

    assert validate_persistent_root("/home/u/eunis") == "/home/u/eunis"
    assert validate_persistent_root("/groups/project/eunis") == "/groups/project/eunis"


def test_job_id_parser_accepts_oar_output_and_rejects_ambiguous_text() -> None:
    assert parse_job_id("[AO] Adding job 123456\n") == "123456"
    with pytest.raises(ValueError):
        parse_job_id("submission failed")
    with pytest.raises(ValueError, match="exactly one"):
        parse_job_id("Adding job 1\nAdding job 2\n")


def test_policy_status_cancel_and_ssh_commands_are_argument_arrays() -> None:
    assert build_policy_command() == ("usagepolicycheck", "-t")
    assert build_status_command("123456") == ("oarstat", "-j", "123456")
    assert build_cancel_command("123456") == ("oardel", "123456")
    assert build_ssh_command("flille", ("usagepolicycheck", "-t")) == (
        "ssh",
        "flille",
        "usagepolicycheck",
        "-t",
    )
    with pytest.raises(ValueError):
        build_status_command("not-a-job")


def test_rsync_excludes_credentials_and_ephemeral_project_state() -> None:
    command = build_rsync_command(
        Path("/workspace/eunis"),
        "flille",
        "/home/u/eunis/source",
    )

    assert command[:2] == ("rsync", "-az")
    assert "--exclude=.git" in command
    assert "--exclude=.venv" in command
    assert "--exclude=.eunis-run-final" in command
    assert "--exclude=.env" in command
    assert "--exclude=.cache" in command
    assert command[-2:] == (
        "/workspace/eunis/",
        "flille:/home/u/eunis/source/",
    )
    assert "HF_TOKEN" not in " ".join(command)


def test_grid_job_is_immutable() -> None:
    from osm_polygon_eunis.grid5000 import Grid5000Job

    job = Grid5000Job(
        job_id="123456",
        submitted_at="2026-09-22T10:00:00Z",
        config=Grid5000Config(frontend="flille", persistent_root="/home/u/eunis"),
        source_revision="abc123",
    )

    with pytest.raises(AttributeError):
        job.job_id = "654321"  # type: ignore[misc]
