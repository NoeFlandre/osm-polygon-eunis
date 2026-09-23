import subprocess
from pathlib import Path

import pytest

import osm_polygon_eunis.grid5000 as grid5000
from osm_polygon_eunis.grid5000 import (
    Grid5000Config,
    Grid5000Job,
    build_cancel_command,
    build_oarsub_command,
    build_policy_command,
    build_rsync_command,
    build_ssh_command,
    build_status_command,
    parse_job_id,
    submit_grid5000,
    validate_persistent_root,
)


def _config(
    *,
    frontend: str = "fgrenoble",
    persistent_root: str = "/home/u/eunis",
    site: str = "grenoble",
    cluster: str = "dahu",
    cores: int = 16,
    workers: int = 16,
    walltime: str = "12:00:00",
    batch_size: int = 256,
) -> Grid5000Config:
    return Grid5000Config(
        frontend=frontend,
        persistent_root=persistent_root,
        site=site,
        cluster=cluster,
        cores=cores,
        workers=workers,
        walltime=walltime,
        batch_size=batch_size,
    )


def test_profile_requests_one_cpu_host_on_any_explicit_site() -> None:
    config = _config()

    assert build_oarsub_command(
        config,
        "/home/u/eunis/source/scripts/grid5000/release.sh",
    ) == (
        "oarsub",
        "-q",
        "default",
        "-p",
        "cluster='dahu'",
        "-l",
        "host=1/core=16,walltime=12:00:00",
        "-S",
        "/home/u/eunis/source/scripts/grid5000/release.sh",
    )


def test_config_rejects_invalid_resource_values() -> None:
    with pytest.raises(ValueError, match="frontend"):
        _config(frontend="")
    with pytest.raises(ValueError, match="cores"):
        _config(cores=0)
    with pytest.raises(ValueError, match="workers"):
        _config(workers=0)
    with pytest.raises(ValueError, match="walltime"):
        _config(walltime="forever")
    with pytest.raises(ValueError, match="persistent"):
        _config(persistent_root="relative/eunis")


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
    assert build_ssh_command("fgrenoble", ("usagepolicycheck", "-t")) == (
        "ssh",
        "fgrenoble",
        "usagepolicycheck",
        "-t",
    )
    with pytest.raises(ValueError):
        build_status_command("not-a-job")


def test_rsync_excludes_credentials_and_ephemeral_project_state() -> None:
    command = build_rsync_command(
        Path("/workspace/eunis"),
        "fgrenoble",
        "/home/u/eunis/source",
    )

    assert command[:2] == ("rsync", "-az")
    assert "--exclude=.git" in command
    assert "--exclude=.venv" in command
    assert "--exclude=.eunis-run-final" in command
    assert "--exclude=.grid5000-*.json" in command
    assert "--exclude=.env" in command
    assert "--exclude=.cache" in command
    assert "--exclude=.coverage*" in command
    assert "--exclude=coverage.json" in command
    assert command[-2:] == (
        "/workspace/eunis/",
        "fgrenoble:/home/u/eunis/source/",
    )
    assert "HF_TOKEN" not in " ".join(command)


def test_grid_job_is_immutable() -> None:
    job = Grid5000Job(
        job_id="123456",
        submitted_at="2026-09-22T10:00:00Z",
        config=_config(),
        source_revision="abc123",
    )

    with pytest.raises(AttributeError):
        job.__setattr__("job_id", "654321")


def test_submit_runs_policy_sync_oar_and_post_policy_without_secrets(tmp_path: Path) -> None:
    config = _config()
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command[:3] == ("ssh", "fgrenoble", "oarsub"):
            return "[AO] Adding job 123456\n"
        return ""

    result = submit_grid5000(
        config,
        tmp_path,
        source_revision="abc123",
        runner=fake_runner,
    )

    assert result.job is not None
    assert result.job.job_id == "123456"
    assert result.datasets == ("website", "wikidata", "description")
    assert result.source_revision == "abc123"
    assert calls[0] == ("ssh", "fgrenoble", "usagepolicycheck", "-t")
    assert calls[1] == (
        "ssh",
        "fgrenoble",
        "mkdir",
        "-p",
        "/home/u/eunis/source",
    )
    assert calls[2][0:2] == ("rsync", "-az")
    assert calls[3][:3] == ("ssh", "fgrenoble", "oarsub")
    assert "cluster='dahu'" in calls[3]
    assert calls[3][-1] == "/home/u/eunis/source/scripts/grid5000/release.sh"
    assert calls[4] == ("ssh", "fgrenoble", "usagepolicycheck", "-t")
    assert all("HF_TOKEN" not in " ".join(command) for command in calls)


def test_submit_rejects_an_existing_active_job(tmp_path: Path) -> None:
    config = _config()
    state = tmp_path / "job.json"
    state.write_text('{"job_id": "123456"}\n', encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        return "running"

    with pytest.raises(RuntimeError, match="already active"):
        submit_grid5000(
            config,
            tmp_path,
            source_revision="abc123",
            state_path=state,
            runner=fake_runner,
        )

    assert calls == [("ssh", "fgrenoble", "oarstat", "-j", "123456")]


def test_config_rejects_capacity_batch_and_blank_profile_fields() -> None:
    with pytest.raises(ValueError, match="workers must not exceed"):
        _config(workers=17)
    with pytest.raises(ValueError, match="batch_size"):
        _config(batch_size=0)
    with pytest.raises(ValueError, match="site"):
        _config(site=" ")


def test_path_and_command_builders_reject_unsafe_inputs() -> None:
    for path in ("", "\x00", "/home", "/home/u/../eunis"):
        with pytest.raises(ValueError):
            validate_persistent_root(path)

    config = _config()
    with pytest.raises(ValueError, match="absolute remote"):
        build_oarsub_command(config, "scripts/worker.sh")
    with pytest.raises(ValueError, match="frontend"):
        build_ssh_command("fl ille", ("true",))
    with pytest.raises(ValueError, match="must not be empty"):
        build_ssh_command("fgrenoble", ())


def test_submit_dry_run_builds_commands_without_contacting_grid5000(tmp_path: Path) -> None:
    config = _config()
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        return ""

    result = submit_grid5000(
        config,
        tmp_path,
        source_revision="abc123",
        runner=fake_runner,
        dry_run=True,
    )

    assert result.job is None
    assert result.commands[0] == ("ssh", "fgrenoble", "usagepolicycheck", "-t")
    assert calls == []


def test_submit_replaces_state_after_terminal_job_and_writes_safe_state(
    tmp_path: Path,
) -> None:
    config = _config()
    state = tmp_path / "state.json"
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command[:3] == ("ssh", "fgrenoble", "oarstat"):
            raise subprocess.CalledProcessError(1, command)
        if command[:3] == ("ssh", "fgrenoble", "oarsub"):
            return "Adding job 654321"
        return ""

    first = submit_grid5000(
        config,
        tmp_path,
        source_revision="abc123",
        runner=fake_runner,
        state_path=state,
    )
    calls.clear()
    second = submit_grid5000(
        config,
        tmp_path,
        source_revision="def456",
        runner=fake_runner,
        state_path=state,
    )

    assert first.job is not None
    assert second.job is not None
    assert second.job.job_id == "654321"
    payload = state.read_text(encoding="utf-8")
    assert '"job_id": "654321"' in payload
    assert '"cluster": "dahu"' in payload
    assert '"datasets": [\n    "website",\n    "wikidata",\n    "description"\n  ]' in payload
    assert '"site": "grenoble"' in payload
    assert "HF_TOKEN" not in payload
    assert calls[0] == ("ssh", "fgrenoble", "oarstat", "-j", first.job.job_id)


def test_submit_rejects_corrupt_or_incomplete_state(tmp_path: Path) -> None:
    config = _config()
    for contents in ("not json", "{}"):
        state = tmp_path / "state.json"
        state.write_text(contents, encoding="utf-8")
        with pytest.raises(ValueError, match="job state"):
            submit_grid5000(
                config,
                tmp_path,
                source_revision="abc123",
                runner=lambda command: "",
                state_path=state,
            )


def test_config_rejects_unsafe_cluster() -> None:
    with pytest.raises(ValueError, match="cluster"):
        _config(cluster="dahu' OR 1=1")


def test_resolve_source_revision_checks_cleanliness() -> None:
    def clean_runner(command: tuple[str, ...]) -> str:
        return "abc123\n" if command[-1] == "HEAD" else ""

    assert grid5000.resolve_source_revision(
        Path("/workspace/eunis"), runner=clean_runner
    ) == "abc123"

    def dirty_runner(command: tuple[str, ...]) -> str:
        return "M README.md" if command[-1] == "--porcelain" else ""

    with pytest.raises(RuntimeError, match="dirty"):
        grid5000.resolve_source_revision(Path("/workspace/eunis"), runner=dirty_runner)
    assert grid5000.resolve_source_revision(
        Path("/workspace/eunis"),
        explicit="def456",
        allow_dirty=True,
        runner=dirty_runner,
    ) == "def456-dirty"


def test_run_command_returns_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **kwargs):
        assert command == ("true",)
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(grid5000.subprocess, "run", fake_run)
    assert grid5000.run_command(("true",)) == "ok"
