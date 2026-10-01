import json
import shlex
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import NotRequired, TypedDict, Unpack

import pytest

from osm_polygon_eunis import grid5000
from osm_polygon_eunis.grid5000 import (
    Grid5000Config,
    Grid5000Job,
    build_active_eunis_jobs_command,
    build_oarsub_command,
    build_policy_command,
    build_source_sync_command,
    build_ssh_command,
    build_status_command,
    parse_job_id,
    submit_grid5000,
    validate_persistent_root,
)

_EMPTY_ALL_SITE_REPORT = '{"active_jobs": [], "errors": []}'


class _ConfigOverrides(TypedDict):
    frontend: NotRequired[str]
    persistent_root: NotRequired[str]
    site: NotRequired[str]
    cluster: NotRequired[str]
    cores: NotRequired[int]
    workers: NotRequired[int]
    walltime: NotRequired[str]
    batch_size: NotRequired[int]
    excluded_sites: NotRequired[tuple[str, ...]]


def _config(**overrides: Unpack[_ConfigOverrides]) -> Grid5000Config:
    return Grid5000Config(
        frontend=overrides.get("frontend", "fgrenoble"),
        persistent_root=overrides.get("persistent_root", "/home/u/eunis"),
        site=overrides.get("site", "grenoble"),
        cluster=overrides.get("cluster", "dahu"),
        cores=overrides.get("cores", 16),
        workers=overrides.get("workers", 16),
        walltime=overrides.get("walltime", "1:00:00"),
        batch_size=overrides.get("batch_size", 256),
        excluded_sites=overrides.get("excluded_sites", ()),
    )


def test_grid5000_default_walltime_is_one_hour() -> None:
    config = Grid5000Config(
        frontend="fgrenoble",
        persistent_root="/home/u/eunis",
        site="grenoble",
        cluster="dahu",
    )

    assert config.walltime == "1:00:00"


def test_grid5000_rejects_walltime_over_one_hour() -> None:
    with pytest.raises(ValueError, match="one hour"):
        _config(walltime="1:00:01")


def test_grid5000_rejects_walltime_that_cannot_leave_default_stop_margin() -> None:
    with pytest.raises(ValueError, match="stop margin"):
        _config(walltime="0:05:00")


def test_grid5000_accepts_walltime_one_second_over_default_stop_margin() -> None:
    assert _config(walltime="0:05:01").walltime == "0:05:01"


def test_profile_requests_one_cpu_host_on_any_explicit_site() -> None:
    config = _config()

    assert build_oarsub_command(
        config,
        "/home/u/eunis/source/scripts/grid5000/release.sh",
    ) == (
        "oarsub",
        "-q",
        "default",
        "-n",
        "osm-polygon-eunis",
        "-p",
        "cluster='dahu'",
        "-l",
        "host=1/core=16,walltime=1:00:00",
        (
            "GRID5000_PERSISTENT_ROOT=/home/u/eunis GRID5000_FRONTEND=fgrenoble "
            "GRID5000_SITE=grenoble GRID5000_CLUSTER=dahu GRID5000_QUEUE=default "
            "GRID5000_CORES=16 GRID5000_WORKERS=16 GRID5000_WALLTIME=1:00:00 "
            "GRID5000_BATCH_SIZE=256 GRID5000_EXCLUDED_SITES='[]' "
            "/home/u/eunis/source/scripts/grid5000/release.sh"
        ),
    )


def test_submission_command_records_optional_job_metadata() -> None:
    config = replace(_config(), job_type="deploy")

    command = build_oarsub_command(
        config,
        "/home/u/eunis/source/scripts/grid5000/release.sh",
        source_revision="abc123",
    )

    assert command[command.index("-t") + 1] == "deploy"
    assert command[command.index("-n") + 1] == "osm-polygon-eunis"
    assert "GRID5000_JOB_TYPE=deploy" in command[-1]
    assert "GRID5000_SOURCE_REVISION=abc123" in command[-1]


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


def test_unreachable_site_exclusion_is_explicit_and_cannot_hide_compute_site() -> None:
    assert _config(excluded_sites=("bordeaux",)).excluded_sites == ("bordeaux",)
    with pytest.raises(ValueError, match="selected compute site"):
        _config(excluded_sites=("grenoble",))
    with pytest.raises(ValueError, match="unique"):
        _config(excluded_sites=("bordeaux", "bordeaux"))
    with pytest.raises(ValueError, match="excluded_sites"):
        _config(excluded_sites=("bordeaux;false",))


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
    assert parse_job_id("OAR_JOB_ID=123456\n") == "123456"
    with pytest.raises(ValueError):
        parse_job_id("submission failed")
    with pytest.raises(ValueError, match="exactly one"):
        parse_job_id("Adding job 1\nAdding job 2\n")


def test_all_site_inventory_script_writes_json_without_print() -> None:
    assert "sys.stdout.write(" in grid5000._ACTIVE_EUNIS_JOBS_SCRIPT
    assert "json.dumps(" in grid5000._ACTIVE_EUNIS_JOBS_SCRIPT
    assert "print(" not in grid5000._ACTIVE_EUNIS_JOBS_SCRIPT


@pytest.mark.parametrize(
    ("output", "state"),
    [
        ("Job 123456 running", "active"),
        ("ERROR: job not found", "missing"),
        ("Job 123456 terminated", "terminal"),
        ("No status returned", "unknown"),
    ],
)
def test_job_status_state_classifies_oar_output(output: str, state: str) -> None:
    assert grid5000._job_status_state(output) == state


def test_live_submission_state_path_is_required_only_for_live_jobs() -> None:
    assert grid5000._validate_live_submission_state(None, dry_run=True) is None
    with pytest.raises(ValueError, match="state_path is required"):
        grid5000._validate_live_submission_state(None, dry_run=False)


def test_policy_status_and_ssh_commands() -> None:
    assert build_policy_command() == ("usagepolicycheck", "-t")
    assert build_status_command("123456") == ("oarstat", "-j", "123456")
    assert build_ssh_command("fgrenoble", ("usagepolicycheck", "-t")) == (
        "ssh",
        "fgrenoble",
        "usagepolicycheck -t",
    )
    with pytest.raises(ValueError):
        build_status_command("not-a-job")


def test_policy_command_checks_every_api_site_except_explicit_exclusions() -> None:
    command = build_policy_command(("bordeaux",))

    assert command[:2] == ("python3", "-c")
    assert '"--sites"' in command[2]
    assert '"bordeaux"' in command[2]
    assert 'items(BASE + "/sites")' in command[2]
    assert "usagepolicycheck" in command[2]
    compile(command[2], "<Grid5000 usage policy check>", "exec")


def test_ssh_command_preserves_oar_expression_quotes_for_remote_shell() -> None:
    remote_command = ("oarsub", "-p", "cluster='grappe'")

    ssh_command = build_ssh_command("fnancy", remote_command)

    assert ssh_command == ("ssh", "fnancy", shlex.join(remote_command))
    assert tuple(shlex.split(ssh_command[2])) == remote_command


def test_source_sync_streams_only_the_requested_git_commit() -> None:
    command = build_source_sync_command(
        Path("/workspace/eunis"),
        "fgrenoble",
        "/home/u/eunis",
        "0123456789abcdef",
    )

    assert command[:4] == ("bash", "-o", "pipefail", "-c")
    assert "git -C /workspace/eunis archive --format=tar 0123456789abcdef" in command[4]
    assert "ssh fgrenoble" in command[4]
    assert "tar -xf -" in command[4]
    assert ".eunis-run-final" not in command[4]
    assert "rsync" not in " ".join(command)
    assert "HF_TOKEN" not in " ".join(command)


def test_all_site_job_check_uses_oarstat_on_api_discovered_sites() -> None:
    command = build_active_eunis_jobs_command("fgrenoble")
    remote_args = shlex.split(command[2])

    assert command[:2] == ("ssh", "fgrenoble")
    assert "https://api.grid5000.fr/stable" in command[2]
    assert '"/sites"' in command[2]
    assert "jobs" in command[2]
    assert "oarstat" in command[2]
    assert "osm-polygon-eunis" in command[2]
    compile(remote_args[-1], "<Grid5000 active-job check>", "exec")


def test_all_site_job_check_skips_only_explicitly_excluded_sites() -> None:
    command = build_active_eunis_jobs_command("fgrenoble", ("bordeaux",))
    script = shlex.split(command[2])[-1]

    assert '"bordeaux"' in script
    assert "if site_id in EXCLUDED_SITE_IDS" in script
    assert 'items(BASE + "/sites")' in script
    assert '"excluded_sites": sorted(EXCLUDED_SITE_IDS)' in script
    compile(script, "<Grid5000 active-job check>", "exec")


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
    state = tmp_path / "job.json"
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command == grid5000.build_active_eunis_jobs_command("fgrenoble"):
            return _EMPTY_ALL_SITE_REPORT
        if command[:2] == ("ssh", "fgrenoble") and shlex.split(command[2])[0] == "oarsub":
            return "[AO] Adding job 123456\n"
        return ""

    result = submit_grid5000(
        config,
        tmp_path,
        source_revision="abc123",
        state_path=state,
        runner=fake_runner,
    )

    assert result.job is not None
    assert result.job.job_id == "123456"
    assert result.datasets == ("website", "wikidata", "description")
    assert result.source_revision == "abc123"
    assert calls[0] == ("ssh", "fgrenoble", "usagepolicycheck -t")
    assert calls[1] == grid5000.build_active_eunis_jobs_command("fgrenoble")
    assert calls[2] == ("ssh", "fgrenoble", "mkdir -p /home/u/eunis")
    assert calls[3][:4] == ("bash", "-o", "pipefail", "-c")
    assert calls[4][:2] == ("ssh", "fgrenoble")
    remote_oarsub = tuple(shlex.split(calls[4][2]))
    assert remote_oarsub[remote_oarsub.index("-p") + 1] == "cluster='dahu'"
    assert remote_oarsub[-1] == (
        "GRID5000_PERSISTENT_ROOT=/home/u/eunis GRID5000_FRONTEND=fgrenoble "
        "GRID5000_SITE=grenoble GRID5000_CLUSTER=dahu GRID5000_QUEUE=default "
        "GRID5000_CORES=16 GRID5000_WORKERS=16 GRID5000_WALLTIME=1:00:00 "
        "GRID5000_BATCH_SIZE=256 GRID5000_EXCLUDED_SITES='[]' "
        "GRID5000_SOURCE_REVISION=abc123 "
        "/home/u/eunis/source/scripts/grid5000/release.sh"
    )
    assert calls[5] == ("ssh", "fgrenoble", "usagepolicycheck -t")
    assert all("HF_TOKEN" not in " ".join(command) for command in calls)
    assert json.loads(state.read_text(encoding="utf-8"))["job_id"] == "123456"


def test_submission_persists_explicit_site_exclusions_in_its_receipt(tmp_path: Path) -> None:
    config = _config(excluded_sites=("bordeaux",))
    state = tmp_path / "job.json"
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command == grid5000.build_active_eunis_jobs_command("fgrenoble", ("bordeaux",)):
            return json.dumps({"active_jobs": [], "errors": [], "excluded_sites": ["bordeaux"]})
        if command[:2] == ("ssh", "fgrenoble") and shlex.split(command[2])[0] == "oarsub":
            return "Adding job 123456"
        return ""

    submit_grid5000(
        config,
        tmp_path,
        source_revision="abc123",
        state_path=state,
        runner=fake_runner,
    )

    assert json.loads(state.read_text(encoding="utf-8"))["excluded_sites"] == ["bordeaux"]
    submission = next(
        command
        for command in calls
        if command[:2] == ("ssh", "fgrenoble") and "oarsub" in command[2]
    )
    worker_command = shlex.split(submission[2])[-1]
    excluded_setting = next(
        token
        for token in shlex.split(worker_command)
        if token.startswith("GRID5000_EXCLUDED_SITES=")
    )
    assert json.loads(excluded_setting.split("=", maxsplit=1)[1]) == ["bordeaux"]


def test_live_submission_requires_a_persistent_state_path(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    with pytest.raises(ValueError, match="state_path is required"):
        submit_grid5000(
            _config(),
            tmp_path,
            source_revision="abc123",
            runner=lambda command: calls.append(command) or "",
        )

    assert calls == []


def test_submit_persists_job_id_before_post_submission_policy_check(tmp_path: Path) -> None:
    config = _config()
    state = tmp_path / "job.json"
    policy_checks = 0

    def fake_runner(command: tuple[str, ...]) -> str:
        nonlocal policy_checks
        if command[-1] == "usagepolicycheck -t":
            policy_checks += 1
            if policy_checks == 2:
                raise subprocess.CalledProcessError(1, command, stderr="policy violation")
        if command == grid5000.build_active_eunis_jobs_command("fgrenoble"):
            return _EMPTY_ALL_SITE_REPORT
        if command[:2] == ("ssh", "fgrenoble") and shlex.split(command[2])[0] == "oarsub":
            return "[AO] Adding job 123456\n"
        return ""

    with pytest.raises(subprocess.CalledProcessError, match="returned non-zero"):
        submit_grid5000(
            config,
            tmp_path,
            source_revision="abc123",
            state_path=state,
            runner=fake_runner,
        )

    assert state.exists()
    assert json.loads(state.read_text(encoding="utf-8"))["job_id"] == "123456"


def test_submit_rejects_an_existing_active_job(tmp_path: Path) -> None:
    config = _config()
    state = tmp_path / "job.json"
    state.write_text('{"job_id": "123456"}\n', encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command == grid5000.build_active_eunis_jobs_command("fgrenoble"):
            return _EMPTY_ALL_SITE_REPORT
        if shlex.split(command[2])[0] == "oarstat":
            return "running"
        return "running"

    with pytest.raises(RuntimeError, match="already active"):
        submit_grid5000(
            config,
            tmp_path,
            source_revision="abc123",
            state_path=state,
            runner=fake_runner,
        )

    assert calls[:2] == [
        ("ssh", "fgrenoble", "usagepolicycheck -t"),
        grid5000.build_active_eunis_jobs_command("fgrenoble"),
    ]
    assert calls[2] == ("ssh", "fgrenoble", "oarstat -j 123456")


def test_submit_refuses_active_eunis_job_reported_by_any_site(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command == grid5000.build_active_eunis_jobs_command("fgrenoble"):
            return (
                '{"active_jobs": [{"job_id": "6942984", "site": "nancy", '
                '"state": "running"}], "errors": []}'
            )
        return ""

    with pytest.raises(RuntimeError, match="nancy:6942984"):
        submit_grid5000(
            _config(),
            tmp_path,
            source_revision="abc123",
            state_path=tmp_path / "state.json",
            runner=fake_runner,
        )

    assert calls == [
        ("ssh", "fgrenoble", "usagepolicycheck -t"),
        grid5000.build_active_eunis_jobs_command("fgrenoble"),
    ]
    assert not (tmp_path / "state.json").exists()


def test_submit_fails_closed_when_all_site_job_report_is_invalid(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command == grid5000.build_active_eunis_jobs_command("fgrenoble"):
            return "permission denied"
        return ""

    with pytest.raises(RuntimeError, match="cannot verify active EUNIS jobs"):
        submit_grid5000(
            _config(),
            tmp_path,
            source_revision="abc123",
            state_path=tmp_path / "state.json",
            runner=fake_runner,
        )

    assert len(calls) == 2


def test_submit_fails_closed_when_any_site_inventory_failed(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command == grid5000.build_active_eunis_jobs_command("fgrenoble"):
            return '{"active_jobs": [], "errors": [{"site": "bordeaux"}]}'
        return ""

    with pytest.raises(RuntimeError, match="bordeaux"):
        submit_grid5000(
            _config(),
            tmp_path,
            source_revision="abc123",
            state_path=tmp_path / "state.json",
            runner=fake_runner,
        )

    assert len(calls) == 2


def test_submit_fails_closed_when_exclusion_scope_does_not_match_request(tmp_path: Path) -> None:
    config = _config(excluded_sites=("bordeaux",))
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command == grid5000.build_active_eunis_jobs_command("fgrenoble", ("bordeaux",)):
            return _EMPTY_ALL_SITE_REPORT
        return ""

    with pytest.raises(RuntimeError, match="exclusion scope does not match"):
        submit_grid5000(
            config,
            tmp_path,
            source_revision="abc123",
            state_path=tmp_path / "state.json",
            runner=fake_runner,
        )

    assert len(calls) == 2


def test_parse_active_eunis_jobs_report_returns_validated_job_lists() -> None:
    parse_report = getattr(grid5000, "_parse_active_eunis_jobs_report", None)
    assert callable(parse_report), "active-job report parsing should be a separate unit"

    active = [{"site": "nancy", "job_id": "6942984", "state": "running"}]
    errors = [{"site": "bordeaux", "error": "unreachable"}]
    output = json.dumps({"active_jobs": active, "errors": errors})

    assert parse_report(output) == (active, errors)


@pytest.mark.parametrize(
    "output",
    [
        "[]",
        '{"active_jobs": "invalid", "errors": []}',
        '{"active_jobs": [null], "errors": []}',
        '{"active_jobs": [], "errors": "invalid"}',
        '{"active_jobs": [], "errors": [null]}',
    ],
)
def test_parse_active_eunis_jobs_report_rejects_malformed_lists(output: str) -> None:
    parse_report = getattr(grid5000, "_parse_active_eunis_jobs_report", None)
    assert callable(parse_report), "active-job report parsing should be a separate unit"

    with pytest.raises(TypeError, match="cannot verify active EUNIS jobs"):
        parse_report(output)


def test_submit_refuses_when_existing_job_status_cannot_be_verified(tmp_path: Path) -> None:
    config = _config()
    state = tmp_path / "job.json"
    state.write_text('{"job_id": "123456"}\n', encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command == grid5000.build_active_eunis_jobs_command("fgrenoble"):
            return _EMPTY_ALL_SITE_REPORT
        if command[:2] == ("ssh", "fgrenoble") and shlex.split(command[2])[0] == "oarstat":
            raise subprocess.CalledProcessError(255, command, stderr="Connection timed out")
        return ""

    with pytest.raises(RuntimeError, match="cannot verify"):
        submit_grid5000(
            config,
            tmp_path,
            source_revision="abc123",
            state_path=state,
            runner=fake_runner,
        )

    assert calls[:2] == [
        ("ssh", "fgrenoble", "usagepolicycheck -t"),
        grid5000.build_active_eunis_jobs_command("fgrenoble"),
    ]
    assert calls[2] == ("ssh", "fgrenoble", "oarstat -j 123456")


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
    assert result.commands[0] == ("ssh", "fgrenoble", "usagepolicycheck -t")
    assert result.commands[1] == grid5000.build_active_eunis_jobs_command("fgrenoble")
    assert calls == []


def test_submit_replaces_state_after_terminal_job_and_writes_safe_state(
    tmp_path: Path,
) -> None:
    config = _config()
    state = tmp_path / "state.json"
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command == grid5000.build_active_eunis_jobs_command("fgrenoble"):
            return _EMPTY_ALL_SITE_REPORT
        if command[:2] == ("ssh", "fgrenoble") and shlex.split(command[2])[0] == "oarstat":
            raise subprocess.CalledProcessError(1, command, stderr="ERROR: job not found")
        if command[:2] == ("ssh", "fgrenoble") and shlex.split(command[2])[0] == "oarsub":
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
    assert calls[0] == (
        "ssh",
        "fgrenoble",
        "usagepolicycheck -t",
    )
    assert calls[1] == grid5000.build_active_eunis_jobs_command("fgrenoble")
    assert calls[2] == (
        "ssh",
        "fgrenoble",
        f"oarstat -j {first.job.job_id}",
    )


def test_status_error_output_preserves_both_captured_streams() -> None:
    error = subprocess.CalledProcessError(
        255,
        ("ssh", "fgrenoble", "oarstat -j 123456"),
        output="stdout detail",
        stderr="stderr detail",
    )

    assert grid5000._status_error_output(error) == "stdout detail\nstderr detail"


def test_submit_rejects_corrupt_or_incomplete_state(tmp_path: Path) -> None:
    config = _config()
    for contents in ("not json", "{}"):
        state = tmp_path / "state.json"
        state.write_text(contents, encoding="utf-8")
        with pytest.raises(ValueError, match="job state"):

            def fake_runner(command: tuple[str, ...]) -> str:
                if command == grid5000.build_active_eunis_jobs_command("fgrenoble"):
                    return _EMPTY_ALL_SITE_REPORT
                return ""

            submit_grid5000(
                config,
                tmp_path,
                source_revision="abc123",
                runner=fake_runner,
                state_path=state,
            )


def test_config_rejects_unsafe_cluster() -> None:
    with pytest.raises(ValueError, match="cluster"):
        _config(cluster="dahu' OR 1=1")


def test_resolve_source_revision_checks_cleanliness() -> None:
    def clean_runner(command: tuple[str, ...]) -> str:
        return "abc123\n" if command[-1] == "HEAD" else ""

    assert (
        grid5000.resolve_source_revision(Path("/workspace/eunis"), runner=clean_runner) == "abc123"
    )

    def untracked_runner(command: tuple[str, ...]) -> str:
        return "?? .eunis-run-final/\n" if "--porcelain" in command else "abc123\n"

    assert (
        grid5000.resolve_source_revision(Path("/workspace/eunis"), runner=untracked_runner)
        == "abc123"
    )

    def dirty_runner(command: tuple[str, ...]) -> str:
        return "M README.md" if "--porcelain" in command else ""

    with pytest.raises(RuntimeError, match="dirty"):
        grid5000.resolve_source_revision(Path("/workspace/eunis"), runner=dirty_runner)
    assert (
        grid5000.resolve_source_revision(
            Path("/workspace/eunis"),
            explicit="def456",
            allow_dirty=True,
            runner=dirty_runner,
        )
        == "def456-dirty"
    )


def test_run_command_returns_combined_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **kwargs):
        assert command == ("true",)
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(command, 0, stdout="out", stderr="err")

    monkeypatch.setattr(grid5000.subprocess, "run", fake_run)
    assert grid5000.run_command(("true",)) == "outerr"
