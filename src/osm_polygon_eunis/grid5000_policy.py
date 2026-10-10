"""Usage-policy and duplicate-job gates that run before a Grid'5000 submission.

The module builds the remote checks and interprets their output. A failed or
unclear result raises RuntimeError, so the controller in grid5000.py never
submits after an ambiguous policy report.
"""

from __future__ import annotations

import json
import re
from typing import Final, TypeGuard

from .grid5000_commands import Command, build_ssh_command, validate_host

_POLICY_CHECK_SCRIPT: Final = r"""import json
import subprocess
import sys
import urllib.parse
import urllib.request

BASE = "https://api.grid5000.fr/stable"
EXCLUDED_SITE_IDS = set(__EXCLUDED_SITE_IDS__)

def items(url):
    visited = set()
    while url:
        if url in visited:
            raise RuntimeError("Grid5000 API pagination loop")
        visited.add(url)
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.load(response)
        if isinstance(payload, list):
            yield from payload
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise RuntimeError("unexpected Grid5000 API collection response")
        yield from payload["items"]
        next_link = next(
            (link.get("href") for link in payload.get("links", [])
             if isinstance(link, dict) and link.get("rel") == "next"),
            None,
        )
        url = urllib.parse.urljoin(BASE + "/", next_link) if next_link else None

sites = list(items(BASE + "/sites"))
site_ids = set()
for site in sites:
    if not isinstance(site, dict):
        raise RuntimeError("unexpected Grid5000 site entry")
    site_id = site.get("uid") or site.get("id")
    if not isinstance(site_id, str) or not site_id:
        raise RuntimeError("Grid5000 site entry has no identifier")
    site_ids.add(site_id)
unknown_exclusions = sorted(EXCLUDED_SITE_IDS - site_ids)
if unknown_exclusions:
    raise RuntimeError("excluded site is absent from the current API inventory: "
                       + ",".join(unknown_exclusions))
included_site_ids = sorted(site_ids - EXCLUDED_SITE_IDS)
if not included_site_ids:
    raise RuntimeError("no Grid5000 sites remain in the policy check")
sys.stderr.write("Grid'5000 policy sites: " + ",".join(included_site_ids) + "\n")
result = subprocess.run(
    ["usagepolicycheck", "-t", "--sites", ",".join(included_site_ids)],
    check=False,
    capture_output=True,
    text=True,
)
sys.stdout.write(result.stdout)
sys.stderr.write(result.stderr)
sys.exit(result.returncode)
"""
_ACTIVE_EUNIS_JOBS_SCRIPT: Final = r"""import json
import sys
import subprocess
import urllib.parse
import urllib.request

BASE = "https://api.grid5000.fr/stable"
TERMINAL = {"terminated", "error", "killed", "deleted", "finished", "completed"}
MARKERS = ("osm-polygon-eunis", "/scripts/grid5000/release.sh", "grid5000_source_revision")
EXCLUDED_SITE_IDS = set(__EXCLUDED_SITE_IDS__)

def get_json(url):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)
    except Exception as error:
        raise RuntimeError(f"Grid5000 API request failed for {url}: {error}") from error

def items(url):
    visited = set()
    while url:
        if url in visited:
            raise RuntimeError("Grid5000 API pagination loop")
        visited.add(url)
        payload = get_json(url)
        if isinstance(payload, list):
            yield from payload
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise RuntimeError("unexpected Grid5000 API collection response")
        yield from payload["items"]
        links = payload.get("links", [])
        next_link = next(
            (link.get("href") for link in links
             if isinstance(link, dict) and link.get("rel") == "next"),
            None,
        )
        url = urllib.parse.urljoin(BASE + "/", next_link) if next_link else None

sites = list(items(BASE + "/sites"))
active_jobs = []
errors = []
site_ids = {
    site.get("uid") or site.get("id")
    for site in sites
    if isinstance(site, dict) and isinstance(site.get("uid") or site.get("id"), str)
}
for site_id in sorted(EXCLUDED_SITE_IDS - site_ids):
    errors.append(
        {"site": site_id, "error": "excluded site is absent from the current API inventory"}
    )
for site in sites:
    if not isinstance(site, dict):
        raise RuntimeError("unexpected Grid5000 site entry")
    site_id = site.get("uid") or site.get("id")
    if not isinstance(site_id, str) or not site_id:
        raise RuntimeError("Grid5000 site entry has no identifier")
    if site_id in EXCLUDED_SITE_IDS:
        continue
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", site_id,
             "oarstat", "-u", "-J"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        jobs = json.loads(result.stdout) if result.stdout.strip() else {}
        if not isinstance(jobs, dict):
            raise RuntimeError("oarstat did not return a job object")
        for job_id, job in jobs.items():
            if not isinstance(job, dict):
                raise RuntimeError("oarstat returned a malformed job entry")
            state = str(job.get("state", "")).casefold()
            searchable = " ".join(
                str(job.get(key, ""))
                for key in ("name", "command", "launching_directory", "initial_request")
            ).casefold()
            if state not in TERMINAL and any(marker in searchable for marker in MARKERS):
                active_jobs.append({
                    "site": site_id,
                    "job_id": str(job.get("id", job_id)),
                    "state": state or "unknown",
                    "name": job.get("name"),
                })
    except Exception as error:
        errors.append({"site": site_id, "error": f"{type(error).__name__}: {error}"})
sys.stdout.write(
    json.dumps(
        {
            "active_jobs": active_jobs,
            "errors": errors,
            "excluded_sites": sorted(EXCLUDED_SITE_IDS),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    + "\n"
)
"""


def validate_excluded_sites(excluded_sites: tuple[str, ...]) -> tuple[str, ...]:
    """Return the sorted excluded site identifiers, rejecting duplicates and bad names."""

    if not isinstance(excluded_sites, tuple):
        raise TypeError("excluded_sites must be a tuple of site identifiers")
    for site_id in excluded_sites:
        validate_host(site_id, "excluded_sites")
    if len(set(excluded_sites)) != len(excluded_sites):
        raise ValueError("excluded_sites must be unique")
    return tuple(sorted(excluded_sites))


def build_policy_command(excluded_sites: tuple[str, ...] = ()) -> Command:
    """Build a usage-policy check, explicitly omitting only approved sites."""

    exclusions = validate_excluded_sites(excluded_sites)
    if not exclusions:
        return ("usagepolicycheck", "-t")
    script = _POLICY_CHECK_SCRIPT.replace("__EXCLUDED_SITE_IDS__", json.dumps(exclusions))
    return ("python3", "-c", script)


def build_active_eunis_jobs_command(
    frontend: str,
    excluded_sites: tuple[str, ...] = (),
) -> Command:
    """Query all API sites over OAR SSH except explicitly excluded sites."""

    exclusions = validate_excluded_sites(excluded_sites)
    script = _ACTIVE_EUNIS_JOBS_SCRIPT.replace("__EXCLUDED_SITE_IDS__", json.dumps(exclusions))
    return build_ssh_command(frontend, ("python3", "-c", script))


def reject_active_eunis_jobs(output: str, excluded_sites: tuple[str, ...] = ()) -> None:
    """Raise RuntimeError when the all-site report lists active EUNIS jobs or unverifiable sites."""

    matches, errors = _parse_active_eunis_jobs_report(output, excluded_sites)
    if matches:
        summary = _active_eunis_job_summary(matches)
        raise RuntimeError(f"active EUNIS Grid'5000 job(s) already exist: {summary}")
    if errors:
        sites = _unreachable_eunis_site_summary(errors)
        raise RuntimeError(f"cannot verify active EUNIS jobs on Grid'5000 site(s): {sites}")


def reject_unclean_policy_output(output: str) -> None:
    """Require a completed policy check; warnings about other jobs do not block.

    The check must either say nothing was flagged or print its conformance
    report header, and must not report an error. Flagged day/night notices for
    unrelated jobs are tolerated; EUNIS jobs are limited to one host and one
    hour, and duplicate-job checks run separately.
    """

    completed = re.search(r"(?m)^\s*No jobs flagged\s*$|testing usage policy conformance", output)
    reported_error = re.search(r"(?im)^\s*(?:error|fatal):", output)
    if completed is None or reported_error is not None:
        raise RuntimeError(
            "usagepolicycheck did not produce a clean usage-policy result; "
            "refusing Grid'5000 submission"
        )


def _parse_active_eunis_jobs_report(
    output: str,
    excluded_sites: tuple[str, ...] = (),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    try:
        report = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError("cannot verify active EUNIS jobs across Grid'5000 sites") from error
    if not isinstance(report, dict):
        raise TypeError("cannot verify active EUNIS jobs across Grid'5000 sites")
    _validate_report_exclusion_scope(report, excluded_sites)
    matches = report.get("active_jobs")
    errors = report.get("errors")
    if not _is_eunis_report_entries(matches) or not _is_eunis_report_entries(errors):
        raise TypeError("cannot verify active EUNIS jobs across Grid'5000 sites")
    return matches, errors


def _validate_report_exclusion_scope(
    report: dict[str, object], excluded_sites: tuple[str, ...]
) -> None:
    if report.get("excluded_sites", []) != list(validate_excluded_sites(excluded_sites)):
        raise RuntimeError(
            "cannot verify active EUNIS jobs: exclusion scope does not match request"
        )


def _is_eunis_report_entries(value: object) -> TypeGuard[list[dict[str, object]]]:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _active_eunis_job_summary(matches: list[dict[str, object]]) -> str:
    return ", ".join(
        f"{item.get('site', 'unknown')}:{item.get('job_id', 'unknown')}"
        f" ({item.get('state', 'unknown')})"
        for item in matches
    )


def _unreachable_eunis_site_summary(errors: list[dict[str, object]]) -> str:
    return ", ".join(str(item.get("site", "unknown")) for item in errors)
