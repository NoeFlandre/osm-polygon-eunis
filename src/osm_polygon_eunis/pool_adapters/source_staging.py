"""Source-identity-scoped paths for verified input staging."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Metadata that uniquely identifies one checked-in raw-source tree."""

    text_source: str
    repository: str
    revision: str
    path_prefix: str
    tree_inventory_sha256: str


def staged_source_path(
    staging_root: Path,
    source_identity: SourceIdentity,
    source_path: str,
) -> Path:
    """Name a staged shard using its full pinned source identity and path."""

    identity_json = json.dumps(
        (
            source_identity.text_source,
            source_identity.repository,
            source_identity.revision,
            source_identity.path_prefix,
            source_identity.tree_inventory_sha256,
            source_path,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()
    return staging_root / f"{digest}-{PurePosixPath(source_path).name}"
