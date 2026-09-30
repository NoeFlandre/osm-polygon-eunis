"""Reference staging and worker-cache lifecycle behavior."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from osm_polygon_eunis import reference_cache, reference_staging
from osm_polygon_eunis._protocols import StreamClient
from osm_polygon_eunis.eea import EeaGroup, RemoteAsset


def _asset(path: str, *, code: str | None = "R11") -> RemoteAsset:
    return RemoteAsset(
        path=path,
        url=f"https://example.test/{path.lstrip('/')}",
        size=1,
        etag="etag",
        code=code,
        name="steppe" if code else None,
        record_id="record",
        source_version="EEA-test",
    )


def test_staged_asset_digest_rejects_missing_or_mismatched_artifacts(tmp_path: Path) -> None:
    asset = _asset("/reference.tif")
    path = tmp_path / "reference.tif"

    assert reference_cache._staged_asset_digest(asset, path) is None
    path.write_bytes(b"too long")
    assert reference_cache._staged_asset_digest(asset, path) is None
    path.write_bytes(b"x")
    assert reference_cache._staged_asset_digest(asset, path) is None


def test_staged_asset_digest_rejects_invalid_receipts(tmp_path: Path) -> None:
    asset = _asset("/reference.tif")
    path = tmp_path / "reference.tif"
    path.write_bytes(b"x")
    receipt_path = reference_cache._asset_cache_receipt_path(path)
    matching_asset = reference_cache._asset_cache_identity(asset)
    receipts = (
        "not JSON",
        json.dumps([]),
        json.dumps({"asset": {}, "sha256": "digest"}),
        json.dumps({"asset": matching_asset, "sha256": None}),
    )

    for receipt in receipts:
        receipt_path.write_text(receipt, encoding="utf-8")
        assert reference_cache._staged_asset_digest(asset, path) is None


def test_staged_asset_digest_handles_receipt_read_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = _asset("/reference.tif")
    path = tmp_path / "reference.tif"
    path.write_bytes(b"x")
    receipt_path = reference_cache._asset_cache_receipt_path(path)
    receipt_path.write_text("{}", encoding="utf-8")
    original_read_text = Path.read_text

    def fail_for_receipt(target: Path, *args, **kwargs):
        if target == receipt_path:
            raise OSError("receipt became unavailable")
        return original_read_text(target, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_for_receipt)

    assert reference_cache._staged_asset_digest(asset, path) is None


def test_open_reference_group_reuses_client_for_raster_and_vector(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeReference:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    downloaded: list[tuple[str, object]] = []

    def fake_download(client, asset, destination):
        downloaded.append((asset.path, client))
        destination.write_bytes(b"asset")
        return f"sha-{asset.path}"

    monkeypatch.setattr(reference_cache, "download_asset", fake_download)
    monkeypatch.setattr(reference_staging, "RasterReference", FakeReference)
    monkeypatch.setattr(reference_staging, "GeoPackageReference", FakeReference)
    client = cast(StreamClient, object())
    checksums: dict[str, str] = {}
    raster_group = EeaGroup(
        "raster-record",
        "raster",
        "folder",
        "service",
        {"R11": "steppe"},
        (_asset("/Prob_R11.tif"),),
        None,
    )
    vector_group = EeaGroup(
        "vector-record",
        "vector",
        "folder",
        "service",
        {"Q11": "bog"},
        (),
        _asset("/habitats.gpkg", code=None),
    )

    with reference_staging.open_reference_group(
        raster_group, tmp_path / "raster", threshold=0, checksums=checksums, client=client
    ) as reference:
        assert isinstance(reference, FakeReference)
    with reference_staging.open_reference_group(
        vector_group, tmp_path / "vector", threshold=1, checksums=checksums, client=client
    ) as reference:
        assert isinstance(reference, FakeReference)

    assert downloaded == [("/Prob_R11.tif", client), ("/habitats.gpkg", client)]
    assert checksums == {
        "raster-record:/Prob_R11.tif": "sha-/Prob_R11.tif",
        "vector-record:/habitats.gpkg": "sha-/habitats.gpkg",
    }

    empty_group = EeaGroup("empty", "empty", "folder", "service", {}, (), None)
    with (
        pytest.raises(ValueError, match="no reference asset"),
        reference_staging.open_reference_group(
            empty_group, tmp_path / "empty", threshold=0, client=client
        ),
    ):
        pass


class _TrackedReference:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.closed = True


def test_worker_reference_batch_opens_local_groups_once_and_closes_on_exit(
    tmp_path: Path, monkeypatch
) -> None:
    opened: list[_TrackedReference] = []

    def fake_reference(*args, **kwargs):
        opened.append(_TrackedReference(*args, **kwargs))
        return opened[-1]

    monkeypatch.setattr(reference_staging, "RasterReference", fake_reference)
    monkeypatch.setattr(reference_staging, "GeoPackageReference", fake_reference)
    raster_group = EeaGroup(
        "raster-record", "raster", "folder", "service", {}, (_asset("/Prob_R11.tif"),), None
    )
    vector_group = EeaGroup(
        "vector-record",
        "vector",
        "folder",
        "service",
        {"Q11": "bog"},
        (),
        _asset("/habitats.gpkg", code=None),
    )
    groups = (raster_group, vector_group)

    first = reference_staging._worker_reference_batch(groups, tmp_path, 3, start_index=2)
    second = reference_staging._worker_reference_batch(groups, tmp_path, 3, start_index=2)
    third = reference_staging._worker_reference_batch((raster_group,), tmp_path, 3, start_index=4)

    assert first is second
    assert first == tuple(opened[:2])
    layers = opened[0].args[0]
    assert [(layer.code, layer.name, layer.path) for layer in layers] == [
        ("R11", "steppe", tmp_path / "02-raster-r" / "R11.tif")
    ]
    assert opened[0].kwargs == {"threshold": 3}
    assert opened[1].args[0] == tmp_path / "03-vector-r" / "habitats.gpkg"
    assert len(reference_staging._WORKER_REFERENCES) == 1
    assert all(cast(_TrackedReference, reference).closed for reference in first)
    assert not cast(_TrackedReference, third[0]).closed
    reference_staging._close_worker_reference_cache()
    assert all(reference.closed for reference in opened)
    assert reference_staging._WORKER_REFERENCES == {}


def test_worker_reference_batch_closes_opened_groups_on_failure(
    tmp_path: Path, monkeypatch
) -> None:
    opened: list[_TrackedReference] = []

    def fake_reference(*args, **kwargs):
        opened.append(_TrackedReference(*args, **kwargs))
        return opened[-1]

    monkeypatch.setattr(reference_staging, "RasterReference", fake_reference)
    raster_group = EeaGroup(
        "raster-record", "raster", "folder", "service", {}, (_asset("/Prob_R11.tif"),), None
    )
    empty_group = EeaGroup("empty", "empty", "folder", "service", {}, (), None)

    with pytest.raises(ValueError, match="no reference asset"):
        reference_staging._worker_reference_batch(
            (raster_group, empty_group), tmp_path, 0, start_index=0
        )

    assert [reference.closed for reference in opened] == [True]
    assert reference_staging._WORKER_REFERENCES == {}


def test_reference_assets_for_staging_rejects_assetless_group() -> None:
    group = EeaGroup("empty", "empty", "folder", "service", {}, (), None)

    with pytest.raises(ValueError, match="no reference asset"):
        reference_staging._reference_assets_for_staging(group)


def test_staged_reference_assets_use_grid_scratch_and_reuse_verified_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = RemoteAsset(
        path="/Prob_R11_10m.tif",
        url="https://example.test/Prob_R11_10m.tif",
        size=5,
        etag="etag-v1",
        code="R11",
        name="steppe",
        record_id="record",
        source_version="EEA-test",
    )
    group = EeaGroup("record", "raster", "folder", "service", {}, (asset,), None)
    reference_root = tmp_path / "node-scratch" / "reference"
    workdir = tmp_path / "persistent" / "runs"
    downloads: list[Path] = []
    digest = hashlib.sha256(b"asset").hexdigest()

    def fake_download(_client, _asset, destination: Path) -> str:
        downloads.append(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"asset")
        return digest

    monkeypatch.setenv("EUNIS_REFERENCE_DIR", str(reference_root))
    monkeypatch.setattr(reference_cache, "download_asset", fake_download)
    client = cast(StreamClient, object())

    for _ in range(2):
        checksums: dict[str, str] = {}
        with reference_staging._stage_reference_groups(
            (group,), workdir=workdir, checksums=checksums, client=client
        ) as staged_root:
            staged_asset = staged_root / "00-record" / "R11.tif"
            assert staged_root == reference_root
            assert staged_asset.read_bytes() == b"asset"
            assert checksums == {"record:/Prob_R11_10m.tif": digest}

    assert len(downloads) == 1
    assert staged_asset.is_file()

    staged_asset.write_bytes(b"other")
    with reference_staging._stage_reference_groups(
        (group,), workdir=workdir, checksums={}, client=client
    ):
        pass

    assert len(downloads) == 2
    assert staged_asset.read_bytes() == b"asset"


def test_reference_staging_removes_only_stale_partial_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_root = tmp_path / "persistent" / "cache" / "reference"
    stale_partial = reference_root / "00-record" / ".R11.tif.abcd.part"
    stale_partial.parent.mkdir(parents=True)
    stale_partial.write_bytes(b"incomplete")
    cached_asset = stale_partial.parent / "R11.tif"
    cached_asset.write_bytes(b"complete")
    monkeypatch.setenv("EUNIS_REFERENCE_DIR", str(reference_root))

    with reference_staging._reference_staging_root(tmp_path / "runs", "record"):
        pass

    assert not stale_partial.exists()
    assert cached_asset.read_bytes() == b"complete"


@pytest.mark.parametrize("kind", ["raster", "vector"])
def test_open_reference_group_accepts_optional_checksum_collection(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeReference:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

        def __enter__(self) -> FakeReference:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(reference_staging, "RasterReference", FakeReference)
    monkeypatch.setattr(reference_staging, "GeoPackageReference", FakeReference)
    monkeypatch.setattr(
        reference_cache,
        "download_asset",
        lambda _client, asset, path: (path.write_bytes(b"asset"), f"sha-{asset.path}")[1],
    )
    if kind == "raster":
        group = EeaGroup("raster", "raster", "folder", "service", {}, (_asset("/r.tif"),), None)
    else:
        group = EeaGroup(
            "vector", "vector", "folder", "service", {}, (), _asset("/v.gpkg", code=None)
        )

    with reference_staging.open_reference_group(
        group,
        tmp_path / kind,
        threshold=0,
        client=cast(StreamClient, object()),
    ) as reference:
        assert isinstance(reference, FakeReference)


def test_indexed_reference_batches_cover_raster_and_vector_transitions() -> None:
    rasters = tuple(
        EeaGroup(
            f"raster-{index}", "raster", "folder", "service", {}, (_asset(f"/{index}.tif"),), None
        )
        for index in range(3)
    )
    vectors = tuple(
        EeaGroup(
            f"vector-{index}",
            "vector",
            "folder",
            "service",
            {},
            (),
            _asset(f"/{index}.gpkg", code=None),
        )
        for index in range(2)
    )

    batches = reference_staging._indexed_reference_group_batches((*rasters, *vectors))

    assert batches == ((0, rasters[:2]), (2, (rasters[2],)), (3, vectors))


def test_stage_reference_groups_downloads_each_asset_and_cleans_temporary_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raster = EeaGroup(
        "raster-record", "raster", "folder", "service", {}, (_asset("/Prob_R11.tif"),), None
    )
    vector = EeaGroup(
        "vector-record",
        "vector",
        "folder",
        "service",
        {},
        (),
        _asset("/habitats.gpkg", code=None),
    )
    downloaded: list[tuple[str, Path]] = []

    def fake_download(_client, asset, destination: Path) -> str:
        destination.write_bytes(b"reference")
        downloaded.append((asset.path, destination))
        return f"sha-{asset.path}"

    monkeypatch.setattr(reference_cache, "download_asset", fake_download)
    checksums: dict[str, str] = {}
    with reference_staging._stage_reference_groups(
        (raster, vector),
        workdir=tmp_path,
        checksums=checksums,
        client=cast(StreamClient, object()),
    ) as root:
        assert root.is_dir()
        assert [path.relative_to(root).as_posix() for _, path in downloaded] == [
            "00-raster-r/R11.tif",
            "01-vector-r/habitats.gpkg",
        ]

    assert checksums == {
        "raster-record:/Prob_R11.tif": "sha-/Prob_R11.tif",
        "vector-record:/habitats.gpkg": "sha-/habitats.gpkg",
    }
    assert not root.exists()
