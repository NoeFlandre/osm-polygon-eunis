from pathlib import Path

import pytest

from osm_polygon_eunis.options import BatchLimits, ReleaseOptions, ShardContext


def test_batch_limits_collect_existing_processing_defaults() -> None:
    limits = BatchLimits()

    assert limits.workers == 8
    assert limits.parquet_batch_size == 256
    assert limits.retained_source_shards_per_worker == 128
    assert limits.raster_groups_per_batch == 2
    assert limits.geometry_tasks_per_worker == 4


def test_release_options_accept_one_configurable_batch_limits_value() -> None:
    limits = BatchLimits(
        workers=3,
        parquet_batch_size=64,
        retained_source_shards_per_worker=12,
        raster_groups_per_batch=1,
        geometry_tasks_per_worker=2,
    )
    options = ReleaseOptions(
        reference_config=Path("reference.json"),
        workdir=Path("run"),
        limits=limits,
    )

    assert options.limits is limits
    assert options.workdir == Path("run")


@pytest.mark.parametrize(
    "field",
    [
        "workers",
        "parquet_batch_size",
        "retained_source_shards_per_worker",
        "raster_groups_per_batch",
        "geometry_tasks_per_worker",
    ],
)
def test_batch_limits_reject_non_positive_values(field: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        BatchLimits(**{field: 0})


def test_shard_context_keeps_publication_scope_together() -> None:
    api = object()
    plan = object()
    options = object()
    http_client = object()
    context = ShardContext(
        api=api,
        plan=plan,
        options=options,
        http_client=http_client,
    )

    assert context.api is api
    assert context.plan is plan
    assert context.options is options
    assert context.http_client is http_client
