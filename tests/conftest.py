from hypothesis import HealthCheck, settings

settings.register_profile(
    "deterministic",
    settings(
        deadline=None,
        derandomize=True,
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
    ),
)
settings.load_profile("deterministic")
