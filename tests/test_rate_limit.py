from app.services import rate_limit
from app.services.rate_limit import LoginRateLimiter


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make_limiter(monkeypatch, max_attempts=3, window_seconds=60):
    clock = FakeClock()
    monkeypatch.setattr(rate_limit.time, "monotonic", clock)
    return LoginRateLimiter(max_attempts, window_seconds), clock


def test_blocks_after_max_attempts(monkeypatch):
    limiter, _ = make_limiter(monkeypatch)

    for _ in range(2):
        limiter.register_failure("ip")
    assert limiter.retry_after("ip") == 0

    limiter.register_failure("ip")
    assert limiter.retry_after("ip") == 60


def test_unblocks_when_window_expires(monkeypatch):
    limiter, clock = make_limiter(monkeypatch)
    for _ in range(3):
        limiter.register_failure("ip")

    clock.now += 45
    assert limiter.retry_after("ip") == 15

    clock.now += 15
    assert limiter.retry_after("ip") == 0


def test_keys_are_independent(monkeypatch):
    limiter, _ = make_limiter(monkeypatch)
    for _ in range(3):
        limiter.register_failure("ip-a")

    assert limiter.retry_after("ip-a") > 0
    assert limiter.retry_after("ip-b") == 0


def test_expired_keys_are_purged_when_tracking_too_many(monkeypatch):
    limiter, clock = make_limiter(monkeypatch)
    monkeypatch.setattr(LoginRateLimiter, "MAX_TRACKED_KEYS", 10)
    for i in range(10):
        limiter.register_failure(f"user-{i}")

    clock.now += 61
    limiter.register_failure("new-user")

    assert list(limiter._failures) == ["new-user"]
