from temporun_pipeline.progress import ProgressMeter, format_duration


def test_format_duration() -> None:
    assert format_duration(None) == "--"
    assert format_duration(59) == "59s"
    assert format_duration(65) == "1m 05s"
    assert format_duration(3_661) == "1h 01m"
    assert format_duration(90_061) == "1d 01h 01m"


def test_progress_meter_eta_and_rolling_window() -> None:
    meter = ProgressMeter(
        total=1_000,
        initial_completed=100,
        window_size=2,
        started_at=10.0,
    )
    first = meter.snapshot(200, now=20.0)
    assert first["rate"] == 10.0
    assert first["eta_seconds"] == 80.0

    meter.snapshot(300, now=30.0)
    latest = meter.snapshot(500, now=40.0)
    assert latest["rate"] == 15.0
    assert latest["eta_seconds"] == 500 / 15


def test_progress_render() -> None:
    meter = ProgressMeter(total=100, started_at=0.0)
    text = meter.render(25, unit="frames", now=5.0)
    assert "25/100 frames (25.00%)" in text
    assert "rate=5.00 frames/s" in text
    assert "ETA=15s" in text
