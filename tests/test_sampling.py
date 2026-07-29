from PIL import Image

from temporun_pipeline.sampling import (
    fine_scan_times,
    resize_keep_aspect,
    sample_frame_times,
)


def test_short_shot_uses_midpoint():
    values = sample_frame_times(1_000, 3_000)
    assert [(value.frame_ms, value.sample) for value in values] == [(2_000, "mid")]


def test_medium_shot_uses_thirds():
    values = sample_frame_times(1_000, 4_000)
    assert [value.frame_ms for value in values] == [2_000, 3_000]


def test_long_shot_uses_quarter_fps():
    values = sample_frame_times(0, 13_000, long_shot_fps=0.25)
    assert [value.frame_ms for value in values] == [2_000, 6_000, 10_000]


def test_reranker_long_shot_uses_one_fps():
    values = sample_frame_times(0, 5_500, long_shot_fps=1.0)
    assert [value.frame_ms for value in values] == [
        500,
        1_500,
        2_500,
        3_500,
        4_500,
    ]


def test_fine_scan_is_eight_fps_and_clamped():
    values = fine_scan_times(1_000, 500, 1_500, window_ms=750, fps=8)
    assert values[0] == 500
    assert values[-1] == 1_500
    assert 1_000 in values
    assert all(right - left == 125 for left, right in zip(values, values[1:]))


def test_resize_preserves_aspect_ratio():
    image = Image.new("RGB", (1920, 1080))
    resized = resize_keep_aspect(image, 512)
    assert resized.size == (512, 288)
