from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    total_seconds = int(round(seconds))
    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


@dataclass
class ProgressMeter:
    total: int
    initial_completed: int = 0
    window_size: int = 8
    started_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.total < 0:
            raise ValueError("total must be non-negative")
        if not 0 <= self.initial_completed <= self.total:
            raise ValueError("initial_completed must be in [0, total]")
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        self._samples: deque[tuple[float, int]] = deque(
            [(self.started_at, self.initial_completed)],
            maxlen=self.window_size + 1,
        )

    def snapshot(
        self,
        completed: int,
        *,
        now: float | None = None,
    ) -> dict[str, float | int | None]:
        completed = min(max(int(completed), 0), self.total)
        current_time = time.monotonic() if now is None else float(now)
        if self._samples and completed < self._samples[-1][1]:
            raise ValueError("completed progress cannot move backwards")
        self._samples.append((current_time, completed))
        sample_time, sample_completed = self._samples[0]
        sample_seconds = current_time - sample_time
        sample_delta = completed - sample_completed
        rate = (
            sample_delta / sample_seconds
            if sample_seconds > 0 and sample_delta > 0
            else None
        )
        remaining = max(self.total - completed, 0)
        eta_seconds = remaining / rate if rate and remaining else (0.0 if not remaining else None)
        percent = (completed / self.total * 100.0) if self.total else 100.0
        return {
            "completed": completed,
            "total": self.total,
            "percent": percent,
            "rate": rate,
            "eta_seconds": eta_seconds,
            "elapsed_seconds": max(current_time - self.started_at, 0.0),
        }

    def render(
        self,
        completed: int,
        *,
        unit: str,
        now: float | None = None,
    ) -> str:
        snapshot = self.snapshot(completed, now=now)
        rate = snapshot["rate"]
        rate_text = f"{rate:.2f} {unit}/s" if isinstance(rate, float) else "--"
        return (
            f"{snapshot['completed']:,}/{snapshot['total']:,} {unit} "
            f"({snapshot['percent']:.2f}%) rate={rate_text} "
            f"ETA={format_duration(snapshot['eta_seconds'])} "
            f"elapsed={format_duration(snapshot['elapsed_seconds'])}"
        )
