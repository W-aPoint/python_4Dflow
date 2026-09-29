"""为工作流重要阶段提供可复用的耗时记录"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator


@dataclass(frozen=True, slots=True)
class TimingRecord:
    """一个命名工作流阶段的墙钟耗时"""

    step_name: str
    elapsed_seconds: float

    @property
    def elapsed_milliseconds(self) -> float:
        """以毫秒返回同一段耗时"""

        return self.elapsed_seconds * 1000.0


@dataclass(slots=True)
class RuntimeTimer:
    """使用单调高分辨率时钟按顺序收集耗时记录"""

    _started_at: float = field(init=False, repr=False)
    _records: list[TimingRecord] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._started_at = perf_counter()

    @contextmanager
    def measure(self, step_name: str) -> Iterator[None]:
        """测量一个重要阶段，并在退出时追加记录"""

        name = step_name.strip()
        if not name:
            raise ValueError("计时步骤名称不能为空")

        started_at = perf_counter()
        try:
            yield
        finally:
            self._records.append(
                TimingRecord(
                    step_name=name,
                    elapsed_seconds=max(0.0, perf_counter() - started_at),
                )
            )

    @property
    def records(self) -> tuple[TimingRecord, ...]:
        """返回已完成阶段耗时的不可变快照"""

        return tuple(self._records)

    @property
    def total_elapsed_seconds(self) -> float:
        """返回计时器创建以来经过的墙钟秒数"""

        return max(0.0, perf_counter() - self._started_at)
