"""Action schedule with freshest-observation-wins merging for DRTC.

Maintains a sorted schedule of future actions keyed by action step,
supporting merge of overlapping chunks from different inference calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sortedcontainers import SortedDict


@dataclass
class ScheduledAction:
    """An action scheduled for execution at a specific step.

    Attributes:
        action: The action array to execute.
        src_control_step: The control-loop tick t that produced this action (freshness key).
        chunk_start_step: The action step n_k where the source chunk starts (for RTC offset math).
    """

    action: np.ndarray
    src_control_step: int
    chunk_start_step: int


@dataclass
class MergeStats:
    """Statistics from merging an action chunk into the schedule.

    Attributes:
        overlap_count: Number of overlapping actions compared.
        mean_l2: Mean L2 distance across overlapping actions.
        max_l2: Maximum L2 distance across overlapping actions.
    """

    overlap_count: int
    mean_l2: float
    max_l2: float


@dataclass
class TimedAction:
    """An action with timing and step metadata from the server."""

    action: np.ndarray
    action_step: int
    src_control_step: int
    chunk_start_step: int


class ActionSchedule:
    """Sorted schedule of future actions with freshest-observation-wins merging."""

    def __init__(self):
        self._schedule: SortedDict = SortedDict()

    def __len__(self) -> int:
        return len(self._schedule)

    def pop_front(self) -> tuple[int, np.ndarray, int, int] | None:
        """Pop and return the first (lowest action step) scheduled action.

        Returns:
            Tuple of (step, action, src_control_step, chunk_start_step) or None if empty.
        """
        if not self._schedule:
            return None
        step, scheduled = self._schedule.popitem(0)
        return step, scheduled.action, scheduled.src_control_step, scheduled.chunk_start_step

    def get_masking_chunk_spans(
        self, *, current_step: int, max_len: int
    ) -> list[tuple[int, int, int]] | None:
        """Get list of (src_control_step, start_idx, end_idx) spans for RTC masking prefix.

        Returns information needed to look up raw actions in the server's cache
        (keyed by src_control_step). The offset within a cached chunk is
        step - scheduled.chunk_start_step.

        Args:
            current_step: The current action step being executed.
            max_len: Total number of actions to include.

        Returns:
            List of (src_control_step, start_idx, end_idx) tuples or None if empty.
        """
        if max_len <= 0:
            return None

        chunks: list[tuple[int, int, int]] = []
        current_src_control_step: int | None = None
        current_start: int | None = None
        current_end: int = 0
        count = 0

        for step, scheduled in self._schedule.items():
            if step <= current_step:
                continue

            chunk_idx = step - scheduled.chunk_start_step

            if current_src_control_step is None:
                current_src_control_step = scheduled.src_control_step
                current_start = chunk_idx
                current_end = chunk_idx + 1
            elif scheduled.src_control_step == current_src_control_step and chunk_idx == current_end:
                current_end = chunk_idx + 1
            else:
                if current_start is not None:
                    chunks.append((current_src_control_step, current_start, current_end))
                current_src_control_step = scheduled.src_control_step
                current_start = chunk_idx
                current_end = chunk_idx + 1

            count += 1
            if count >= max_len:
                break

        if current_src_control_step is not None and current_start is not None:
            chunks.append((current_src_control_step, current_start, current_end))

        return chunks if chunks else None

    def get_size(self) -> int:
        return len(self._schedule)

    def is_empty(self) -> bool:
        return len(self._schedule) == 0

    def merge(
        self,
        incoming_actions: list[TimedAction],
        src_control_step: int,
        chunk_start_step: int,
        current_action_step: int,
        logger: logging.Logger | None = None,
    ) -> MergeStats:
        """Merge incoming actions using freshest-observation-wins strategy.

        Args:
            incoming_actions: List of TimedAction from the server.
            src_control_step: The control-loop tick t that produced this chunk.
            chunk_start_step: The action step n_k where this chunk starts.
            current_action_step: The most recently executed action step.
            logger: Optional logger for debug output.

        Returns:
            MergeStats with L2 discrepancy metrics for overlapping actions.
        """
        stale_count = 0
        inserted_count = 0
        updated_count = 0
        l2_distances: list[float] = []

        for timed_action in incoming_actions:
            step = timed_action.action_step
            action = timed_action.action

            if step <= current_action_step:
                stale_count += 1
                continue

            existing = self._schedule.get(step)
            if existing is None:
                self._schedule[step] = ScheduledAction(
                    action=action, src_control_step=src_control_step, chunk_start_step=chunk_start_step
                )
                inserted_count += 1
                continue

            old_arr = np.asarray(existing.action, dtype=np.float32).reshape(-1)
            new_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            if old_arr.shape == new_arr.shape and old_arr.size > 0:
                l2 = float(np.linalg.norm(new_arr - old_arr))
                l2_distances.append(l2)

            if src_control_step > existing.src_control_step:
                self._schedule[step] = ScheduledAction(
                    action=action, src_control_step=src_control_step, chunk_start_step=chunk_start_step
                )
                updated_count += 1

        if logger and stale_count:
            logger.debug(
                "Merge stats: %d stale, %d inserted, %d updated",
                stale_count, inserted_count, updated_count,
            )

        overlap_count = len(l2_distances)
        if overlap_count > 0:
            mean_l2 = float(np.mean(l2_distances))
            max_l2 = float(np.max(l2_distances))
        else:
            mean_l2 = 0.0
            max_l2 = 0.0

        return MergeStats(overlap_count=overlap_count, mean_l2=mean_l2, max_l2=max_l2)

    def clear(self) -> None:
        """Clear all scheduled actions."""
        self._schedule.clear()
