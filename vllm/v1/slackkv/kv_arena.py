# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU ownership of a fixed, homogeneous GPU LayerPage arena."""

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import msgspec

from vllm.v1.slackkv.protocol import PageHandle, RequestRef


class PageOwner(msgspec.Struct, frozen=True):
    ref: RequestRef
    layer: int
    logical_block: int


@dataclass(slots=True)
class ArenaPage:
    generation: int = 0
    state: str = "FREE"
    owner: PageOwner | None = None


class LayerPageArena:
    """Own pages; CUDA storage belongs to the worker, never to this object."""

    def __init__(self, capacity: int, page_bytes: int):
        if capacity < 2 or page_bytes <= 0:
            raise ValueError("LayerPage arena requires usable pages and page bytes")
        self.capacity = capacity
        self.page_bytes = page_bytes
        self.pages = [ArenaPage() for _ in range(capacity)]
        self.pages[0].state = "RESERVED"
        self.free_ids = deque(range(1, capacity))
        self.counts = {
            "FREE": capacity - 1, "RESERVED": 1, "LIVE": 0, "RETIRING": 0,
        }
        self.observers: list[Callable] = []

    def add_observer(self, observer: Callable):
        self.observers.append(observer)

    def _notify(self, action: str, handles: tuple[PageHandle, ...]):
        if not handles:
            return
        for observer in self.observers:
            observer(action, handles)

    def _page(self, handle: PageHandle, expected: tuple[str, ...]):
        if not 0 < handle.page_id < self.capacity:
            raise RuntimeError("invalid or system LayerPage handle")
        page = self.pages[handle.page_id]
        if page.generation != handle.generation or page.state not in expected:
            raise RuntimeError(
                f"stale LayerPage handle/state: {handle!r}; {page!r}"
            )
        return page

    def reserve(self, owners: tuple[PageOwner, ...]):
        pages = self.pages
        free_ids = self.free_ids
        counts = self.counts
        if len(owners) > counts["FREE"]:
            return None
        reserved = []
        try:
            for owner in owners:
                page_id = free_ids[0]
                page = pages[page_id]
                if page.state != "FREE" or page.owner is not None:
                    raise RuntimeError("LayerPage free queue ownership mismatch")
                new_generation = page.generation + 1
                handle = PageHandle(page_id, new_generation)
                reserved.append(handle)
                free_ids.popleft()
                page.generation = new_generation
                page.state = "RESERVED"
                page.owner = owner
                counts["FREE"] -= 1
                counts["RESERVED"] += 1
            result = tuple(reserved)
        except BaseException:
            # Only this unpublished reservation is rolled back.
            self.rollback(tuple(reserved))
            raise
        try:
            self._notify("reserve", result)
        except BaseException:
            self.rollback(result)
            raise
        return result

    def commit(self, handles: tuple[PageHandle, ...]):
        for handle in handles:
            self._page(handle, ("RESERVED",))
        for handle in handles:
            self.pages[handle.page_id].state = "LIVE"
        self.counts["RESERVED"] -= len(handles)
        self.counts["LIVE"] += len(handles)
        self._notify("commit", handles)

    def rollback(self, handles: tuple[PageHandle, ...]):
        for handle in handles:
            self._page(handle, ("RESERVED",))
        for handle in reversed(handles):
            page = self.pages[handle.page_id]
            page.state = "FREE"
            page.owner = None
            self.free_ids.appendleft(handle.page_id)
        self.counts["RESERVED"] -= len(handles)
        self.counts["FREE"] += len(handles)
        self._notify("rollback", handles)

    def retire(self, handles: tuple[PageHandle, ...]):
        for handle in handles:
            self._page(handle, ("LIVE",))
        for handle in handles:
            self.pages[handle.page_id].state = "RETIRING"
        self.counts["LIVE"] -= len(handles)
        self.counts["RETIRING"] += len(handles)
        self._notify("retire", handles)

    def reclaim(self, handles: tuple[PageHandle, ...]):
        for handle in handles:
            self._page(handle, ("LIVE", "RETIRING"))
        pages = self.pages
        counts = self.counts
        free_ids = self.free_ids
        for handle in handles:
            page = pages[handle.page_id]
            counts[page.state] -= 1
            page.state = "FREE"
            page.owner = None
            counts["FREE"] += 1
            free_ids.append(handle.page_id)
        self._notify("reclaim", handles)

    def validate_owners(
        self, handles: Iterable[PageHandle], ref: RequestRef,
    ):
        for handle in handles:
            page = self._page(handle, ("LIVE",))
            owner = page.owner
            if owner is None or (owner.ref is not ref and owner.ref != ref):
                raise RuntimeError("LayerPage belongs to another request")

    def snapshot(self):
        return {
            "unit": "LayerPage", "capacity": self.capacity,
            "system_reserved": 1, "page_bytes": self.page_bytes,
            "arena_bytes": self.capacity * self.page_bytes,
            **self.counts,
            "bytes": {k: v * self.page_bytes for k, v in self.counts.items()},
        }
