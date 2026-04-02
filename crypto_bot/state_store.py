"""
state_store.py — Atomic JSON state persistence for the crypto bot.
"""

from __future__ import annotations

import json
import os
import time

from crypto_bot.models import RuntimeState


class StateStore:
    def __init__(self, state_dir: str):
        os.makedirs(state_dir, exist_ok=True)
        self._path = os.path.join(state_dir, "runtime_state.json")
        self._last_save = 0.0
        self._dirty = False

    def load(self) -> RuntimeState | None:
        if not os.path.exists(self._path):
            return None
        with open(self._path) as f:
            return RuntimeState.from_dict(json.load(f))

    def save(self, state: RuntimeState, force: bool = False):
        now = time.time()
        if not force and now - self._last_save < 1.0:
            self._dirty = True
            return
        state.touch()
        tmp = self._path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
        os.replace(tmp, self._path)
        self._last_save = now
        self._dirty = False

    def save_if_dirty(self, state: RuntimeState):
        if self._dirty:
            self.save(state, force=True)
