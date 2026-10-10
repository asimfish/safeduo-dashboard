"""Durable diagnostics only. Never steps or reads native physics."""
from pathlib import Path
import datetime
import faulthandler
import json
import os
import time

class PhaseTrace:
    def __init__(self, parent, *, interval_seconds=60):
        parent = Path(parent).resolve()
        allowed = Path('/mnt/nas/data/lyf/double_hand/safety_verified_fallback_20261010_0352/astra_full74_camera_repair_development_v1')
        if not parent.is_relative_to(allowed) or not 0 < interval_seconds <= 60:
            raise ValueError('exclusive owned native diagnostics scope/interval')
        self.root = parent / 'native_initialization_diagnostics_v1'
        self.root.mkdir(exist_ok=False)
        self.events = (self.root / 'phases.jsonl').open('x')
        self.stacks = (self.root / 'python_stacks.log').open('x')
        self.closed = False
        self.phase('DIAGNOSTICS_READY')
        faulthandler.dump_traceback_later(interval_seconds, repeat=True, file=self.stacks, exit=False)

    def phase(self, name):
        if self.closed:
            raise ValueError('diagnostics already closed')
        self.events.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            pid=os.getpid(), phase=name, monotonic_seconds=time.monotonic(),
            process_cpu_seconds=time.process_time(), diagnostic_only=True,
            physical_safety_certified=False), allow_nan=False) + '\n')
        self.events.flush()
        os.fsync(self.events.fileno())

    def close(self):
        if self.closed:
            return
        self.phase('DIAGNOSTICS_CLOSED_BEFORE_APP_SHUTDOWN')
        faulthandler.cancel_dump_traceback_later()
        self.stacks.flush()
        os.fsync(self.stacks.fileno())
        self.events.close()
        self.stacks.close()
        self.closed = True
