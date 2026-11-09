"""Prepare-time budget and the single log budget. Pure Python, inlined flat into carriers.

The judge keeps the last 16,000 characters of the log. Human lines get HUMAN_CAP, raw rounds RAW_CAP, and the
closing CK_TABLE + CK_RESULT at most TABLE_CAP + RESULT_CAP. Buffer raw records until finalization so profiler
output between cells cannot evict them. Output after prepare can still truncate evidence; ingestion checks hashes.
"""

import json
import time

TAIL = 16000
RESULT_CAP = 6000
TABLE_CAP = 1200
HUMAN_CAP = 6000
RAW_CAP = 2000  # 6000 + 2000 + 1200 + 6000 = 15200, leaving 800 for the entry's own prints


class BudgetExpired(RuntimeError):
    """Stop admitting work; never rank the interrupted timing round."""


class Budget:
    """Wall-clock stops against prepare()'s 300 s limit. `ok(near=False)` says whether another cell may start."""

    def __init__(self, t_start, stop_s=235.0, near_stop_s=266.0, clock=time.time):
        self.t_start, self.stop_s, self.near_stop_s, self.clock = t_start, stop_s, near_stop_s, clock
        self.skipped = []

    def elapsed(self):
        return self.clock() - self.t_start

    def ok(self, label, near=False):
        if self.elapsed() > (self.near_stop_s if near else self.stop_s):
            self.skipped.append("%s@%.0fs" % (label, self.elapsed()))
            return False
        return True

    def require(self, label, near=False):
        if not self.ok(label, near):
            raise BudgetExpired("budget exhausted before " + label)


def priority_order(keys, priority):
    """Cells run in `priority` order first, then the rest in their listed order."""
    first = [k for k in priority if k in keys]
    return first + [k for k in keys if k not in first]


class Emitter:
    """Every check line goes through here. `line` prints until its class cap is used, then drops (and counts);
    raw records wait for `final`, before CK_TABLE and CK_RESULT. `out` is the print function."""

    def __init__(self, out=None, human_cap=HUMAN_CAP, raw_cap=RAW_CAP, xr_cap=0):
        self.out = out or (lambda s: print(s, flush=True))
        self.caps = {"human": human_cap, "raw": raw_cap, "xr": xr_cap}
        self.used = {"human": 0, "raw": 0, "xr": 0}
        self.dropped = {"human": 0, "raw": 0, "xr": 0}
        self.raw_records = []

    def line(self, text, kind="human"):
        """kind "human" (prefixed CK), "raw" (a CK_RAW line, built by `raw`) or "xr" (an X-ray line, already
        prefixed XR; opportunity_map.py parses these)."""
        text = "CK " + text if kind == "human" else text
        if self.used[kind] + len(text) + 1 > self.caps[kind]:
            self.dropped[kind] += 1
            return False
        self.used[kind] += len(text) + 1
        if kind == "raw":
            self.raw_records.append(text)
        else:
            self.out(text)
        return True

    def line_xr(self, text):
        return self.line(text, "xr")

    def raw(self, key, samples):
        """Emit one complete cell at float round-trip precision, or none if RAW_CAP cannot hold it.
        CK_RESULT remains the bounded decision summary; raw output is opt-in and may omit entire cells."""
        return self.line("CK_RAW " + json.dumps(dict(key=key, **samples), separators=(",", ":")), "raw")

    def final(self, table_rows, result_line):
        """table_rows: short strings, one per shape. Rows beyond TABLE_CAP are cut; the result line must already
        fit RESULT_CAP (schema.trim)."""
        if len(result_line) > RESULT_CAP:
            raise ValueError("CK_RESULT is %d characters, cap %d" % (len(result_line), RESULT_CAP))
        used, kept = 0, []
        for row in table_rows:
            row = "CK_TABLE " + row
            if used + len(row) + 1 > TABLE_CAP:
                break
            used += len(row) + 1
            kept.append(row)
        for record in self.raw_records:
            self.out(record)
        for row in kept:
            self.out(row)
        self.out(result_line)
        return dict(dropped=dict(self.dropped), table_rows=len(kept))
