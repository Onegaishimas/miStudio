"""
Liveness stamping for long-running tasks.

THE PROBLEM THIS SOLVES, observed rather than imagined. Celery's result backend
holds whatever a task last reported. If the worker dies — a pod roll, an OOM
kill, a node drain — nothing writes a terminal state, so `AsyncResult.state`
returns PROGRESS forever. A band report that was killed by a deploy kept
reading as "profiling" for forty minutes, and three separate status checks
reported it as still working.

A STALE HEARTBEAT IS THE ONLY HONEST SIGNAL AVAILABLE. Celery offers no
"my worker vanished" event a poller can see: `acks_late` re-queues the task
(which would silently re-run a forty-minute GPU job), and `inspect().active()`
is a broadcast RPC too heavy to put behind a status endpoint the UI polls every
few seconds. So the task stamps a timestamp with every progress report, and a
reader compares it to the clock.

THE THRESHOLD MUST EXCEED THE SLOWEST GAP BETWEEN BEATS, or a genuinely slow
task is declared dead. That is why the tasks beat inside their loops rather
than only at stage boundaries: a stage boundary beat on a 45-minute stage would
force a threshold so generous the check stops being useful.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

#: How long a task may go without reporting before a reader treats it as dead.
#:
#: Generous on purpose. A false "dead" on a working task is worse than a slow
#: truth: it would send someone to re-run a job that was going to finish.
STALE_AFTER_SECONDS = 600


def beat(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Progress meta carrying a liveness timestamp.

    Every `update_state` on a long task should go through this. A progress
    report without a timestamp is indistinguishable from one made an hour ago.
    """
    meta: Dict[str, Any] = dict(extra or {})
    meta["heartbeat"] = time.time()
    return meta


def seconds_since_beat(info: Any, now: Optional[float] = None) -> Optional[float]:
    """Age of a task's last heartbeat, or None when it never sent one.

    None is NOT "stale". Tasks predating this, and tasks that have not reached
    their first progress report, legitimately have no heartbeat — reporting
    those as dead would condemn every short task that never beats at all.
    """
    if not isinstance(info, dict):
        return None
    stamp = info.get("heartbeat")
    if not isinstance(stamp, (int, float)):
        return None
    return max(0.0, (now if now is not None else time.time()) - float(stamp))


def looks_orphaned(state: str, info: Any, now: Optional[float] = None) -> bool:
    """Whether a task reporting progress has actually stopped reporting.

    Only ever true for a task claiming to be in progress. A SUCCESS or FAILURE
    is terminal and its age says nothing — a report finished last week is not
    orphaned, it is done.
    """
    if state not in ("PROGRESS", "STARTED"):
        return False
    age = seconds_since_beat(info, now=now)
    return age is not None and age > STALE_AFTER_SECONDS
