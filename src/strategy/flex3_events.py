"""Opt-in, append-only JSONL event log for the read-only live chart (EVAL PASSER / FUNDED 12K).
OFF unless env FLEX3_EVENT_LOG is set to a file path. Never raises, never touches orders or trading state.
Events: entry, sl_move, exit. One JSON object per line; 't' = UTC ISO time."""
import json, os
from datetime import datetime, timezone
PATH = os.getenv("FLEX3_EVENT_LOG", "").strip()
def emit(event, **kw):
    if not PATH: return
    try:
        rec = {"t": datetime.now(timezone.utc).isoformat(), "event": event}
        rec.update({k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in kw.items()})
        d = os.path.dirname(PATH)
        if d: os.makedirs(d, exist_ok=True)
        with open(PATH, "a", encoding="utf-8") as fh: fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass
