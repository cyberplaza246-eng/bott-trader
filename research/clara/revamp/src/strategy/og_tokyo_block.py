"""OG bot Tokyo-open entry block (OG_TOKYO_BLOCK_MIN, default 0 = off).
Blocks NEW OG entries in the N minutes after 09:00 Asia/Tokyo (DST-aware via ET).
Never used for exits / SL-TP / open positions."""
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')
TOKYO = ZoneInfo('Asia/Tokyo')


def block_minutes() -> int:
    try:
        return max(0, int(float(os.getenv('OG_TOKYO_BLOCK_MIN', '0') or 0)))
    except ValueError:
        return 0


def tokyo_window_end(now=None, minutes=None):
    """Return window end (ET-aware) if `now` is inside [Tokyo 09:00, +minutes), else None."""
    m = block_minutes() if minutes is None else int(minutes)
    if m <= 0:
        return None
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    tk = now.astimezone(TOKYO)
    start = tk.replace(hour=9, minute=0, second=0, microsecond=0)
    end = start + timedelta(minutes=m)
    if start <= tk < end:
        return end.astimezone(ET)
    return None
