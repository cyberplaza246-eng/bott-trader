"""Surgical insert of cancel_take_profit into rithmic_connector.py (Paragan-safe)."""
from __future__ import annotations
import sys
from pathlib import Path

MARKER = '''            return bool(ok)
        except Exception as e:
            error_logger.error(f"Rithmic update_stop_loss error for {order_id}: {e}")
            return False

    async def _async_replace_protective_sl('''

def main(target: str, snippet_path: str) -> int:
    path = Path(target)
    text = path.read_text(encoding='utf-8')
    if 'def cancel_take_profit(' in text:
        print('cancel_take_profit already present — skip')
        return 0
    snippet = Path(snippet_path).read_text(encoding='utf-8')
    if not snippet.strip().startswith('def cancel_take_profit'):
        # allow indented
        pass
    # Ensure snippet ends with blank line before next method
    if not snippet.endswith('\n\n'):
        snippet = snippet.rstrip() + '\n\n'
    insert = MARKER.replace(
        '    async def _async_replace_protective_sl(',
        snippet + '    async def _async_replace_protective_sl(',
    )
    # Use CRLF-aware: normalize then restore
    nl = '\r\n' if '\r\n' in text else '\n'
    norm = text.replace('\r\n', '\n')
    marker_n = MARKER.replace('\r\n', '\n')
    insert_n = insert.replace('\r\n', '\n')
    if marker_n not in norm:
        print('ERROR: update_stop_loss -> _async_replace_protective_sl marker not found', file=sys.stderr)
        return 1
    if norm.count(marker_n) != 1:
        print('ERROR: marker not unique', norm.count(marker_n), file=sys.stderr)
        return 1
    out = norm.replace(marker_n, insert_n, 1)
    path.write_text(out.replace('\n', nl), encoding='utf-8', newline='')
    print('Inserted cancel_take_profit into', path)
    return 0

if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
