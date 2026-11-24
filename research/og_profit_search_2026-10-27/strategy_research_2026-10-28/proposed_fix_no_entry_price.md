# Proposed fix: "SL move skipped ... no entry price" (NOT APPLIED - read-only diagnosis, 2026-10-28)

Nothing in the repo was edited. The live bot (PID 35724, started 08:51:32) runs the current files
(`src/broker/rithmic_connector.py` last edited 10/26 18:13, `start_live_rithmic.py` 10/26 18:53).

## Root cause (plain words)

Two different "tracking tables" exist and they disagree after a release:

1. The **connector** (`src/broker/rithmic_connector.py`) keeps its own dict `self._orders[order_id]`.
   `place_order()` (~L1236-1295) and `_async_place_order()` (~L8470-8480, L8555-8566) write
   `entry_price`, `stop_loss`, `price_ref`, protective leg ids into it.
2. The **bot** (`start_live_rithmic.py`) keeps its own `self.positions[order_id]` (a `Position`,
   which has `entry_price`, `sl`, `tp`).

At 09:52:53 `RithmicConnector._cleanup_closed_orders()` (~L9609-9785) decided the trade was gone:
broker `list_positions` flat (the sim route always reports flat), no SL/TP legs visible for 45s/3 passes,
no exit fill, `_async_entry_has_working_legs()` -> False. It then ran
`remember_closed_entry()`, `_async_cancel_remaining_bracket_legs(force=True)` and
`self._orders.pop(oid)` (~L9759-9760) -> log "removed - SL/TP gone, no fill (manual/unknown) (path=cleanup_closed_orders)".

One second later the bot's own ghost-flat sync (`start_live_rithmic.py` ~L3200-3283) saw that the connector no
longer tracks the order, broker flat, `recover_exit_fill` found nothing, `entry_protection_live()` returned
None (unknown), so it logged "Keeping bot_030145a729 tracked ... not wiping" and kept its `Position`.
Result: the bot thinks it is long with entry 29787.50 / SL 29777.50, the connector has **no record at all**.

Every later stop move goes bot `_apply_stop_update()` (start_live ~L645-676) ->
`broker.update_stop_loss(ticket, symbol, side, size, new_sl, take_profit)` (connector ~L4571) ->
`_async_replace_protective_sl()` (~L4859). That function re-reads the connector's own record only:

```
_o = self._orders.get(order_id, {}) or {}
entry_px = float(_o.get("entry_price") or 0)     # ~L4880  -> 0.0 because the record was popped
...
elif entry_px <= 0:
    reason = "no entry price"                    # ~L4886-4887  <-- the log line
```
and the move is skipped ("SL move skipped for ... no entry price - keeping current stop", logged once per
distinct message via `_log_sl_skip_once`). `update_stop_loss` has the same blind spot: `entry_px`/`cur_sl` come
only from the record (~L4594-4595), so `_stop_on_profit_side()` and the ratchet check `_stop_is_looser()` both
see 0 and silently do nothing useful.

The bot never passes its own `pos.entry_price` into `update_stop_loss` (it DOES pass it into
`confirm_bracket_exit_fill(..., entry_price=pos.entry_price)` at start_live ~L2724, so the plumbing precedent exists).

Side effects of the missing record (same cause):
- `_cleanup_closed_orders`, protection verify, `ensure_protective_orders` all iterate `_orders`, so this trade
  is no longer verified/re-protected by the connector at all.
- `_async_cancel_take_profit()` still works without the record (it finds legs by the `<id>_tp` naming and the
  closed-entry registry), so the **past-TP path cancelled the TP and then could not trail the stop**. That is the
  dangerous ordering: target gone, trail dead.

## Can it happen on a LIVE (non-paused) account?

- A late fill by itself does NOT cause it. In `_async_place_order` an unconfirmed entry ("not confirmed filled
  within wait window", ~L8598-8602) keeps the record with `entry_price` = signal price (L8476; L1241), brackets
  are still attached, and while the legs are working `_async_entry_has_working_legs` is True so cleanup
  refuses to release ("refusing ghost-flat release", ~L9731-9737). Only the re-anchor-to-fill step is skipped
  (stop/target stay at signal-based levels).
- The record is lost only when the connector's cleanup decides "released": broker net == 0 AND its legs gone for
  45s/3 passes AND no exit fill. On a real Lucid live route with correct `list_positions` an open position gives
  net != 0 and the release is blocked (L9652-9662). So on live the exposure is **narrow but real**:
  - Lucid/Rithmic returning flat for an open position (the code comment at ~L9687 says even right after a fill),
  - stop AND target rejected/cancelled (bad price after a gap, manual cancel, connection loss) while the
    position is open, or
  - an entry that really never filled but the bot keeps a phantom `Position`.
- Real consequence when it does happen: the original fixed stop (if it rests) never trails, profit locks
  never fire, and after "past TP" the target is cancelled with no trailing stop -> a winner can fall back
  to the original stop. Also `cur_sl` reads 0 so the "never loosen" ratchet is blind. Bot's local `pos.sl`
  never updates either, so local stop checks keep using the original level.
- On the current paused sim account none of this matters financially (no real fills), which matches the log.

## Proposed minimal fix (behind env switch, DEFAULT OFF)

Switch: `RITHMIC_SL_ENTRY_FALLBACK` (unset / 0 = exactly today's behaviour; 1 = use fallbacks).
Fallback order for entry price: connector record `entry_price` -> record `price_ref` -> the caller's
`pos.entry_price` (bot's Position, which came from the order record at fill time). Same for current stop:
record `stop_loss` -> caller's `pos.sl`.

Why this is safe: it only fills in a missing number for the *existing* guards (in-profit check,
profit-side check, tighter-than-current check, valid-vs-market buffer) - all of those still run. Old stops are
still found via `_bracket_leg_ids()` (`<id>_sl` naming + closed-entry registry), and the new stop is still
placed first, old one cancelled after, so no duplicate-stop window is added. It does NOT recreate the
connector record with a `symbol`, so `_cleanup_closed_orders` will not start managing/cancelling it again.

### Patch A - `src/broker/rithmic_connector.py`

```diff
--- a/src/broker/rithmic_connector.py
+++ b/src/broker/rithmic_connector.py
@@ def update_stop_loss(
         new_sl: float,
         take_profit: Optional[float] = None,
         _from_modify: bool = False,
+        entry_price: Optional[float] = None,    # caller's own entry (bot Position); fallback only
+        current_sl: Optional[float] = None,     # caller's own current stop; fallback only
     ) -> bool:
@@
             is_long = str(side).upper() in ("BUY", "LONG")
             entry_px = float(order.get("entry_price") or 0)
             cur_sl = float(order.get("stop_loss") or 0)
+            _fb = os.getenv("RITHMIC_SL_ENTRY_FALLBACK", "0").strip().lower() in ("1", "true", "yes", "on")
+            if _fb:
+                if entry_px <= 0:
+                    entry_px = float(order.get("price_ref") or 0) or float(entry_price or 0)
+                if cur_sl <= 0:
+                    cur_sl = float(current_sl or 0)
@@
             ok = self._run_sync(
                 self._async_replace_protective_sl(
                     order_id=order_id,
                     symbol=symbol,
                     side=side,
                     qty=int(size),
                     new_sl=new_sl,
+                    entry_hint=entry_px if _fb else 0.0,
+                    cur_sl_hint=cur_sl if _fb else 0.0,
                 ),
@@ async def _async_replace_protective_sl(
         side: str,
         qty: int,
         new_sl: float,
+        entry_hint: float = 0.0,
+        cur_sl_hint: float = 0.0,
     ) -> bool:
@@
         with self._order_lock:
             _o = self._orders.get(order_id, {}) or {}
             entry_px = float(_o.get("entry_price") or 0)
             cur_sl = float(_o.get("stop_loss") or 0)
+        if entry_px <= 0 and entry_hint > 0:
+            entry_px = float(_o.get("price_ref") or 0) or float(entry_hint)
+            self._log_sl_skip_once(
+                f"{order_id}:entry_fallback",
+                f"SL move {order_id} {symbol}: connector record has no entry price - "
+                f"using fallback entry {entry_px:.2f} (RITHMIC_SL_ENTRY_FALLBACK)",
+            )
+        if cur_sl <= 0 and cur_sl_hint > 0:
+            cur_sl = float(cur_sl_hint)
         buf_ticks = int(os.getenv("SL_MOVE_MIN_BUFFER_TICKS", "4"))
```
(`_log_sl_skip_once` dedups per key; the `f"{order_id}:entry_fallback"` key keeps the new line from clobbering
the normal skip message. If preferred, use a plain `bot_logger.info` guarded by a set.)

### Patch B - `start_live_rithmic.py` (`_apply_stop_update`, ~L656-667)

```diff
         if hasattr(self.broker, 'update_stop_loss'):
             side = 'BUY' if pos.direction == 'long' else 'SELL'
+            _kw = {}
+            if os.getenv('RITHMIC_SL_ENTRY_FALLBACK', '0').strip().lower() in ('1', 'true', 'yes', 'on'):
+                _kw = dict(entry_price=float(pos.entry_price or 0), current_sl=float(pos.sl or 0))
             ok = bool(
                 self.broker.update_stop_loss(
                     ticket=pos.order_id,
                     symbol=pos.symbol,
                     side=side,
                     size=pos.size,
                     new_sl=new_sl,
                     take_profit=pos.tp,
+                    **_kw,
                 )
             )
```
(Only the Rithmic connector gets the extra kwargs, and only when the switch is on, so other broker classes are untouched.)

### Optional Patch C (separate switch, also default off) - don't cancel the TP unless a trail is possible

In `start_live_rithmic.py` `update_past_tp_trail` (~L735) the broker TP is cancelled BEFORE the stop move is tried.
With `RITHMIC_PAST_TP_REQUIRE_TRAIL=1`, skip the cancel (leave the TP working) when
`broker.get_order_info(order_id)` is None or has no `entry_price` and the fallback switch is off. This keeps the
original target as a safety net exactly in the broken-record case. Not needed if Patch A+B are on.

## How to verify BEFORE enabling anywhere (no live orders)

1. Unit test: build a connector with `_orders = {}` (record absent), call `_async_replace_protective_sl` with
   `entry_hint=29787.5, cur_sl_hint=29777.5`, a mocked `_live_market_price` of 29914.25, `new_sl=29812.5`:
   expect fallback log + proceeds to `_async_place_fresh_sl` (mock it). With switch off: expect "no entry price".
2. Negative tests: market below entry -> still "trade not in profit"; new_sl looser than hint cur_sl -> still
   skipped; no hints -> still "no entry price".
3. Replay on the paused sim account first (needs a flatten/restart to load new code - the running bot does not
   pick up edits). Do not enable on the funded account until a real fill shows the stop actually moving.

## Not changed / out of scope

- The root disagreement (connector pops the record while the bot keeps the `Position`) is left as is; making the
  bot re-register a record with the connector (or stopping the connector releasing an entry the bot still tracks)
  would be the fuller fix but touches order tracking and needs real-fill testing.
- Sim route reporting flat positions and rejecting SL/TP is a separate (account/route) issue.