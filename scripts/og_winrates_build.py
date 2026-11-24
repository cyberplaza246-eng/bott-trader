"""Build src/strategy/og_winrates.json from logs/trades.log (OG bot real trade history, MNQ only).
Read-only on the log. Entries are matched to exits FIFO by direction; exits with no bot ENTRY
(adopted/manual positions) go in their own bucket. Consecutive identical EXIT lines are counted once."""
import json, re, collections, datetime
LOG = 'logs/trades.log'; OUT = 'src/strategy/og_winrates.json'
ent = re.compile(r'^ENTRY (long|short) MNQ @ ([\d.]+).*?(?:reason=(\S+))?')
ree = re.compile(r'^REENTRY (long|short) MNQ')
exr = re.compile(r'^EXIT (\w+) (long|short) MNQ @ [\d.]+ PnL=\$(-?[\d.]+)')
openq = {'long': collections.deque(), 'short': collections.deque()}
stats = collections.defaultdict(list); prev = None; dup = 0
for line in open(LOG, encoding='utf-8', errors='ignore'):
    try: m = json.loads(line).get('message', '')
    except Exception: m = line.strip()
    if m.startswith('ENTRY ') and ' MNQ ' in m:
        d = m.split()[1]
        r = re.search(r'reason=(\S+)', m); r = r.group(1) if r else ''
        kind = 'addon' if r.startswith('addon') else 'sweep'
        openq[d].append(f'{kind}_{d}'); prev = None; continue
    if ree.match(m):
        d = m.split()[1]; openq[d].append(f'reentry_{d}'); prev = None; continue
    x = exr.match(m)
    if x:
        if m == prev: dup += 1; continue
        prev = m
        d = x.group(2); pnl = float(x.group(3))
        key = openq[d].popleft() if openq[d] else f'adopted_{d}'
        stats[key].append(pnl)
names = {'sweep_long': 'Sweep long (main setup)', 'sweep_short': 'Sweep short (main setup)',
         'reentry_long': 'Re-entry long', 'reentry_short': 'Re-entry short',
         'addon_long': 'Add-on long (ONL/6PM/E1)', 'addon_short': 'Add-on short',
         'adopted_long': 'Adopted/manual long', 'adopted_short': 'Adopted/manual short'}
out = {'source': LOG, 'built': datetime.datetime.now().isoformat(timespec='seconds'), 'dupe_exit_lines_skipped': dup,
       'note': 'trades.log has no timestamps; whole history, costs included as logged', 'setups': {}}
for k, v in sorted(stats.items()):
    w = sum(1 for p in v if p > 0)
    out['setups'][k] = {'name': names.get(k, k), 'n': len(v), 'win_pct': round(100*w/len(v), 1),
                        'avg_usd': round(sum(v)/len(v), 2), 'total_usd': round(sum(v), 2)}
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps(out, indent=1))
