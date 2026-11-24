
---
## Addendum 9 (2026-10-28): "capitalize on the midday selloff" (BACKTEST / ANALYSIS ONLY)
Question from Shawayne: news around 12-1 PM ET today changed the chart's trajectory (dip and selloff) - how do we capitalize? Scripts `x9_today.py`, `x9_today2.py` (today's bars), `x9_test.py` (backtest); outputs `x9_rules.csv`, `x9_events.json`, `x9.out`, `x9_nq_1m_today.csv` (Yahoo NQ=F 1m). Backup of this report: `REPORT.md.bak_pre_add9`. Live bot (PID 35724) not touched, no orders, nothing bought.

### Part 1: what actually happened today (Yahoo NQ=F 1m; MNQ tracks it within a few points)
- **No news-driven selloff at noon-1 PM.** 12:00-12:59 ET NQ ranged only 29,827-29,921 (open 29,849, close 29,885). The largest midday move was a **~109-point dip, 29,920 (12:54) to 29,811 (13:28)**, a normal-size wiggle, and no news item I found explains it. I cannot verify any cause for it and will not invent one.
- **What the news really was:** (a) Fed held at 3.75-4.00% with two hawkish dissents and Warsh's first press conference (2:00 / 2:30 PM): NQ whipsawed 29,786 to 30,009 between 14:16 and 14:43, but AP/Barron's say stocks "shrugged"; S&P +0.3%, Nasdaq +0.27% at the close. (b) Earlier AI news was bullish (SK Hynix record profit, Oracle +11% on a 2030 outlook). (c) **The real selloff came AFTER the 4 PM cash close**: NQ spiked to 30,178 at 16:01 then fell to 29,674 by 16:50 (**about -504 points, roughly $1,000 per MNQ**), as Alphabet fell about 9% after a revenue/op-income miss and negative free cash flow, with ServiceNow -15% and Fortinet -25%, while Microsoft rose about 2% (CNBC after-hours movers, IBD, Newsquawk). That is the move the bot's 4:40 PM log shows (29,740). So the "selloff" happened roughly 3 hours later than he remembers, and it was an earnings gap, not midday news.
- Sources: CNBC (cnbc.com/2026/10/28/stocks-making-the-biggest-moves-after-hours-googl-msft-cmg-now.html), AP News (Fed/stocks recap), Economic Times Fed live blog, IBD after-hours piece.
- Bot note: at 16:41 the bot printed a trend SELL signal and skipped it ("SHORT setup is against open trade 1 (LONG)") because the paper long from the morning was still open. That is a design rule (only same-direction adds), not a bug.

### Part 2: objective backtest of "midday selloff" rules (2-year MNQ 1m; 466 trading days; train = before 2026-01-01, 2026 window = from 2026-01-01)
**Rule grid, fixed before any result was seen:** at a 1-minute close between 11:30 and 14:00 ET, if price is X points below the highest high of the trailing W minutes, take the first such event per day. X in {80, 120, 180} pts, W in {60, 90} min (6 event definitions; 299/321, 209/244, 106/142 event days). Enter at the next 1m open with 1 tick slippage; exit at the open H minutes later (H in {30, 60, 120}, capped at 15:55) with 1 tick slippage; $4 round trip; 1 contract; no stop. Two directions: **buy-the-dip (fade)** and **continuation short**. **36 variants total; every one is reported (no dropped variants).**
**Significance:** each variant's average is compared with a placebo of the same number of random entries (any day, any minute 11:30-14:00, same side and hold, 4,000 draws); Bonferroni x36 for the family. Pass requires positive average in train AND 2026.

| Result across the 36 variants | count |
|---|---|
| Positive average after costs (all data) | 8 of 36 (28 lose) |
| Positive in BOTH train and 2026 | 3 of 36 |
| Raw placebo p < 0.05 | 0 of 36 (about 1.8 expected by chance) |
| Raw p < 0.10 | 1 of 36 (about 3.6 expected) |
| Bonferroni-adjusted p < 0.05 | 0 |

- **Continuation short (chase the selloff): loses in every threshold.** 17 of 18 variants lose money overall; the worst, X=180/W=60, H=60 is -$30/trade (-$3,179, 106 trades, both windows negative). Shorting after a midday drop is the wrong side on this data.
- **Buy-the-dip: nothing reliable.** Small thresholds (80, 120) average about -$8 to +$6/trade, essentially the cost of trading ($5 per trade). The large-drop buckets look better but are post-hoc: **X=180/W=60 fade, H=60: +$20/trade, +$2,119, 106 trades, PF 1.22, train +$15.1 (59), 2026 +$26.1 (47), raw p 0.08, Bonferroni p 1.0.** Its sibling H=120 is +$41.5/trade in train but -$4.5 in 2026, and X=180/W=90 H=120 is +$21.2 in train and -$32.5 in 2026, which are exactly the flips a noise result shows. The two others positive in both windows (X=120/W=60/H=120 +$6.5, X=180/W=60/H=30 +$6.4) are $6/trade, below the cost noise. **All three are post-hoc picks from a 36-variant grid: treat as noise, not an edge.**
- **Slippage stress was not run** because no variant passed the both-windows + significance screen.

### Part 3: did the 3R baseline already capture these days?
- Baseline average day P&L is $65 on X=80/W=60 event days vs $39 on other days (299 vs 167 days), $66 vs $43 for X=120/W=90, but **$41 vs $56 on X=180/W=60 days** (106 days; baseline is a little worse than usual on very big drop days). Mixed and not a systematic effect.
- Baseline trades entered within 4 hours after the trigger on event days (X=80/W=60): 133 trades, +$3,324, PF 1.52 (train +$2,876, 2026 +$448); X=180/W=60: 40 trades, +$1,520, PF 2.13 (train +$991, 2026 +$529). So **the baseline already trades through these days and is positive after the trigger**, with the usual 2026 softening (X=120/W=60: 2026 -$457). This is an overlap description, not a new rule.
- If the dip-buy fade were added on top it would trade on the same days as the baseline (which is long/short by its own signals) and add concurrency risk without proven edge.

### Part 4: what today's data says (anecdote, outside the 2-year data, one day)
Only the X=80 rules fired (13:18 ET; the 109-point dip). Buying it: H=30 +$3.5, H=60 -$49, H=120 +$38 per MNQ; shorting it: -$13.5, +$39, -$48. A coin flip, as the backtest says. None of the larger thresholds fired today. The biggest drop of the day (-504 pts) came after 16:00, outside both this test's window and the regular session.

### Verdict
- **Nothing to add.** Neither "buy the midday dip" nor "short the midday selloff" holds up on both windows after costs and a 36-variant discount. Shorting after a drop loses in nearly every setting. The only both-window-positive large-drop dip buys are post-hoc and not significant.
- Today's big move was an after-hours earnings gap (Alphabet miss), not midday news; the bot already trades the 16:00-18:00 session (baseline there is about flat: -$117 over 56 trades, Addendum 8) and a rule built on one earnings night would be a guess.
- Possible follow-up only if wanted: shadow-log midday drops >=180 pts (60 min) and the follow-through, then re-test in ~6 months (about 1 event per 4-5 days at that threshold, so the sample grows fast). No live change.
