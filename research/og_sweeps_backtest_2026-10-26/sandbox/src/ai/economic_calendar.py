"""
Economic Event Calendar — High-Impact News Filter

Avoids trading during major economic releases that cause extreme volatility:
  - NFP (Non-Farm Payrolls) — first Friday of month
  - FOMC (Federal Reserve rate decisions) — scheduled 8x/year
  - CPI/PPI (Inflation data) — mid-month
  - ECB / BOE / BOJ rate decisions
  - GDP releases

Two data sources:
  1. Static recurring schedule (always available, no API needed)
  2. ForexFactory RSS feed (optional, real-time)

The bot will widen its avoid-window around high-impact events.
"""
import os
import re
from datetime import datetime, timedelta, timezone
from src.utils.logger import bot_logger


# ── Tiered buffer minutes before/after events ──────────────────
# High impact (FOMC, NFP, CPI, ECB, BOE, BOJ): wider buffer
PRE_EVENT_BUFFER_HIGH = int(os.getenv('EVENT_PRE_BUFFER_HIGH_MINUTES', '30'))
POST_EVENT_BUFFER_HIGH = int(os.getenv('EVENT_POST_BUFFER_HIGH_MINUTES', '30'))
# Medium impact (PPI, Retail Sales): tighter buffer
PRE_EVENT_BUFFER_MEDIUM = int(os.getenv('EVENT_PRE_BUFFER_MEDIUM_MINUTES', '15'))
POST_EVENT_BUFFER_MEDIUM = int(os.getenv('EVENT_POST_BUFFER_MEDIUM_MINUTES', '15'))


# ── Static recurring schedule (UTC times) ───────────────────────
# Format: (name, affected_currencies, day_rule, hour, minute)
# day_rule: 'first_friday' | 'weekday:X' (0=Mon) | 'dates:D1,D2,...'
#
# This covers the most impactful releases. The bot will block
# trading the affected currencies during the buffer window.
RECURRING_EVENTS = [
    # US events (affect USD pairs)
    {
        'name': 'NFP (Non-Farm Payrolls)',
        'currencies': ['USD'],
        'rule': 'first_friday',
        'hour': 13, 'minute': 30,
        'impact': 'high',
    },
    {
        'name': 'US CPI',
        'currencies': ['USD'],
        'rule': 'monthly_around:13',  # ~13th of month
        'hour': 13, 'minute': 30,
        'impact': 'high',
    },
    {
        'name': 'FOMC Rate Decision',
        'currencies': ['USD'],
        'rule': 'fomc',  # 8 scheduled meetings/year
        'hour': 19, 'minute': 0,
        'impact': 'high',
    },
    {
        'name': 'US PPI',
        'currencies': ['USD'],
        'rule': 'monthly_around:14',
        'hour': 13, 'minute': 30,
        'impact': 'medium',
    },
    {
        'name': 'US Retail Sales',
        'currencies': ['USD'],
        'rule': 'monthly_around:15',
        'hour': 13, 'minute': 30,
        'impact': 'medium',
    },
    # ECB (affect EUR pairs)
    {
        'name': 'ECB Rate Decision',
        'currencies': ['EUR'],
        'rule': 'ecb',
        'hour': 13, 'minute': 15,
        'impact': 'high',
    },
    # BOE (affect GBP pairs)
    {
        'name': 'BOE Rate Decision',
        'currencies': ['GBP'],
        'rule': 'boe',
        'hour': 12, 'minute': 0,
        'impact': 'high',
    },
    # BOJ (affect JPY pairs)
    {
        'name': 'BOJ Rate Decision',
        'currencies': ['JPY'],
        'rule': 'boj',
        'hour': 3, 'minute': 0,
        'impact': 'high',
    },
    # UK CPI
    {
        'name': 'UK CPI',
        'currencies': ['GBP'],
        'rule': 'monthly_around:15',
        'hour': 7, 'minute': 0,
        'impact': 'high',
    },
]

# FOMC meeting dates for 2025-2027 (add more as published)
# Source: Federal Reserve schedule
FOMC_DATES = [
    # 2025
    '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18',
    '2025-07-30', '2025-09-17', '2025-11-05', '2025-12-17',
    # 2026
    '2026-01-28', '2026-03-18', '2026-05-06', '2026-06-17',
    '2026-07-29', '2026-09-16', '2026-11-04', '2026-12-16',
    # 2027 (estimated — same pattern)
    '2027-01-27', '2027-03-17', '2027-05-05', '2027-06-16',
    '2027-07-28', '2027-09-15', '2027-11-03', '2027-12-15',
]

# ECB meeting dates (approximate — 6-week cycle, Thursdays)
ECB_DATES = [
    '2025-01-30', '2025-03-06', '2025-04-17', '2025-06-05',
    '2025-07-24', '2025-09-11', '2025-10-30', '2025-12-18',
    '2026-01-22', '2026-03-05', '2026-04-16', '2026-06-04',
    '2026-07-16', '2026-09-10', '2026-10-29', '2026-12-10',
]

# BOE meeting dates (approximate)
BOE_DATES = [
    '2025-02-06', '2025-03-20', '2025-05-08', '2025-06-19',
    '2025-08-07', '2025-09-18', '2025-11-06', '2025-12-18',
    '2026-02-05', '2026-03-19', '2026-05-07', '2026-06-18',
    '2026-08-06', '2026-09-17', '2026-11-05', '2026-12-17',
]

# BOJ meeting dates (approximate)
BOJ_DATES = [
    '2025-01-24', '2025-03-14', '2025-05-01', '2025-06-13',
    '2025-07-31', '2025-09-19', '2025-10-31', '2025-12-19',
    '2026-01-23', '2026-03-13', '2026-04-28', '2026-06-12',
    '2026-07-17', '2026-09-18', '2026-10-30', '2026-12-18',
]


class EconomicCalendar:
    """
    Check if a high-impact economic event is near, and block trading
    on affected pairs during the danger window.
    """

    def __init__(self):
        self.pre_buffer_high = timedelta(minutes=PRE_EVENT_BUFFER_HIGH)
        self.post_buffer_high = timedelta(minutes=POST_EVENT_BUFFER_HIGH)
        self.pre_buffer_medium = timedelta(minutes=PRE_EVENT_BUFFER_MEDIUM)
        self.post_buffer_medium = timedelta(minutes=POST_EVENT_BUFFER_MEDIUM)
        self._event_cache = {}
        self._cache_date = None

    def is_event_blocked(self, pair: str, now: datetime = None) -> tuple:
        """
        Check if trading this pair should be blocked due to an upcoming
        or recent high-impact event.

        Args:
            pair: e.g. 'EUR/USD'
            now:  current UTC datetime (auto-detected if None)

        Returns:
            (blocked: bool, event_name: str or None)
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Extract currencies from pair
        currencies = self._extract_currencies(pair)

        # Get today's events
        events = self._get_events_for_date(now.date())

        for event in events:
            # Check if this event affects our pair's currencies
            if not any(c in event['currencies'] for c in currencies):
                continue

            # Build event time
            event_time = datetime.combine(
                now.date(),
                datetime.min.time().replace(hour=event['hour'], minute=event['minute']),
            ).replace(tzinfo=timezone.utc)

            # Tiered buffer based on impact level
            impact = event.get('impact', 'high')
            if impact == 'high':
                pre_buf = self.pre_buffer_high
                post_buf = self.post_buffer_high
            else:  # medium or low
                pre_buf = self.pre_buffer_medium
                post_buf = self.post_buffer_medium

            # Check if we're in the danger window
            window_start = event_time - pre_buf
            window_end = event_time + post_buf

            if window_start <= now <= window_end:
                return True, event['name']

        return False, None

    def get_upcoming_events(self, hours_ahead: int = 4, now: datetime = None) -> list:
        """Get events happening in the next N hours."""
        if now is None:
            now = datetime.now(timezone.utc)

        upcoming = []
        for offset in range(2):  # Check today and tomorrow
            check_date = (now + timedelta(days=offset)).date()
            events = self._get_events_for_date(check_date)
            for event in events:
                event_time = datetime.combine(
                    check_date,
                    datetime.min.time().replace(hour=event['hour'], minute=event['minute']),
                ).replace(tzinfo=timezone.utc)

                delta = (event_time - now).total_seconds() / 3600
                if 0 < delta <= hours_ahead:
                    upcoming.append({
                        'name': event['name'],
                        'time': event_time.strftime('%H:%M UTC'),
                        'currencies': event['currencies'],
                        'impact': event['impact'],
                        'hours_away': round(delta, 1),
                    })

        return upcoming

    def _get_events_for_date(self, date) -> list:
        """Get all scheduled events for a specific date."""
        cache_key = str(date)
        if cache_key == self._cache_date and cache_key in self._event_cache:
            return self._event_cache[cache_key]

        events = []
        for event_def in RECURRING_EVENTS:
            if self._matches_rule(event_def['rule'], date):
                events.append(event_def)

        self._event_cache[cache_key] = events
        self._cache_date = cache_key
        return events

    @staticmethod
    def _matches_rule(rule: str, date) -> bool:
        """Check if a date matches an event's scheduling rule."""
        if rule == 'first_friday':
            # First Friday of the month
            if date.weekday() == 4 and date.day <= 7:
                return True
            return False

        if rule.startswith('monthly_around:'):
            # Match if date.day is within ±1 of target day AND it's a weekday
            target_day = int(rule.split(':')[1])
            if abs(date.day - target_day) <= 1 and date.weekday() < 5:
                return True
            return False

        if rule == 'fomc':
            return date.isoformat() in FOMC_DATES

        if rule == 'ecb':
            return date.isoformat() in ECB_DATES

        if rule == 'boe':
            return date.isoformat() in BOE_DATES

        if rule == 'boj':
            return date.isoformat() in BOJ_DATES

        return False

    @staticmethod
    def _extract_currencies(pair: str) -> list:
        """Extract currency codes from a pair string."""
        clean = pair.replace('/', '')
        if len(clean) == 6:
            return [clean[:3].upper(), clean[3:].upper()]
        return [pair.upper()]
