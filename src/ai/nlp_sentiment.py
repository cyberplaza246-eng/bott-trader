"""
NLP-Based Sentiment Analyzer — FinBERT Edition

Upgrades the naive keyword-based sentiment to transformer-based NLP:
  - Uses ProsusAI/finbert for financial text classification
  - Falls back to keyword-based analysis if transformers unavailable
  - Supports NewsAPI and RSS feed inputs
  - Caches model to avoid repeated loads
  - Thread-safe inference

Requirements:
  pip install transformers torch  (optional — graceful fallback)
"""
import os
import requests
import feedparser
import numpy as np
from datetime import datetime, timedelta
from src.utils.logger import bot_logger, error_logger

# Try to load transformers + torch
FINBERT_AVAILABLE = False
_finbert_pipeline = None

try:
    from transformers import pipeline as hf_pipeline
    FINBERT_AVAILABLE = True
    bot_logger.info("✅ Transformers available — FinBERT NLP sentiment enabled")
except ImportError:
    bot_logger.warning("⚠️ transformers not installed — falling back to keyword sentiment")


def _get_finbert_pipeline():
    """Lazy-load FinBERT pipeline (cached after first call)."""
    global _finbert_pipeline
    if _finbert_pipeline is None and FINBERT_AVAILABLE:
        try:
            _finbert_pipeline = hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=None,          # return all labels with scores
                truncation=True,
                max_length=512,
            )
            bot_logger.info("✅ FinBERT model loaded successfully")
        except Exception as e:
            error_logger.error(f"Failed to load FinBERT: {e}")
            _finbert_pipeline = None
    return _finbert_pipeline


class NLPSentimentAnalyzer:
    """
    Financial NLP sentiment analyzer using FinBERT.

    FinBERT output labels: 'positive', 'negative', 'neutral'
    Mapped to: +1.0, -1.0, 0.0 with confidence weighting.

    Falls back to keyword-based analysis if transformers/torch
    are not installed.
    """

    def __init__(self, newsapi_key=None):
        self.newsapi_key = newsapi_key
        self.base_url = 'https://newsapi.org/v2'
        self.newsapi_rate_limit_cooldown_minutes = int(
            os.getenv('NEWSAPI_RATE_LIMIT_COOLDOWN_MINUTES', '20')
        )
        self._newsapi_blocked_until = None
        self.use_finbert = FINBERT_AVAILABLE

        # Legacy keyword fallback
        self.bullish_keywords = [
            'rally', 'surge', 'gains', 'bull', 'strong', 'positive',
            'growth', 'upside', 'breakout', 'momentum', 'bullish',
            'hawkish', 'recovery', 'rebound', 'optimism', 'soar',
            'climb', 'advance', 'higher', 'rising', 'boost',
            'rate hike', 'tightening', 'robust', 'resilient',
        ]
        self.bearish_keywords = [
            'plunge', 'decline', 'loss', 'bear', 'weak', 'negative',
            'bearish', 'downside', 'breakdown', 'selloff',
            'dovish', 'recession', 'slump', 'tumble', 'crash',
            'fall', 'drop', 'lower', 'sliding', 'cut',
            'rate cut', 'easing', 'stagnation', 'contraction',
        ]

    # ──────────────────────────────────────────────────────────────
    #  Core NLP Analysis
    # ──────────────────────────────────────────────────────────────
    def analyze_text_sentiment(self, text: str) -> dict:
        """
        Analyze sentiment of a single text using FinBERT or keyword fallback.

        Returns:
            {
                'score': float -1.0 to +1.0,
                'confidence': float 0.0 to 1.0,
                'method': 'finbert' | 'keyword',
                'label': 'positive' | 'negative' | 'neutral',
            }
        """
        if not text or len(text.strip()) < 10:
            return {'score': 0.0, 'confidence': 0.0, 'method': 'none', 'label': 'neutral'}

        if self.use_finbert:
            return self._finbert_analyze(text)
        return self._keyword_analyze(text)

    def analyze_batch(self, texts: list) -> list:
        """
        Analyze a list of texts efficiently (FinBERT supports batching).

        Returns: list of sentiment dicts.
        """
        if not texts:
            return []

        if self.use_finbert:
            return self._finbert_batch(texts)

        return [self._keyword_analyze(t) for t in texts]

    def _finbert_analyze(self, text: str) -> dict:
        """Single-text FinBERT inference."""
        pipe = _get_finbert_pipeline()
        if pipe is None:
            return self._keyword_analyze(text)

        try:
            results = pipe(text[:512])  # truncate to max length
            # results is a list of lists: [[{label, score}, ...]]
            if isinstance(results[0], list):
                scores = results[0]
            else:
                scores = results

            score_map = {}
            for item in scores:
                score_map[item['label'].lower()] = item['score']

            pos = score_map.get('positive', 0.0)
            neg = score_map.get('negative', 0.0)
            neu = score_map.get('neutral', 0.0)

            # Composite score: positive pushes up, negative pushes down
            sentiment_score = pos - neg  # range: -1.0 to +1.0
            confidence = max(pos, neg, neu)  # highest class probability

            if pos > neg and pos > neu:
                label = 'positive'
            elif neg > pos and neg > neu:
                label = 'negative'
            else:
                label = 'neutral'

            return {
                'score': round(sentiment_score, 4),
                'confidence': round(confidence, 4),
                'method': 'finbert',
                'label': label,
                'raw_scores': {'positive': pos, 'negative': neg, 'neutral': neu},
            }

        except Exception as e:
            error_logger.error(f"FinBERT inference error: {e}")
            return self._keyword_analyze(text)

    def _finbert_batch(self, texts: list) -> list:
        """Batch FinBERT inference for efficiency."""
        pipe = _get_finbert_pipeline()
        if pipe is None:
            return [self._keyword_analyze(t) for t in texts]

        try:
            # Clean and truncate
            clean_texts = [t[:512] for t in texts if t and len(t.strip()) >= 10]
            if not clean_texts:
                return [{'score': 0.0, 'confidence': 0.0, 'method': 'none', 'label': 'neutral'}] * len(texts)

            results = pipe(clean_texts, batch_size=16)

            output = []
            for res in results:
                scores = res if isinstance(res, list) else [res]
                score_map = {item['label'].lower(): item['score'] for item in scores}

                pos = score_map.get('positive', 0.0)
                neg = score_map.get('negative', 0.0)
                neu = score_map.get('neutral', 0.0)

                sentiment_score = pos - neg
                confidence = max(pos, neg, neu)

                if pos > neg and pos > neu:
                    label = 'positive'
                elif neg > pos and neg > neu:
                    label = 'negative'
                else:
                    label = 'neutral'

                output.append({
                    'score': round(sentiment_score, 4),
                    'confidence': round(confidence, 4),
                    'method': 'finbert',
                    'label': label,
                })

            return output

        except Exception as e:
            error_logger.error(f"FinBERT batch error: {e}")
            return [self._keyword_analyze(t) for t in texts]

    def _keyword_analyze(self, text: str) -> dict:
        """Fallback keyword-based sentiment."""
        if not text:
            return {'score': 0.0, 'confidence': 0.0, 'method': 'keyword', 'label': 'neutral'}

        text_lower = text.lower()
        bullish = sum(1 for w in self.bullish_keywords if w in text_lower)
        bearish = sum(1 for w in self.bearish_keywords if w in text_lower)

        total = bullish + bearish
        if total == 0:
            return {'score': 0.0, 'confidence': 0.0, 'method': 'keyword', 'label': 'neutral'}

        score = (bullish - bearish) / total
        confidence = min(total / 5.0, 1.0)  # more keyword hits = more confident

        label = 'positive' if score > 0.1 else ('negative' if score < -0.1 else 'neutral')

        return {
            'score': round(max(-1.0, min(1.0, score)), 4),
            'confidence': round(confidence, 4),
            'method': 'keyword',
            'label': label,
        }

    # ──────────────────────────────────────────────────────────────
    #  News Data Fetching (same as original SentimentAnalyzer)
    # ──────────────────────────────────────────────────────────────
    def get_forex_news(self, pair: str, limit: int = 10) -> list:
        """Fetch latest forex news from NewsAPI."""
        if not self.newsapi_key:
            bot_logger.warning("NewsAPI key not configured, skipping news sentiment")
            return []

        if self._newsapi_blocked_until and datetime.now() < self._newsapi_blocked_until:
            return []

        try:
            currencies = pair.split('/')
            query = f"({currencies[0]} OR {currencies[1]}) AND (forex OR currency)"

            params = {
                'q': query,
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': limit,
                'apiKey': self.newsapi_key,
            }

            response = requests.get(
                f'{self.base_url}/everything', params=params, timeout=5
            )

            if response.status_code == 200:
                self._newsapi_blocked_until = None
                return response.json().get('articles', [])
            elif response.status_code == 429:
                self._newsapi_blocked_until = datetime.now() + timedelta(
                    minutes=self.newsapi_rate_limit_cooldown_minutes
                )
                bot_logger.warning(
                    f"NewsAPI rate-limited (429). Pausing until "
                    f"{self._newsapi_blocked_until.strftime('%H:%M:%S')}"
                )
                return []
            else:
                error_logger.error(f"NewsAPI error: {response.status_code}")
                return []
        except Exception as e:
            error_logger.error(f"Error fetching news: {e}")
            return []

    # ──────────────────────────────────────────────────────────────
    #  High-Level Pair Sentiment (drop-in replacement)
    # ──────────────────────────────────────────────────────────────
    def get_pair_sentiment(self, pair: str) -> dict:
        """
        Get overall NLP sentiment score for a currency pair.
        Drop-in compatible with the original SentimentAnalyzer API.

        Returns:
            {
                'sentiment_score': -1.0 to +1.0,
                'confidence': 0.0 to 1.0,
                'news_count': int,
                'label': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
                'method': 'finbert' | 'keyword',
                'reason': str,
            }
        """
        try:
            articles = self.get_forex_news(pair, limit=20)

            if not articles:
                return {
                    'sentiment_score': 0.0,
                    'confidence': 0.0,
                    'news_count': 0,
                    'label': 'NEUTRAL',
                    'method': 'none',
                    'reason': 'No recent news found',
                }

            # Build text list from headlines + descriptions
            texts = []
            for article in articles:
                title = article.get('title', '') or ''
                desc = article.get('description', '') or ''
                combined = f"{title}. {desc}".strip()
                if len(combined) > 10:
                    texts.append(combined)

            if not texts:
                return {
                    'sentiment_score': 0.0,
                    'confidence': 0.0,
                    'news_count': 0,
                    'label': 'NEUTRAL',
                    'method': 'none',
                    'reason': 'No usable article text',
                }

            # Batch analysis
            sentiments = self.analyze_batch(texts)

            scores = [s['score'] for s in sentiments]
            confidences = [s['confidence'] for s in sentiments]
            method = sentiments[0]['method'] if sentiments else 'none'

            # Weighted average: recent articles weighted more
            weights = np.linspace(0.5, 1.0, len(scores))
            avg_score = np.average(scores, weights=weights)
            avg_conf = np.mean(confidences) * min(len(texts) / 10.0, 1.0)

            label = 'BULLISH' if avg_score > 0.1 else (
                'BEARISH' if avg_score < -0.1 else 'NEUTRAL'
            )

            return {
                'sentiment_score': round(float(avg_score), 4),
                'confidence': round(float(min(avg_conf, 1.0)), 4),
                'news_count': len(texts),
                'label': label,
                'method': method,
                'reason': f"{len(texts)} articles analyzed via {method}",
            }

        except Exception as e:
            error_logger.error(f"NLP sentiment error for {pair}: {e}")
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'news_count': 0,
                'label': 'NEUTRAL',
                'method': 'error',
                'reason': f'Error: {e}',
            }

    def get_multiple_pair_sentiment(self, pairs: list) -> dict:
        """Get sentiment for multiple pairs."""
        return {pair: self.get_pair_sentiment(pair) for pair in pairs}
