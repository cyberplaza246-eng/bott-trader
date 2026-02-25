"""
Sentiment Analysis from News and Social Media
"""
import requests
import feedparser
import pandas as pd
from datetime import datetime, timedelta
from src.utils.logger import bot_logger, error_logger


class SentimentAnalyzer:
    """
    Analyze market sentiment from multiple sources:
    - News API
    - RSS feeds
    - Social media (Twitter keywords)
    """
    
    def __init__(self, newsapi_key=None):
        self.newsapi_key = newsapi_key
        self.base_url = 'https://newsapi.org/v2'
        
        # Sentiment keywords
        self.bullish_keywords = [
            'rally', 'surge', 'gains', 'bull', 'strong', 'positive',
            'growth', 'upside', 'breakout', 'momentum', 'bullish'
        ]
        
        self.bearish_keywords = [
            'plunge', 'decline', 'loss', 'bear', 'weak', 'negative',
            'bearish', 'downside', 'breakdown', 'decline', 'selloff'
        ]
    
    def get_forex_news(self, pair, limit=10):
        """
        Fetch latest forex news from NewsAPI
        
        Args:
            pair: Currency pair (e.g., 'EUR/USD')
            limit: Number of articles to fetch
        
        Returns:
            List of news articles
        """
        if not self.newsapi_key:
            bot_logger.warning("NewsAPI key not configured, skipping news sentiment")
            return []
        
        try:
            # Extract currency codes
            currencies = pair.split('/')
            query = f"({currencies[0]} OR {currencies[1]}) AND (forex OR currency)"
            
            params = {
                'q': query,
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': limit,
                'apiKey': self.newsapi_key
            }
            
            response = requests.get(f'{self.base_url}/everything', params=params, timeout=5)
            
            if response.status_code == 200:
                return response.json().get('articles', [])
            else:
                error_logger.error(f"NewsAPI error: {response.status_code}")
                return []
        
        except Exception as e:
            error_logger.error(f"Error fetching news: {str(e)}")
            return []
    
    def analyze_text_sentiment(self, text):
        """
        Simple sentiment analysis based on keyword matching
        
        Returns: -1.0 (bearish) to +1.0 (bullish)
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        bullish_count = sum(1 for word in self.bullish_keywords if word in text_lower)
        bearish_count = sum(1 for word in self.bearish_keywords if word in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        sentiment = (bullish_count - bearish_count) / total
        return max(-1.0, min(1.0, sentiment))
    
    def get_pair_sentiment(self, pair):
        """
        Get overall sentiment score for a currency pair
        
        Returns:
            {
                'sentiment_score': -1.0 to +1.0,
                'confidence': 0.0 to 1.0,
                'news_count': number of articles analyzed
            }
        """
        try:
            articles = self.get_forex_news(pair, limit=20)
            
            if not articles:
                return {
                    'sentiment_score': 0.0,
                    'confidence': 0.0,
                    'news_count': 0,
                    'reason': 'No recent news found'
                }
            
            sentiments = []
            for article in articles:
                # Analyze both headline and description
                headline_sentiment = self.analyze_text_sentiment(article.get('title', ''))
                description_sentiment = self.analyze_text_sentiment(article.get('description', ''))
                
                # Weight headline more heavily
                overall_sentiment = (headline_sentiment * 0.6) + (description_sentiment * 0.4)
                sentiments.append(overall_sentiment)
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiments) / len(sentiments)
            confidence = min(len(sentiments) / 10.0, 1.0)  # More articles = higher confidence
            
            sentiment_label = 'BULLISH' if avg_sentiment > 0.2 else ('BEARISH' if avg_sentiment < -0.2 else 'NEUTRAL')
            
            return {
                'sentiment_score': avg_sentiment,
                'confidence': confidence,
                'news_count': len(articles),
                'label': sentiment_label,
                'reason': f"{len(articles)} articles analyzed"
            }
        
        except Exception as e:
            error_logger.error(f"Error analyzing sentiment for {pair}: {str(e)}")
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'news_count': 0,
                'reason': f'Error: {str(e)}'
            }
    
    def get_multiple_pair_sentiment(self, pairs):
        """
        Get sentiment for multiple pairs at once
        
        Returns:
            Dictionary of pair -> sentiment scores
        """
        sentiments = {}
        for pair in pairs:
            sentiments[pair] = self.get_pair_sentiment(pair)
        
        return sentiments
