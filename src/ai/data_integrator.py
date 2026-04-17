"""
Ultimate Data Integration Module
Combines all data sources: Alpha Vantage, Polymarket, News, ETF flows, etc.
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional, Tuple
import logging

class UltimateDataIntegrator:
    """Integrates data from all sources for the ultimate trading bot"""

    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
        self.session = requests.Session()
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

        # API endpoints
        self.alpha_base = 'https://www.alphavantage.co/query'
        self.polymarket_gamma = 'https://gamma-api.polymarket.com'
        self.polymarket_clob = 'https://clob.polymarket.com'

        self.logger = logging.getLogger('DataIntegrator')

    def get_comprehensive_data(self, symbol: str) -> Dict:
        """Get all available data for a symbol"""

        # Check cache first
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M')}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        data = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'price_data': self._get_price_data(symbol),
            'technical_indicators': self._get_technical_indicators(symbol),
            'news_sentiment': self._get_news_sentiment(symbol),
            'social_sentiment': self._get_social_sentiment(symbol),
            'options_data': self._get_options_data(symbol),
            'institutional_data': self._get_institutional_data(symbol),
            'prediction_markets': self._get_prediction_market_data(),
            'macro_economic': self._get_macro_economic_data(),
            'etf_flows': self._get_etf_flows(),
            'order_flow': self._get_order_flow_data(symbol),
            'market_microstructure': self._get_market_microstructure(symbol)
        }

        # Cache the result
        self.cache[cache_key] = data

        return data

    def _get_price_data(self, symbol: str) -> pd.DataFrame:
        """Get OHLCV data from Alpha Vantage"""
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': '1min',
            'outputsize': 'full',
            'apikey': self.alpha_vantage_key
        }

        try:
            response = self.session.get(self.alpha_base, params=params, timeout=10)
            data = response.json()

            if 'Time Series (1min)' in data:
                df = pd.DataFrame.from_dict(data['Time Series (1min)'], orient='index')
                df = df.astype(float)
                df.index = pd.to_datetime(df.index)
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df = df.sort_index()

                # Add additional calculations
                df['Returns'] = df['Close'].pct_change()
                df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
                df['Volatility'] = df['Returns'].rolling(20).std()
                df['Volume_MA'] = df['Volume'].rolling(20).mean()

                return df

        except Exception as e:
            self.logger.error(f"Error fetching price data for {symbol}: {e}")

        return pd.DataFrame()

    def _get_technical_indicators(self, symbol: str) -> Dict:
        """Get comprehensive technical indicators"""
        indicators = {}

        # RSI
        indicators.update(self._get_rsi(symbol))

        # MACD
        indicators.update(self._get_macd(symbol))

        # Bollinger Bands
        indicators.update(self._get_bollinger_bands(symbol))

        # Stochastic Oscillator
        indicators.update(self._get_stochastic(symbol))

        # Williams %R
        indicators.update(self._get_williams_r(symbol))

        # CCI (Commodity Channel Index)
        indicators.update(self._get_cci(symbol))

        # ADX (Average Directional Index)
        indicators.update(self._get_adx(symbol))

        # Ichimoku Cloud
        indicators.update(self._get_ichimoku(symbol))

        return indicators

    def _get_rsi(self, symbol: str) -> Dict:
        """Get RSI indicator"""
        params = {
            'function': 'RSI',
            'symbol': symbol,
            'interval': '5min',
            'time_period': 14,
            'series_type': 'close',
            'apikey': self.alpha_vantage_key
        }

        try:
            response = self.session.get(self.alpha_base, params=params)
            data = response.json()

            if 'Technical Analysis: RSI' in data:
                latest_rsi = list(data['Technical Analysis: RSI'].values())[0]
                return {'rsi': float(latest_rsi['RSI'])}
        except:
            pass

        return {'rsi': 50.0}

    def _get_macd(self, symbol: str) -> Dict:
        """Get MACD indicator"""
        params = {
            'function': 'MACD',
            'symbol': symbol,
            'interval': '5min',
            'series_type': 'close',
            'apikey': self.alpha_vantage_key
        }

        try:
            response = self.session.get(self.alpha_base, params=params)
            data = response.json()

            if 'Technical Analysis: MACD' in data:
                latest_macd = list(data['Technical Analysis: MACD'].values())[0]
                return {
                    'macd': float(latest_macd['MACD']),
                    'macd_signal': float(latest_macd['MACD_Signal']),
                    'macd_hist': float(latest_macd['MACD_Hist'])
                }
        except:
            pass

        return {'macd': 0.0, 'macd_signal': 0.0, 'macd_hist': 0.0}

    def _get_bollinger_bands(self, symbol: str) -> Dict:
        """Get Bollinger Bands"""
        params = {
            'function': 'BBANDS',
            'symbol': symbol,
            'interval': '5min',
            'time_period': 20,
            'series_type': 'close',
            'apikey': self.alpha_vantage_key
        }

        try:
            response = self.session.get(self.alpha_base, params=params)
            data = response.json()

            if 'Technical Analysis: BBANDS' in data:
                latest_bb = list(data['Technical Analysis: BBANDS'].values())[0]
                return {
                    'bb_upper': float(latest_bb['Real Upper Band']),
                    'bb_middle': float(latest_bb['Real Middle Band']),
                    'bb_lower': float(latest_bb['Real Lower Band'])
                }
        except:
            pass

        return {'bb_upper': 0.0, 'bb_middle': 0.0, 'bb_lower': 0.0}

    def _get_stochastic(self, symbol: str) -> Dict:
        """Get Stochastic Oscillator"""
        params = {
            'function': 'STOCH',
            'symbol': symbol,
            'interval': '5min',
            'apikey': self.alpha_vantage_key
        }

        try:
            response = self.session.get(self.alpha_base, params=params)
            data = response.json()

            if 'Technical Analysis: STOCH' in data:
                latest_stoch = list(data['Technical Analysis: STOCH'].values())[0]
                return {
                    'stoch_k': float(latest_stoch['SlowK']),
                    'stoch_d': float(latest_stoch['SlowD'])
                }
        except:
            pass

        return {'stoch_k': 50.0, 'stoch_d': 50.0}

    def _get_williams_r(self, symbol: str) -> Dict:
        """Get Williams %R"""
        params = {
            'function': 'WILLR',
            'symbol': symbol,
            'interval': '5min',
            'time_period': 14,
            'apikey': self.alpha_vantage_key
        }

        try:
            response = self.session.get(self.alpha_base, params=params)
            data = response.json()

            if 'Technical Analysis: WILLR' in data:
                latest_willr = list(data['Technical Analysis: WILLR'].values())[0]
                return {'williams_r': float(latest_willr['WILLR'])}
        except:
            pass

        return {'williams_r': -50.0}

    def _get_cci(self, symbol: str) -> Dict:
        """Get Commodity Channel Index"""
        params = {
            'function': 'CCI',
            'symbol': symbol,
            'interval': '5min',
            'time_period': 20,
            'apikey': self.alpha_vantage_key
        }

        try:
            response = self.session.get(self.alpha_base, params=params)
            data = response.json()

            if 'Technical Analysis: CCI' in data:
                latest_cci = list(data['Technical Analysis: CCI'].values())[0]
                return {'cci': float(latest_cci['CCI'])}
        except:
            pass

        return {'cci': 0.0}

    def _get_adx(self, symbol: str) -> Dict:
        """Get Average Directional Index"""
        params = {
            'function': 'ADX',
            'symbol': symbol,
            'interval': '5min',
            'time_period': 14,
            'apikey': self.alpha_vantage_key
        }

        try:
            response = self.session.get(self.alpha_base, params=params)
            data = response.json()

            if 'Technical Analysis: ADX' in data:
                latest_adx = list(data['Technical Analysis: ADX'].values())[0]
                return {'adx': float(latest_adx['ADX'])}
        except:
            pass

        return {'adx': 25.0}

    def _get_ichimoku(self, symbol: str) -> Dict:
        """Calculate Ichimoku Cloud components"""
        # This would require raw price data - simplified implementation
        return {
            'tenkan_sen': 0.0,
            'kijun_sen': 0.0,
            'senkou_span_a': 0.0,
            'senkou_span_b': 0.0,
            'chikou_span': 0.0
        }

    def _get_news_sentiment(self, symbol: str) -> Dict:
        """Get news sentiment from Alpha Vantage"""
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'topics': 'financial_markets,earnings,mergers_and_acquisitions,financial_results',
            'apikey': self.alpha_vantage_key,
            'limit': 50
        }

        try:
            response = self.session.get(self.alpha_base, params=params)
            data = response.json()

            if 'feed' in data:
                sentiments = []
                for article in data['feed']:
                    if 'overall_sentiment_score' in article:
                        sentiments.append(float(article['overall_sentiment_score']))

                if sentiments:
                    return {
                        'news_sentiment_avg': np.mean(sentiments),
                        'news_sentiment_std': np.std(sentiments),
                        'news_count': len(sentiments),
                        'bullish_news': sum(1 for s in sentiments if s > 0.1),
                        'bearish_news': sum(1 for s in sentiments if s < -0.1)
                    }
        except Exception as e:
            self.logger.error(f"Error fetching news sentiment: {e}")

        return {
            'news_sentiment_avg': 0.0,
            'news_sentiment_std': 0.0,
            'news_count': 0,
            'bullish_news': 0,
            'bearish_news': 0
        }

    def _get_social_sentiment(self, symbol: str) -> Dict:
        """Get social media sentiment (Twitter, Reddit, etc.)"""
        # Placeholder - would integrate with social media APIs
        return {
            'twitter_sentiment': 0.0,
            'reddit_sentiment': 0.0,
            'combined_social_sentiment': 0.0
        }

    def _get_options_data(self, symbol: str) -> Dict:
        """Get options flow and implied volatility data"""
        # Placeholder - would integrate with options data providers
        return {
            'call_put_ratio': 1.0,
            'implied_volatility': 0.2,
            'options_volume': 0,
            'gamma_exposure': 0.0
        }

    def _get_institutional_data(self, symbol: str) -> Dict:
        """Get institutional holdings and flows"""
        # Placeholder - would integrate with institutional data providers
        return {
            'institutional_ownership': 0.0,
            'insider_trading': 0.0,
            'short_interest': 0.0
        }

    def _get_prediction_market_data(self) -> Dict:
        """Get prediction market data from Polymarket"""
        try:
            # Get major market predictions
            markets = [
                'will-the-sp-500-be-above-5000-on-december-31',
                'will-btc-be-above-100k-on-december-31',
                'will-the-fed-cut-rates-in-december'
            ]

            predictions = {}
            for market_slug in markets:
                response = self.session.get(f"{self.polymarket_gamma}/markets?slug={market_slug}")
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        market = data[0]
                        predictions[market_slug] = {
                            'question': market.get('question', ''),
                            'outcomes': market.get('outcomes', []),
                            'clobTokenIds': market.get('clobTokenIds', [])
                        }

            return predictions

        except Exception as e:
            self.logger.error(f"Error fetching prediction market data: {e}")

        return {}

    def _get_macro_economic_data(self) -> Dict:
        """Get macroeconomic indicators"""
        macro_data = {}

        # Federal Funds Rate
        macro_data['fed_funds_rate'] = self._get_fed_funds_rate()

        # Unemployment Rate
        macro_data['unemployment_rate'] = self._get_unemployment_rate()

        # CPI
        macro_data['cpi'] = self._get_cpi()

        # GDP Growth
        macro_data['gdp_growth'] = self._get_gdp_growth()

        return macro_data

    def _get_fed_funds_rate(self) -> float:
        """Get current Federal Funds Rate"""
        params = {
            'function': 'FEDERAL_FUNDS_RATE',
            'interval': 'monthly',
            'apikey': self.alpha_vantage_key
        }

        try:
            response = self.session.get(self.alpha_base, params=params)
            data = response.json()

            if 'data' in data:
                latest_rate = data['data'][0]
                return float(latest_rate['value'])
        except:
            pass

        return 5.25  # Current rate as fallback

    def _get_unemployment_rate(self) -> float:
        """Get unemployment rate"""
        params = {
            'function': 'UNEMPLOYMENT',
            'apikey': self.alpha_vantage_key
        }

        try:
            response = self.session.get(self.alpha_base, params=params)
            data = response.json()

            if 'data' in data:
                latest = data['data'][0]
                return float(latest['value'])
        except:
            pass

        return 4.1  # Current rate as fallback

    def _get_cpi(self) -> float:
        """Get CPI data"""
        params = {
            'function': 'CPI',
            'interval': 'monthly',
            'apikey': self.alpha_vantage_key
        }

        try:
            response = self.session.get(self.alpha_base, params=params)
            data = response.json()

            if 'data' in data:
                latest = data['data'][0]
                return float(latest['value'])
        except:
            pass

        return 307.671  # Current CPI as fallback

    def _get_gdp_growth(self) -> float:
        """Get GDP growth rate"""
        params = {
            'function': 'REAL_GDP',
            'interval': 'quarterly',
            'apikey': self.alpha_vantage_key
        }

        try:
            response = self.session.get(self.alpha_base, params=params)
            data = response.json()

            if 'data' in data and len(data['data']) >= 2:
                current = float(data['data'][0]['value'])
                previous = float(data['data'][1]['value'])
                return ((current - previous) / previous) * 100
        except:
            pass

        return 3.3  # Current GDP growth as fallback

    def _get_etf_flows(self) -> Dict:
        """Get ETF flow data"""
        # Placeholder - would integrate with ETF flow data providers
        return {
            'spy_flow': 0.0,
            'qqq_flow': 0.0,
            'iwm_flow': 0.0,
            'total_equity_flow': 0.0
        }

    def _get_order_flow_data(self, symbol: str) -> Dict:
        """Get order flow and market microstructure data"""
        # Placeholder - would integrate with order book data
        return {
            'order_imbalance': 0.0,
            'bid_ask_spread': 0.0,
            'market_depth': 0.0,
            'realized_volatility': 0.0
        }

    def _get_market_microstructure(self, symbol: str) -> Dict:
        """Get market microstructure indicators"""
        return {
            'vwap': 0.0,
            'price_impact': 0.0,
            'liquidity_ratio': 0.0,
            'market_efficiency': 0.0
        }

    def get_multi_asset_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get comprehensive data for multiple assets"""
        multi_data = {}
        for symbol in symbols:
            multi_data[symbol] = self.get_comprehensive_data(symbol)
            time.sleep(1)  # Rate limiting

        return multi_data

    def get_market_regime(self) -> str:
        """Determine current market regime"""
        # Analyze various indicators to determine regime
        macro_data = self._get_macro_economic_data()

        # Simple regime detection
        if macro_data['fed_funds_rate'] > 5.0:
            return 'high_rate_environment'
        elif macro_data['unemployment_rate'] < 4.0:
            return 'strong_economy'
        elif macro_data['gdp_growth'] > 4.0:
            return 'growth_acceleration'
        else:
            return 'neutral'