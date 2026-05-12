"""
FinBERT-based financial sentiment analysis.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from xml.etree import ElementTree

import requests

logger = logging.getLogger(__name__)

# Lazy-loaded pipeline
_sentiment_pipeline = None


def _get_pipeline():
    """Lazy-load the FinBERT pipeline on first use."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            from transformers import pipeline
            _sentiment_pipeline = pipeline(
                'sentiment-analysis',
                model='ProsusAI/finbert',
                tokenizer='ProsusAI/finbert',
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            raise
    return _sentiment_pipeline


class SentimentAnalyzer:
    """FinBERT-based financial sentiment analysis."""

    def __init__(self):
        self.db_manager = None
        self._finnhub_key = os.getenv('FINNHUB_API_KEY', '')

    def set_db_manager(self, db_manager):
        """Set the database manager for storing scores."""
        self.db_manager = db_manager

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze a single text. Returns label, score, confidence."""
        try:
            pipe = _get_pipeline()
            result = pipe(text[:512])[0]  # Truncate to model max

            label = result['label'].lower()
            score_raw = result['score']

            # Map to -1 to 1 scale
            if label == 'positive':
                score = score_raw
            elif label == 'negative':
                score = -score_raw
            else:
                score = 0.0

            return {
                'label': label,
                'score': score,
                'confidence': score_raw,
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'label': 'neutral', 'score': 0.0, 'confidence': 0.0}

    def analyze_symbol(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Fetch recent news headlines for symbol, analyze each, return aggregate.
        Stores individual scores in sentiment_scores table via db_manager.
        """
        headlines = self._fetch_news(symbol, days)

        if not headlines:
            return {
                'symbol': symbol,
                'overall_sentiment': 0.0,
                'overall_label': 'neutral',
                'confidence': 0.0,
                'trend': 'stable',
                'headline_count': 0,
                'recent_headlines': [],
            }

        analyzed = []
        for item in headlines:
            result = self.analyze_text(item['headline'])
            entry = {
                'headline': item['headline'],
                'label': result['label'],
                'score': result['score'],
                'confidence': result['confidence'],
                'date': item.get('date', str(datetime.utcnow().date())),
            }
            analyzed.append(entry)

            # Store in DB
            if self.db_manager:
                try:
                    self.db_manager.save_sentiment_score(
                        symbol=symbol,
                        date=datetime.utcnow().date(),
                        score=result['score'],
                        label=result['label'],
                        confidence=result['confidence'],
                        source=item.get('source', 'unknown'),
                        headline=item['headline'],
                    )
                except Exception as e:
                    logger.warning(f"Failed to save sentiment score: {e}")

        # Compute aggregate
        scores = [a['score'] for a in analyzed]
        confidences = [a['confidence'] for a in analyzed]

        # Weighted average by confidence
        total_conf = sum(confidences)
        if total_conf > 0:
            overall = sum(s * c for s, c in zip(scores, confidences)) / total_conf
        else:
            overall = 0.0

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Determine label
        if overall > 0.15:
            overall_label = 'positive'
        elif overall < -0.15:
            overall_label = 'negative'
        else:
            overall_label = 'neutral'

        # Trend detection (compare first half vs second half)
        if len(scores) >= 4:
            mid = len(scores) // 2
            first_half_avg = sum(scores[:mid]) / mid
            second_half_avg = sum(scores[mid:]) / (len(scores) - mid)
            diff = second_half_avg - first_half_avg
            if diff > 0.1:
                trend = 'improving'
            elif diff < -0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'

        return {
            'symbol': symbol,
            'overall_sentiment': round(overall, 4),
            'overall_label': overall_label,
            'confidence': round(avg_confidence, 4),
            'trend': trend,
            'headline_count': len(analyzed),
            'recent_headlines': analyzed[:10],
        }

    def _fetch_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """
        Fetch news headlines.
        1. Try Finnhub API
        2. Fallback: Yahoo Finance RSS feed
        """
        headlines = []

        # Try Finnhub
        if self._finnhub_key:
            headlines = self._fetch_finnhub_news(symbol, days)

        # Fallback to Yahoo RSS
        if not headlines:
            headlines = self._fetch_yahoo_rss(symbol)

        return headlines

    def _fetch_finnhub_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Fetch news from Finnhub API."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)

            url = 'https://finnhub.io/api/v1/company-news'
            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': self._finnhub_key,
            }
            resp = requests.get(url, params=params, timeout=10)

            if resp.status_code != 200:
                return []

            data = resp.json()
            headlines = []
            for item in data[:20]:  # Limit to 20 headlines
                headlines.append({
                    'headline': item.get('headline', ''),
                    'date': datetime.fromtimestamp(item.get('datetime', 0)).strftime('%Y-%m-%d'),
                    'source': 'finnhub',
                })

            return [h for h in headlines if h['headline']]
        except Exception as e:
            logger.warning(f"Finnhub news fetch failed: {e}")
            return []

    def _fetch_yahoo_rss(self, symbol: str) -> List[Dict]:
        """Fetch news from Yahoo Finance RSS feed."""
        try:
            url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US'
            resp = requests.get(url, timeout=10)

            if resp.status_code != 200:
                return []

            root = ElementTree.fromstring(resp.content)
            headlines = []

            for item in root.findall('.//item')[:15]:
                title = item.find('title')
                pub_date = item.find('pubDate')
                if title is not None and title.text:
                    headlines.append({
                        'headline': title.text,
                        'date': pub_date.text[:10] if pub_date is not None and pub_date.text else str(datetime.utcnow().date()),
                        'source': 'yahoo_rss',
                    })

            return headlines
        except Exception as e:
            logger.warning(f"Yahoo RSS fetch failed: {e}")
            return []
