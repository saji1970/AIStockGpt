"""
Stock Market Data Pipeline for AI Stock GPT
Fetches market news and analysis from Yahoo Finance via RapidAPI.
"""

import os
import json
import requests
from datetime import datetime
from typing import Dict, List, Any


class StockMarketDataPipeline:
    """Fetches and processes market news from Yahoo Finance RapidAPI."""

    RAPIDAPI_HOST = "yahoo-finance166.p.rapidapi.com"
    NEWS_URL = "https://yahoo-finance166.p.rapidapi.com/api/news/list"

    def __init__(self, api_key: str = None):
        """Initialize the data pipeline with RapidAPI key.

        Args:
            api_key: RapidAPI key. Falls back to RAPIDAPI_KEY env var.
        """
        self.api_key = api_key or os.getenv("RAPIDAPI_KEY", "")
        if not self.api_key:
            raise ValueError("RAPIDAPI_KEY must be provided or set as environment variable")

        self.headers = {
            "Content-Type": "application/json",
            "x-rapidapi-host": self.RAPIDAPI_HOST,
            "x-rapidapi-key": self.api_key,
        }

    def gather_market_news(self, snippet_count: int = 500, region: str = "US") -> Dict[str, Any]:
        """Fetch market news from Yahoo Finance RapidAPI.

        Args:
            snippet_count: Number of news snippets to retrieve.
            region: Market region (default US).

        Returns:
            Dictionary containing news data and metadata.
        """
        try:
            params = {
                "snippetCount": snippet_count,
                "region": region,
            }

            response = requests.get(
                self.NEWS_URL,
                headers=self.headers,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            # Parse the news items from the API response
            news_items = self._parse_news_response(data)

            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "region": region,
                "total_items": len(news_items),
                "news_data": news_items,
                "raw_response": data,
            }

        except requests.exceptions.Timeout:
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": "Request timed out",
            }
        except requests.exceptions.HTTPError as e:
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": f"HTTP error: {e.response.status_code}",
            }
        except Exception as e:
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

    def _parse_news_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse the Yahoo Finance API response into structured news items."""
        news_items = []

        # Yahoo Finance API typically returns data in a stream or items array
        items = []
        if isinstance(data, dict):
            # Try common response structures
            if "data" in data:
                stream = data["data"]
                if isinstance(stream, dict) and "main" in stream:
                    main = stream["main"]
                    if isinstance(main, dict) and "stream" in main:
                        items = main["stream"]
                elif isinstance(stream, list):
                    items = stream
            elif "items" in data:
                items = data["items"]
            elif "news" in data:
                items = data["news"]

        for item in items:
            if not isinstance(item, dict):
                continue

            content = item.get("content", item)
            if not isinstance(content, dict):
                continue

            news_entry = {
                "title": content.get("title", ""),
                "summary": content.get("summary", content.get("description", "")),
                "url": content.get("clickThroughUrl", {}).get("url", "") if isinstance(content.get("clickThroughUrl"), dict) else content.get("url", ""),
                "publisher": content.get("provider", {}).get("displayName", "") if isinstance(content.get("provider"), dict) else content.get("publisher", ""),
                "published_at": content.get("pubDate", content.get("published_at", "")),
                "tickers": self._extract_tickers(content),
                "category": self._categorize_news(content),
            }

            if news_entry["title"]:
                news_items.append(news_entry)

        return news_items

    def _extract_tickers(self, content: Dict[str, Any]) -> List[str]:
        """Extract stock tickers mentioned in a news item."""
        tickers = []
        # Check for finance-specific ticker data
        if "finance" in content:
            finance = content["finance"]
            if isinstance(finance, dict) and "stockTickers" in finance:
                for ticker in finance["stockTickers"]:
                    if isinstance(ticker, dict) and "symbol" in ticker:
                        tickers.append(ticker["symbol"])

        # Also check for ticker array directly
        if "tickers" in content and isinstance(content["tickers"], list):
            for t in content["tickers"]:
                if isinstance(t, str):
                    tickers.append(t)
                elif isinstance(t, dict) and "symbol" in t:
                    tickers.append(t["symbol"])

        return tickers

    def _categorize_news(self, content: Dict[str, Any]) -> str:
        """Categorize news based on content."""
        title = (content.get("title", "") + " " + content.get("summary", "")).lower()

        categories = {
            "earnings": ["earnings", "revenue", "quarterly", "profit", "loss", "eps"],
            "economic": ["gdp", "inflation", "fed", "interest rate", "employment", "jobs", "cpi"],
            "geopolitical": ["trade war", "tariff", "sanctions", "geopolitical", "political"],
            "sector_tech": ["tech", "ai", "semiconductor", "software", "apple", "google", "microsoft"],
            "sector_energy": ["oil", "energy", "gas", "opec", "renewable", "solar"],
            "sector_finance": ["bank", "financial", "credit", "lending", "mortgage"],
            "sector_healthcare": ["pharma", "biotech", "healthcare", "fda", "drug"],
            "merger_acquisition": ["merger", "acquisition", "takeover", "buyout", "deal"],
            "ipo": ["ipo", "public offering", "listing"],
            "crypto": ["bitcoin", "crypto", "blockchain", "ethereum"],
        }

        for category, keywords in categories.items():
            if any(kw in title for kw in keywords):
                return category

        return "general"

    def analyze_sector_sentiment(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment for different market sectors based on news.

        Args:
            news_data: List of parsed news items from gather_market_news.

        Returns:
            Dictionary containing sector sentiment analysis.
        """
        sectors = {
            "technology": {"count": 0, "items": []},
            "healthcare": {"count": 0, "items": []},
            "finance": {"count": 0, "items": []},
            "energy": {"count": 0, "items": []},
            "consumer": {"count": 0, "items": []},
            "industrial": {"count": 0, "items": []},
            "general": {"count": 0, "items": []},
        }

        category_to_sector = {
            "sector_tech": "technology",
            "sector_healthcare": "healthcare",
            "sector_finance": "finance",
            "sector_energy": "energy",
            "earnings": "general",
            "economic": "general",
            "geopolitical": "general",
            "merger_acquisition": "general",
            "ipo": "general",
            "crypto": "technology",
            "general": "general",
        }

        if isinstance(news_data, list):
            for item in news_data:
                category = item.get("category", "general")
                sector = category_to_sector.get(category, "general")
                sectors[sector]["count"] += 1
                sectors[sector]["items"].append(item.get("title", ""))

        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "sector_summary": {
                sector: {
                    "news_count": data["count"],
                    "sample_headlines": data["items"][:5],
                }
                for sector, data in sectors.items()
                if data["count"] > 0
            },
        }

    def run_complete_pipeline(self, snippet_count: int = 500, region: str = "US") -> Dict[str, Any]:
        """Run the complete data pipeline: news gathering + sector analysis.

        Args:
            snippet_count: Number of news snippets to retrieve.
            region: Market region.

        Returns:
            Complete pipeline results.
        """
        # Step 1: Gather market news
        news_result = self.gather_market_news(snippet_count, region)
        if news_result["status"] != "success":
            return {"status": "error", "error": "Failed to gather news data", "details": news_result}

        # Step 2: Analyze sector sentiment
        sentiment_result = self.analyze_sector_sentiment(news_result["news_data"])

        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "region": region,
            "pipeline_steps": {
                "news_gathering": news_result,
                "sentiment_analysis": sentiment_result,
            },
            "summary": {
                "total_news_items": news_result.get("total_items", 0),
                "sectors_covered": len(sentiment_result.get("sector_summary", {})),
            },
        }

    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save pipeline results to a JSON file.

        Args:
            results: Pipeline results to save.
            filename: Optional custom filename.

        Returns:
            Path to saved file.
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_analysis_{timestamp}.json"

        filepath = os.path.join("data", filename)
        os.makedirs("data", exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return filepath


def main():
    """Main function to run the data pipeline."""
    try:
        pipeline = StockMarketDataPipeline()
        results = pipeline.run_complete_pipeline()

        if results["status"] == "success":
            filepath = pipeline.save_results(results)
            print(f"Pipeline completed successfully. Results saved to: {filepath}")
            print(f"Total news items: {results['summary']['total_news_items']}")
            print(f"Sectors covered: {results['summary']['sectors_covered']}")
        else:
            print(f"Pipeline failed: {results.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"Error running pipeline: {str(e)}")


if __name__ == "__main__":
    main()
