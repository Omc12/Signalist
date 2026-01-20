"""
Stock data service for on-demand stock fetching.
Fetches stock data only when searched - no bulk loading.
"""
import json
import os
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from core.cache import stock_list_cache, cached
from .multi_provider_fetcher import MultiProviderStockFetcher


class StockService:
    """Service for managing stock lists with dynamic API fetching."""
    
    def __init__(self, stocks_json_path: str):
        self.stocks_json_path = stocks_json_path
        # Removed bulk fetcher - using on-demand fetching only
        self.multi_fetcher = MultiProviderStockFetcher(stocks_json_path)  # Initialize multi-provider with local DB path
        self._last_fetch = None
        self._force_refresh_hours = 24  # Refresh every 24 hours
        print("📊 Stock service initialized - On-demand fetching mode")
    
    @cached(stock_list_cache, key_func=lambda self: "all_stocks")
    def load_stocks(self) -> List[Dict]:
        """Load cached stocks from JSON only - no bulk API fetching."""
        # Only load from local cache for search suggestions
        # Actual stock data is fetched on-demand when searched
        return self._load_stocks_from_json()
    
    def _should_refresh_from_api(self) -> bool:
        """Disabled - using on-demand fetching only."""
        return False  # Never bulk refresh
    
    def _load_stocks_from_json(self) -> List[Dict]:
        """Load stocks from JSON file."""
        if not os.path.exists(self.stocks_json_path):
            print(f"JSON file not found: {self.stocks_json_path}")
            return []
        
        try:
            with open(self.stocks_json_path, 'r', encoding='utf-8') as f:
                stocks = json.load(f)
            print(f"Loaded {len(stocks)} stocks from JSON file")
            return stocks
        except Exception as e:
            print(f"Error loading stocks from JSON: {e}")
            return []
    
    def _save_stocks_to_json(self, stocks: List[Dict]) -> None:
        """Save fetched stocks to JSON file for backup."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.stocks_json_path), exist_ok=True)
            
            with open(self.stocks_json_path, 'w', encoding='utf-8') as f:
                json.dump(stocks, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(stocks)} stocks to {self.stocks_json_path}")
        except Exception as e:
            print(f"Error saving stocks to JSON: {e}")
    
    def search_stocks(self, query: str, limit: int = 10) -> List[Dict]:
        """Search stocks with on-demand API fetching - always fetch fresh data."""
        query = query.strip()
        if not query:
            return []

        print(f"[SEARCH] Searching for: '{query}' (on-demand mode)")
        
        # First check local cache for quick results
        local_results = self._search_local_stocks(query)
        print(f"  Found {len(local_results)} local cached results")
        
        # Always try API for fresh data (don't wait for limited results)
        print("  Fetching fresh data from APIs...")
        
        try:
            api_results = self._search_stocks_via_api_comprehensive(query)
            print(f"  API search returned {len(api_results)} results")
            
            if api_results:
                # Add new stocks to local database for future searches
                self._add_stocks_to_database(api_results)
                
                # Combine and rank all results (prefer fresh API data)
                all_results = api_results + local_results
                ranked_results = self._rank_search_results(all_results, query)
                print(f"[OK] Returning {len(ranked_results[:limit])} results (fresh data)")
                return ranked_results[:limit]
            elif local_results:
                print(f"[WARNING] API failed, using {len(local_results[:limit])} cached results")
                return local_results[:limit]
            else:
                print("[ERROR] No results found locally or via APIs")
                return []
                
        except Exception as e:
            print(f"[ERROR] Error in API search: {e}")
            if local_results:
                print(f"  Falling back to {len(local_results[:limit])} cached results")
                return local_results[:limit]
            return []
    
    def force_refresh_stocks(self) -> List[Dict]:
        """Force refresh stocks from API (for admin/debug use)."""
        try:
            print("Force refreshing stocks from APIs...")
            api_stocks = self.stock_fetcher.get_comprehensive_stock_list()
            
            if api_stocks:
                self._last_fetch = datetime.now()
                self._save_stocks_to_json(api_stocks)
                
                # Clear cache to force reload
                if "all_stocks" in stock_list_cache:
                    del stock_list_cache["all_stocks"]
                
                print(f"Force refresh completed: {len(api_stocks)} stocks")
                return api_stocks
            else:
                print("Force refresh failed - no stocks returned")
                return []
                
        except Exception as e:
            print(f"Error in force refresh: {e}")
            return []
    
    def get_stock_by_ticker(self, ticker: str) -> Optional[Dict]:
        """Get stock info by ticker."""
        stocks = self.load_stocks()
        ticker_upper = ticker.upper()
        
        for stock in stocks:
            if stock.get("ticker", "").upper() == ticker_upper:
                return stock
        
        return None

    def _search_local_stocks(self, query: str) -> List[Dict]:
        """Search stocks in local database with progressive word-by-word matching."""
        stocks = self.load_stocks()
        query_lower = query.lower().strip()
        query_words = query_lower.split()
        
        # Filter matching stocks with progressive scoring
        matches = []
        for stock in stocks:
            ticker = stock.get("ticker", "").lower()
            name = stock.get("name", "").lower()
            ticker_base = ticker.replace('.ns', '').replace('.bo', '')
            score = 0
            
            # Progressive matching - stricter as query grows
            # Exact ticker match (highest priority)
            if query_lower == ticker_base:
                score = 100
            # Ticker starts with query (very high priority for partial matches)
            elif ticker_base.startswith(query_lower):
                score = 95
            # For multi-word queries, require ALL words present
            elif len(query_words) > 1:
                # All words must be in name (word-by-word strict matching)
                if all(word in name for word in query_words):
                    # Calculate score based on how well words match
                    if name.startswith(query_lower):
                        score = 90
                    elif all(name.startswith(word) or f" {word}" in name for word in query_words):
                        score = 85
                    else:
                        score = 70
            # Single word queries - check name and ticker
            else:
                if name.startswith(query_lower):
                    score = 80
                elif ticker.startswith(query_lower):
                    score = 75
                elif query_lower in ticker:
                    score = 60
                elif query_lower in name:
                    # Only if word appears as a separate word or at start
                    words_in_name = name.split()
                    if query_lower in words_in_name or any(w.startswith(query_lower) for w in words_in_name):
                        score = 50
            
            if score > 0:
                stock_copy = stock.copy()
                stock_copy['_score'] = score
                matches.append(stock_copy)
        
        # Sort by score (descending), then by name
        matches.sort(key=lambda x: (-x.get('_score', 0), x.get('name', '')))
        return matches

    def _rank_search_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Rank combined search results with progressive word-by-word matching."""
        if not results:
            return []
        
        query_lower = query.lower().strip()
        query_words = query_lower.split()
        
        # Re-score all results for consistency
        for result in results:
            ticker = result.get('ticker', '').lower()
            name = result.get('name', '').lower()
            ticker_base = ticker.replace('.ns', '').replace('.bo', '')
            score = result.get('_score', 0)
            
            # If not already scored, calculate progressive score
            if score == 0:
                # Exact ticker match
                if query_lower == ticker_base:
                    score = 100
                # Ticker starts with query
                elif ticker_base.startswith(query_lower):
                    score = 95
                # Multi-word: ALL words must be present
                elif len(query_words) > 1:
                    if all(word in name for word in query_words):
                        if name.startswith(query_lower):
                            score = 90
                        elif all(name.startswith(word) or f" {word}" in name for word in query_words):
                            score = 85
                        else:
                            score = 70
                # Single word matching
                else:
                    if name.startswith(query_lower):
                        score = 80
                    elif ticker.startswith(query_lower):
                        score = 75
                    elif query_lower in ticker:
                        score = 60
                    elif query_lower in name:
                        words_in_name = name.split()
                        if query_lower in words_in_name or any(w.startswith(query_lower) for w in words_in_name):
                            score = 50
                
                result['_score'] = score
        
        # Remove duplicates (prefer higher scored items)
        seen_tickers = set()
        unique_results = []
        
        # Sort by score first, then by name
        results.sort(key=lambda x: (-x.get('_score', 0), x.get('name', '')))
        
        for result in results:
            ticker = result.get('ticker', '').upper()
            if ticker not in seen_tickers:
                seen_tickers.add(ticker)
                unique_results.append(result)
        
        return unique_results

    def _search_stocks_via_api_comprehensive(self, query: str) -> List[Dict]:
        """Comprehensive API search with intelligent symbol generation."""
        try:
            # Generate smart symbol variations for both tickers and company names
            search_terms = self._generate_comprehensive_symbols(query)
            print(f"Generated {len(search_terms)} search terms for '{query}':")
            print(f"  Symbols: {search_terms[:5]}...")
            
            # Try to fetch from external APIs 
            if hasattr(self, 'multi_fetcher') and self.multi_fetcher:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    results = loop.run_until_complete(
                        asyncio.wait_for(
                            self.multi_fetcher.fetch_comprehensive_stocks(search_terms),
                            timeout=15
                        )
                    )
                    return self._filter_api_results(results, query) if results else []
                except asyncio.TimeoutError:
                    print(f"API search timed out for '{query}'")
                    return []
                finally:
                    loop.close()
            else:
                print("Multi-provider not available for API search")
                return []
                
        except Exception as e:
            print(f"Error in comprehensive API search: {e}")
            return []
    
    def _generate_comprehensive_symbols(self, query: str) -> List[str]:
        """Generate comprehensive symbol variations for both tickers and company names."""
        query_upper = query.upper().strip()
        symbols = []
        
        # Known company name mappings (check first for exact matches)
        company_mappings = {
            "APPLE": ["AAPL"],
            "MICROSOFT": ["MSFT"], 
            "GOOGLE": ["GOOGL", "GOOG"],
            "ALPHABET": ["GOOGL", "GOOG"],
            "TESLA": ["TSLA"],
            "AMAZON": ["AMZN"],
            "FACEBOOK": ["META"],
            "META": ["META"],
            "NETFLIX": ["NFLX"],
            "NVIDIA": ["NVDA"],
            "TWITTER": ["TWTR"],
            "UBER": ["UBER"],
            "SPOTIFY": ["SPOT"],
            "ZOOM": ["ZM"],
            "OLA ELECTRIC": ["OLAELEC"],
            "TATA SILVER": ["TATSILV"],
            "TATA MOTORS": ["TATAMOTORS"],
            "TATA STEEL": ["TATASTEEL"],
            "TATA POWER": ["TATAPOWER"],
            "TATA CONSUMER": ["TATACONSUM"],
            "TATA CONSULTANCY": ["TCS"],
            "RELIANCE INDUSTRIES": ["RELIANCE"],
            "INFOSYS": ["INFY"],
            "HDFC BANK": ["HDFCBANK"],
            "ICICI BANK": ["ICICIBANK"]
        }
        
        # Check for exact company name matches first
        for company, tickers in company_mappings.items():
            if query_upper == company or company in query_upper:
                for ticker in tickers:
                    symbols.extend([f"{ticker}.NS", f"{ticker}.BO", ticker])
                # Add original query for name-based API search
                symbols.append(query)
                return list(dict.fromkeys(symbols))
        
        # For ticker-like queries (no spaces, reasonable length)
        if ' ' not in query_upper and len(query_upper) <= 10:
            symbols.extend([f"{query_upper}.NS", f"{query_upper}.BO", query_upper])
        else:
            # For company name queries, generate ticker variations
            words = query_upper.split()
            symbols.append(query)  # Original query for name-based search
            
            if len(words) >= 2:
                # Generate acronyms from first letters
                acronym = ''.join(word[:2] for word in words[:3])
                if 4 <= len(acronym) <= 8:
                    symbols.extend([f"{acronym}.NS", f"{acronym}.BO", acronym])
                
                # First word + first letter of others
                combo = words[0] + ''.join(w[0] for w in words[1:])
                if 3 <= len(combo) <= 10:
                    symbols.extend([f"{combo}.NS", f"{combo}.BO", combo])
            
            elif len(words[0]) >= 3:
                # Single word - truncate if too long
                base = words[0][:6]
                symbols.extend([f"{base}.NS", f"{base}.BO", base])
        
        return list(dict.fromkeys(symbols))
    
    def _filter_api_results(self, results: List[Dict], original_query: str) -> List[Dict]:
        """Filter and rank API results by relevance to original query."""
        if not results:
            return []
        
        query_lower = original_query.lower().strip()
        query_words = query_lower.split()
        
        # Score results by relevance
        scored = []
        for result in results:
            ticker = result.get('ticker', '').lower()
            name = result.get('name', '').lower()
            score = 0
            
            # Exact ticker match
            if query_lower == ticker.replace('.ns', '').replace('.bo', ''):
                score = 100
            # Ticker starts with query  
            elif ticker.startswith(query_lower):
                score = 80
            # Name contains all words
            elif len(query_words) > 1 and all(word in name for word in query_words):
                score = 70
            # Name contains some words
            elif any(word in name for word in query_words if len(word) > 2):
                score = 50
            # Partial match
            elif query_lower in ticker or query_lower in name:
                score = 30
            
            if score > 0:
                scored.append((score, result))
        
        # Sort by score and return top results
        scored.sort(key=lambda x: x[0], reverse=True)
        return [result for score, result in scored[:5]]
    
    def _add_stocks_to_database(self, new_stocks: List[Dict]) -> None:
        """Add newly found stocks to the local database."""
        try:
            current_stocks = self._load_stocks_from_json()
            current_tickers = {stock.get('ticker') for stock in current_stocks}
            
            # Add only new stocks
            added_count = 0
            for stock in new_stocks:
                if stock.get('ticker') not in current_tickers:
                    current_stocks.append(stock)
                    added_count += 1
            
            if added_count > 0:
                self._save_stocks_to_json(current_stocks)
                print(f"Added {added_count} new stocks to local database")
                
        except Exception as e:
            print(f"Error adding stocks to database: {e}")