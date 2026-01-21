"""
Stock data routes for frontend compatibility.
"""
from fastapi import APIRouter, Query, HTTPException
import yfinance as yf
import pandas as pd
import json
import os
import requests
from typing import Optional

router = APIRouter()

# Load fallback stock database
def load_stock_database():
    """Load all NSE stocks from JSON file as fallback"""
    json_path = os.path.join(os.path.dirname(__file__), "..", "data", "stocks_nse.json")
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Cache the stock database for fallback
STOCK_DATABASE = load_stock_database()


def search_stocks_api(query: str, limit: int = 8):
    """
    Real-time API search using Yahoo Finance API.
    Searches through ALL Indian stocks dynamically.
    """
    if not query:
        # Return top popular stocks when no query
        return STOCK_DATABASE[:limit]
    
    try:
        # Yahoo Finance search endpoint
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {
            "q": query,
            "quotesCount": limit,
            "newsCount": 0,
            "listsCount": 0,
            "quotesQueryId": "tss_match_phrase_query"
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=3)
        response.raise_for_status()
        data = response.json()
        
        stocks = []
        quotes = data.get("quotes", [])
        
        for quote in quotes:
            symbol = quote.get("symbol", "")
            
            # Filter for Indian stocks (NSE/BSE)
            if ".NS" in symbol or ".BO" in symbol:
                exchange = "NSE" if ".NS" in symbol else "BSE"
                
                stocks.append({
                    "symbol": symbol,
                    "name": quote.get("longname") or quote.get("shortname") or symbol,
                    "sector": quote.get("sector", quote.get("industry", "Unknown")),
                    "exchange": exchange,
                    "quoteType": quote.get("quoteType", "EQUITY")
                })
        
        # If API returns results, use them
        if stocks:
            return stocks[:limit]
        
        # Fallback to local database search if API returns no results
        return search_stocks_local(query, limit)
        
    except Exception as e:
        print(f"Yahoo Finance API search failed: {e}, falling back to local search")
        # Fallback to local database search
        return search_stocks_local(query, limit)


def search_stocks_local(query: str, limit: int = 8):
    """
    Fallback local search with word-by-word matching.
    """
    if not query:
        return STOCK_DATABASE[:limit]
    
    query_lower = query.lower().strip()
    search_words = query_lower.split()
    
    scored_stocks = []
    
    for stock in STOCK_DATABASE:
        name_lower = stock["name"].lower()
        symbol_lower = stock["symbol"].lower().replace(".ns", "").replace(".bo", "")
        sector_lower = stock.get("sector", "").lower()
        
        score = 0
        matched_words = 0
        
        # Check each search word
        for word in search_words:
            # Exact symbol match (highest priority)
            if word == symbol_lower:
                score += 100
                matched_words += 1
            # Symbol starts with word
            elif symbol_lower.startswith(word):
                score += 80
                matched_words += 1
            # Symbol contains word
            elif word in symbol_lower:
                score += 50
                matched_words += 1
            # Name starts with word (high priority)
            elif name_lower.startswith(word):
                score += 70
                matched_words += 1
            # Any word in name starts with search word
            elif any(name_word.startswith(word) for name_word in name_lower.split()):
                score += 60
                matched_words += 1
            # Name contains word
            elif word in name_lower:
                score += 40
                matched_words += 1
            # Sector match
            elif word in sector_lower:
                score += 20
                matched_words += 1
        
        # Only include stocks that match at least one search word
        if matched_words > 0:
            # Boost score if all words match
            if matched_words == len(search_words):
                score += 30
            
            scored_stocks.append({
                "stock": stock,
                "score": score,
                "matched_words": matched_words
            })
    
    # Sort by score (descending), then by matched words (descending)
    scored_stocks.sort(key=lambda x: (x["score"], x["matched_words"]), reverse=True)
    
    # Return top results
    return [item["stock"] for item in scored_stocks[:limit]]


@router.get("/stocks")
def get_stocks(search: Optional[str] = Query(None), limit: int = Query(8)):
    """Get stock list with real-time API search"""
    
    if search:
        # Use real-time API search
        stocks = search_stocks_api(search, limit)
    else:
        # Return popular/top stocks when no search
        stocks = STOCK_DATABASE[:limit]
    
    return {
        "stocks": stocks,
        "total": len(stocks)
    }


@router.get("/stocks/details")
def get_stock_details(ticker: str = Query(...)):
    """Get comprehensive real-time stock details using yfinance"""
    try:
        # Clean ticker
        if not ticker.endswith('.NS'):
            ticker = f"{ticker}.NS"
        
        # Get data from yfinance
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get latest intraday data (1-minute intervals for real-time price)
        hist_intraday = stock.history(period="1d", interval="1m")
        # Get recent daily data for previous close
        hist_daily = stock.history(period="5d", interval="1d")
        
        # Determine latest price source (prefer intraday if market is open, else use info)
        if not hist_intraday.empty and len(hist_intraday) > 0:
            latest = hist_intraday.iloc[-1]
            latest_price = float(latest["Close"])
            latest_open = float(latest["Open"]) if not pd.isna(latest["Open"]) else info.get("regularMarketOpen", latest_price)
            latest_high = float(hist_intraday["High"].max())
            latest_low = float(hist_intraday["Low"].min())
            latest_volume = int(hist_intraday["Volume"].sum())
            latest_time = hist_intraday.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        elif not hist_daily.empty:
            latest = hist_daily.iloc[-1]
            latest_price = float(latest["Close"])
            latest_open = float(latest["Open"])
            latest_high = float(latest["High"])
            latest_low = float(latest["Low"])
            latest_volume = int(latest["Volume"])
            latest_time = hist_daily.index[-1].strftime("%Y-%m-%d")
        else:
            # Fallback to info values
            latest_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            latest_open = info.get("regularMarketOpen", latest_price)
            latest_high = info.get("regularMarketDayHigh", latest_price)
            latest_low = info.get("regularMarketDayLow", latest_price)
            latest_volume = info.get("regularMarketVolume", 0)
            latest_time = "N/A"
        
        # Get previous close
        if not hist_daily.empty and len(hist_daily) > 1:
            prev_close = float(hist_daily.iloc[-2]["Close"])
        else:
            prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose", latest_price)
        
        # Build comprehensive response with real-time data
        response = {
            # Basic Info
            "ticker": ticker,
            "symbol": ticker,
            "longName": info.get("longName", ticker.replace(".NS", "")),
            "shortName": info.get("shortName", ticker.replace(".NS", "")),
            "name": info.get("longName", ticker.replace(".NS", "")),
            "lastUpdateTime": latest_time,
            "marketState": info.get("marketState", "CLOSED"),
            
            # Real-time Price Data
            "currentPrice": float(latest_price),
            "previousClose": float(prev_close),
            "regularMarketPrice": float(latest_price),
            "regularMarketPreviousClose": float(prev_close),
            "open": float(latest_open),
            "regularMarketOpen": float(latest_open),
            "dayHigh": float(latest_high),
            "dayLow": float(latest_low),
            "regularMarketDayHigh": float(latest_high),
            "regularMarketDayLow": float(latest_low),
            "high": float(latest_high),
            "low": float(latest_low),
            
            # Intraday Changes
            "regularMarketChange": float(latest_price - prev_close),
            "regularMarketChangePercent": float((latest_price - prev_close) / prev_close * 100) if prev_close > 0 else 0,
            
            # 52 Week Range
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "fiftyTwoWeekChange": info.get("52WeekChange"),
            "fiftyTwoWeekChangePercent": info.get("fiftyTwoWeekChangePercent"),
            "fiftyTwoWeekHighChange": info.get("fiftyTwoWeekHighChange"),
            "fiftyTwoWeekHighChangePercent": info.get("fiftyTwoWeekHighChangePercent"),
            "fiftyTwoWeekLowChange": info.get("fiftyTwoWeekLowChange"),
            "fiftyTwoWeekLowChangePercent": info.get("fiftyTwoWeekLowChangePercent"),
            "fiftyTwoWeekRange": info.get("fiftyTwoWeekRange"),
            
            # Volume
            "volume": int(latest_volume),
            "regularMarketVolume": int(latest_volume),
            "averageVolume": info.get("averageVolume"),
            "averageVolume10days": info.get("averageVolume10days"),
            "averageDailyVolume10Day": info.get("averageDailyVolume10Day"),
            "averageDailyVolume3Month": info.get("averageDailyVolume3Month"),
            
            # Moving Averages
            "fiftyDayAverage": info.get("fiftyDayAverage"),
            "fiftyDayAverageChange": info.get("fiftyDayAverageChange"),
            "fiftyDayAverageChangePercent": info.get("fiftyDayAverageChangePercent"),
            "twoHundredDayAverage": info.get("twoHundredDayAverage"),
            "twoHundredDayAverageChange": info.get("twoHundredDayAverageChange"),
            "twoHundredDayAverageChangePercent": info.get("twoHundredDayAverageChangePercent"),
            
            # Market Cap & Valuation
            "marketCap": info.get("marketCap"),
            "enterpriseValue": info.get("enterpriseValue"),
            "enterpriseToRevenue": info.get("enterpriseToRevenue"),
            "enterpriseToEbitda": info.get("enterpriseToEbitda"),
            
            # Ratios
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "priceToBook": info.get("priceToBook"),
            "priceToSalesTrailing12Months": info.get("priceToSalesTrailing12Months"),
            "pegRatio": info.get("trailingPegRatio"),
            "beta": info.get("beta"),
            
            # Earnings & Dividends
            "trailingEps": info.get("trailingEps") or info.get("epsTrailingTwelveMonths"),
            "forwardEps": info.get("forwardEps") or info.get("epsForward"),
            "epsCurrentYear": info.get("epsCurrentYear"),
            "dividendRate": info.get("dividendRate") or info.get("trailingAnnualDividendRate"),
            "dividendYield": info.get("dividendYield") or info.get("trailingAnnualDividendYield"),
            "exDividendDate": info.get("exDividendDate"),
            "payoutRatio": info.get("payoutRatio"),
            "fiveYearAvgDividendYield": info.get("fiveYearAvgDividendYield"),
            
            # Profitability
            "profitMargins": info.get("profitMargins"),
            "operatingMargins": info.get("operatingMargins"),
            "grossMargins": info.get("grossMargins"),
            "ebitdaMargins": info.get("ebitdaMargins"),
            "revenueGrowth": info.get("revenueGrowth"),
            "earningsGrowth": info.get("earningsGrowth"),
            "earningsQuarterlyGrowth": info.get("earningsQuarterlyGrowth"),
            
            # Financial Health
            "totalRevenue": info.get("totalRevenue"),
            "revenuePerShare": info.get("revenuePerShare"),
            "totalCash": info.get("totalCash"),
            "totalCashPerShare": info.get("totalCashPerShare"),
            "totalDebt": info.get("totalDebt"),
            "debtToEquity": info.get("debtToEquity"),
            "currentRatio": info.get("currentRatio"),
            "quickRatio": info.get("quickRatio"),
            "returnOnAssets": info.get("returnOnAssets"),
            "returnOnEquity": info.get("returnOnEquity"),
            "freeCashflow": info.get("freeCashflow"),
            "operatingCashflow": info.get("operatingCashflow"),
            "ebitda": info.get("ebitda"),
            "grossProfits": info.get("grossProfits"),
            "netIncomeToCommon": info.get("netIncomeToCommon"),
            
            # Business Info
            "sector": info.get("sector", "Unknown"),
            "sectorKey": info.get("sectorKey"),
            "industry": info.get("industry", "Unknown"),
            "industryKey": info.get("industryKey"),
            "fullTimeEmployees": info.get("fullTimeEmployees"),
            "website": info.get("website"),
            "irWebsite": info.get("irWebsite"),
            "country": info.get("country"),
            "city": info.get("city"),
            "address1": info.get("address1"),
            "address2": info.get("address2"),
            "zip": info.get("zip"),
            "phone": info.get("phone"),
            "fax": info.get("fax"),
            
            # Description
            "longBusinessSummary": info.get("longBusinessSummary"),
            "businessSummary": info.get("businessSummary"),
            
            # Trading Info
            "exchange": info.get("exchange", "NSE"),
            "fullExchangeName": info.get("fullExchangeName"),
            "quoteType": info.get("quoteType", "EQUITY"),
            "currency": info.get("currency", "INR"),
            "financialCurrency": info.get("financialCurrency", "INR"),
            "exchangeTimezoneName": info.get("exchangeTimezoneName"),
            "exchangeTimezoneShortName": info.get("exchangeTimezoneShortName"),
            
            # Analyst Recommendations
            "targetHighPrice": info.get("targetHighPrice"),
            "targetLowPrice": info.get("targetLowPrice"),
            "targetMeanPrice": info.get("targetMeanPrice"),
            "targetMedianPrice": info.get("targetMedianPrice"),
            "recommendationKey": info.get("recommendationKey"),
            "recommendationMean": info.get("recommendationMean"),
            "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"),
            "averageAnalystRating": info.get("averageAnalystRating"),
            
            # Additional Metrics
            "sharesOutstanding": info.get("sharesOutstanding"),
            "impliedSharesOutstanding": info.get("impliedSharesOutstanding"),
            "floatShares": info.get("floatShares"),
            "heldPercentInsiders": info.get("heldPercentInsiders"),
            "heldPercentInstitutions": info.get("heldPercentInstitutions"),
            "bookValue": info.get("bookValue"),
            "lastDividendValue": info.get("lastDividendValue"),
            "lastDividendDate": info.get("lastDividendDate"),
            "lastSplitDate": info.get("lastSplitDate"),
            "lastSplitFactor": info.get("lastSplitFactor"),
            
            # Risk Metrics
            "auditRisk": info.get("auditRisk"),
            "boardRisk": info.get("boardRisk"),
            "compensationRisk": info.get("compensationRisk"),
            "shareHolderRightsRisk": info.get("shareHolderRightsRisk"),
            "overallRisk": info.get("overallRisk"),
            
            # Historical Highs/Lows
            "allTimeHigh": info.get("allTimeHigh"),
            "allTimeLow": info.get("allTimeLow"),
            
            # Bid/Ask
            "bid": info.get("bid"),
            "ask": info.get("ask"),
            "bidSize": info.get("bidSize"),
            "askSize": info.get("askSize"),
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch stock details: {str(e)}")


@router.get("/stocks/candles") 
def get_stock_candles(
    ticker: str = Query(...),
    period: str = Query("1mo", description="1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max"),
    interval: str = Query("1d", description="1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo")
):
    """Get stock price history/candles"""
    try:
        # Clean ticker
        if not ticker.endswith('.NS'):
            ticker = f"{ticker}.NS"
            
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            raise ValueError("No data available")
        
        # Convert to candles format
        candles = []
        for index, row in hist.iterrows():
            candles.append({
                "time": index.strftime("%Y-%m-%d"),
                "date": index.strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]), 
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"])
            })
        
        return {
            "candles": candles,
            "ticker": ticker
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch candles: {str(e)}")
