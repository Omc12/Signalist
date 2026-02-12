def get_news_signal_features(ticker: str, date: str) -> dict:
    """
    Returns numeric RAG-based news features for a given stock and date.
    Fetches recent news from Newsdata.io and analyzes sentiment with Gemini LLM.
    
    Features returned:
    - rag_sentiment: Combined sentiment (-1 to +1)
    - rag_sentiment_strength: Sentiment strength (0-1)
    - rag_confidence: Model confidence (0-1)
    - num_bullish_drivers: Count of positive factors
    - num_bearish_risks: Count of negative factors
    - event_present: Whether events were found
    - uncertainty_present: Whether risks were found
    """
    import os
    import json
    import requests
    import logging
    from dotenv import load_dotenv
    
    try:
        import google.generativeai as genai
    except ImportError:
        return _get_neutral_features()
    
    load_dotenv()
    logger = logging.getLogger(__name__)
    
    try:
        # Check API keys
        newsdata_key = os.getenv("NEWSDATA_API_KEY", "").strip()
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        
        if not newsdata_key or newsdata_key == "your_newsdata_api_key_here":
            return _get_neutral_features()
        if not gemini_key or gemini_key == "your_gemini_api_key_here":
            return _get_neutral_features()
        
        # Map ticker to company name for better news results
        ticker_map = {
            "RELIANCE": "Reliance Industries",
            "TCS": "Tata Consultancy Services",
            "INFY": "Infosys",
            "HDFC": "HDFC Bank",
            "ICICIBANK": "ICICI Bank",
            "HDFCBANK": "HDFC Bank",
            "SBIN": "State Bank of India",
            "WIPRO": "Wipro"
        }
        # Strip suffix for mapping
        clean_ticker = ticker.split('.')[0]
        company = ticker_map.get(clean_ticker, clean_ticker)
        
        # Fetch news from Newsdata.io
        logger.debug(f"Fetching news for {ticker}...")
        response = requests.get(
            "https://newsdata.io/api/1/news",
            params={"apikey": newsdata_key, "q": company, "language": "en"},
            timeout=8
        )
        response.raise_for_status()
        
        articles = response.json().get("results", [])[:10]
        if not articles:
            logger.debug(f"No articles found for {ticker}")
            return _get_neutral_features()
        
        # Format articles for LLM analysis
        context = "\n".join([
            f"- {a.get('title', '')}: {a.get('description', '')[:200]}"
            for a in articles if a.get('title')
        ])
        
        if not context.strip():
            return _get_neutral_features()
        
        # Call Gemini for sentiment analysis
        logger.debug(f"Analyzing {len(articles)} articles with Gemini...")
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(
            "models/gemini-2.5-flash",
            generation_config={"temperature": 0, "top_p": 1}
        )
        
        prompt = f"""Analyze {ticker} news. Return ONLY JSON (no markdown):
{context}

{{
  "sentiment": 1 (bullish) / 0 (neutral) / -1 (bearish),
  "strength": 0.0 to 1.0,
  "confidence": 0.0 to 1.0,
  "drivers": ["positive_factor_1"],
  "risks": ["risk_1"]
}}"""
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Extract JSON safely
        start = text.find('{')
        end = text.rfind('}') + 1
        if start < 0 or end <= start:
            logger.warning(f"No JSON in Gemini response for {ticker}")
            return _get_neutral_features()
        
        signal = json.loads(text[start:end])
        
        # Validate and normalize values
        sentiment = signal.get("sentiment", 0)
        if sentiment not in (-1, 0, 1):
            sentiment = 0
        
        strength = max(0, min(1, float(signal.get("strength", 0))))
        confidence = max(0, min(1, float(signal.get("confidence", 0))))
        drivers = signal.get("drivers", [])
        risks = signal.get("risks", [])
        
        logger.debug(f"{ticker}: sentiment={sentiment}, strength={strength:.2f}, confidence={confidence:.2f}")
        
        return {
            "rag_sentiment": sentiment * strength * confidence,
            "rag_sentiment_strength": strength,
            "rag_confidence": confidence,
            "num_bullish_drivers": len(drivers),
            "num_bearish_risks": len(risks),
            "event_present": 1 if drivers or risks else 0,
            "uncertainty_present": 1 if risks else 0
        }
        
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing error for {ticker}: {str(e)}")
        return _get_neutral_features()
    except requests.RequestException as e:
        logger.warning(f"News API error for {ticker}: {str(e)}")
        return _get_neutral_features()
    except Exception as e:
        logger.error(f"RAG error for {ticker}: {str(e)}")
        return _get_neutral_features()


def _get_neutral_features() -> dict:
    """Returns neutral RAG features when API fails."""
    return {
        "rag_sentiment": 0.0,
        "rag_sentiment_strength": 0.0,
        "rag_confidence": 0.0,
        "num_bullish_drivers": 0,
        "num_bearish_risks": 0,
        "event_present": 0,
        "uncertainty_present": 0
    }