"""
Configuration settings for Signalist API.
"""
import os
from pathlib import Path

# Load environment variables from .env file if it exists
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        pass

# API Settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_TITLE = "Signalist API"
API_VERSION = "4.0"

# CORS Settings
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    os.getenv("FRONTEND_URL", ""),
]
CORS_ORIGINS = [url for url in CORS_ORIGINS if url]  # Remove empty strings

# Cache Settings (in seconds)
CACHE_STOCK_DETAILS = 300  # 5 minutes
CACHE_CANDLES_DAILY = 1800  # 30 minutes
CACHE_CANDLES_INTRADAY = 30  # 30 seconds
CACHE_MODEL = 600  # 10 minutes
CACHE_STOCK_LIST = 3600  # 1 hour

# Data Settings
STOCKS_JSON_PATH = "data/stocks_nse.json"

# API Keys
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# Confidence thresholds
CONFIDENCE_HIGH = 0.65
CONFIDENCE_MEDIUM = 0.55

# Default chart settings
DEFAULT_CHART_PERIOD = "6mo"
DEFAULT_CHART_INTERVAL = "1d"
