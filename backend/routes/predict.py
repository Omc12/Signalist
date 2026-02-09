"""
Signalist prediction routes - Production version.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime
from services.enhanced_trading_signals import predict as enhanced_predict
from services.news_signal_service import get_news_signal_features

router = APIRouter()


@router.get("/predict")
def predict_stock(
    ticker: Optional[str] = Query(None, description="Stock ticker like RELIANCE.NS"),
    owns_stock: bool = Query(False, description="Whether you currently own this stock"),
    use_rag: bool = Query(True, description="Whether to use RAG news features")
):
    """
    Get portfolio-aware trading signals for a stock.
    Returns BUY/WAIT if you don't own the stock, or HOLD/SELL if you do.
    Optionally includes RAG-based news analysis.
    """
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")
    
    ticker = ticker.strip().upper()
    
    try:
        # Fetch RAG news features if enabled
        rag_features = None
        if use_rag:
            today = datetime.now().strftime("%Y-%m-%d")
            rag_features = get_news_signal_features(ticker, today)
        
        prediction = enhanced_predict(ticker, owns_stock=owns_stock, rag_features=rag_features)
        return prediction
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
