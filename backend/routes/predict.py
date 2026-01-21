"""
Stock prediction routes - Production version.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from services.enhanced_trading_signals import predict as enhanced_predict

router = APIRouter()


@router.get("/predict")
def predict_stock(
    ticker: Optional[str] = Query(None, description="Stock ticker like RELIANCE.NS"),
    owns_stock: bool = Query(False, description="Whether you currently own this stock")
):
    """
    Get portfolio-aware trading signals for a stock.
    Returns BUY/WAIT if you don't own the stock, or HOLD/SELL if you do.
    """
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")
    
    ticker = ticker.strip().upper()
    
    try:
        prediction = enhanced_predict(ticker, owns_stock=owns_stock)
        return prediction
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
