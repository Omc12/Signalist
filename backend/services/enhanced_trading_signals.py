"""
Enhanced Trading Signals Service with Preprocessing Support
"""

import pandas as pd
import numpy as np
import joblib
import os
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Model path - use relative path from this file's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.path.join(BASE_DIR, 'models', 'enhanced_trading_model.pkl')

class EnhancedFeatureEngineer:
    """Feature engineering matching the training pipeline"""
    
    @staticmethod
    def calculate_technical_indicators(df, ticker=None):
        """Calculate all technical indicators used in training"""
        df = df.copy()
        close = df['Close']
        high = df['High']
        low = df['Low']
        open_ = df['Open']
        volume = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)
        
        # ============ RETURNS ============
        df['returns'] = close.pct_change()
        df['returns_1d'] = close.pct_change()
        df['returns_2d'] = close.pct_change(2)
        df['returns_3d'] = close.pct_change(3)
        df['returns_5d'] = close.pct_change(5)
        df['returns_10d'] = close.pct_change(10)
        df['returns_20d'] = close.pct_change(20)
        
        # ============ MOMENTUM ============
        df['momentum_3'] = close.pct_change(3)
        df['momentum_5'] = close.pct_change(5)
        df['momentum_10'] = close.pct_change(10)
        df['momentum_20'] = close.pct_change(20)
        
        # ============ MOVING AVERAGES ============
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = close.rolling(period).mean()
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
            df[f'price_to_sma_{period}'] = close / df[f'sma_{period}']
            df[f'price_to_ema_{period}'] = close / df[f'ema_{period}']
        
        # ============ BOLLINGER BANDS ============
        bb_window = 20
        bb_std = close.rolling(bb_window).std()
        bb_mean = close.rolling(bb_window).mean()
        df['bb_upper'] = bb_mean + (2 * bb_std)
        df['bb_lower'] = bb_mean - (2 * bb_std)
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_mean
        df['bb_squeeze'] = df['bb_width']
        
        # ============ RSI ============
        delta = close.diff()
        for period in [7, 14, 21]:
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        df['rsi_14_slope'] = df['rsi_14'].diff(5)
        
        # ============ MACD ============
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # ============ STOCHASTIC ============
        high_14 = high.rolling(14).max()
        low_14 = low.rolling(14).min()
        df['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # ============ VOLATILITY ============
        df['volatility_5'] = df['returns_1d'].rolling(5).std()
        df['volatility_10'] = df['returns_1d'].rolling(10).std()
        df['volatility_20'] = df['returns_1d'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_5'] / (df['volatility_20'] + 1e-10)
        
        # ATR
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        df['true_range'] = np.maximum(tr1, np.maximum(tr2, tr3))
        df['atr_14'] = df['true_range'].rolling(14).mean()
        df['atr_ratio'] = df['true_range'] / (df['atr_14'] + 1e-10)
        df['atr'] = df['atr_14']
        
        # ============ VOLUME ============
        df['volume_sma_20'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / (df['volume_sma_20'] + 1)
        df['volume_sma_10'] = volume.rolling(10).mean()
        df['volume_trend'] = df['volume_sma_10'] / (df['volume_sma_20'] + 1)
        df['obv'] = (np.sign(close.diff()) * volume).cumsum()
        df['obv_sma'] = df['obv'].rolling(20).mean()
        df['obv_trend'] = df['obv'] / (df['obv_sma'].abs() + 1)
        
        # ============ SUPPORT/RESISTANCE ============
        df['high_20'] = high.rolling(20).max()
        df['low_20'] = low.rolling(20).min()
        df['dist_to_high_20'] = (df['high_20'] - close) / close
        df['dist_to_low_20'] = (close - df['low_20']) / close
        df['range_position'] = (close - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-10)
        df['distance_to_support'] = df['dist_to_low_20']
        df['distance_to_resistance'] = df['dist_to_high_20']
        
        # ============ WILLIAMS %R ============
        df['williams_r'] = -100 * (high_14 - close) / (high_14 - low_14 + 1e-10)
        
        # ============ CCI ============
        typical_price = (high + low + close) / 3
        df['cci'] = (typical_price - typical_price.rolling(20).mean()) / (0.015 * typical_price.rolling(20).std() + 1e-10)
        
        # ============ ADX ============
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)
        tr_14 = df['true_range'].rolling(14).sum()
        df['plus_di'] = 100 * (plus_dm.rolling(14).sum() / (tr_14 + 1e-10))
        df['minus_di'] = 100 * (minus_dm.rolling(14).sum() / (tr_14 + 1e-10))
        dx = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-10)
        df['adx'] = dx.rolling(14).mean()
        
        # ============ GAPS ============
        df['gap'] = (open_ - close.shift(1)) / close.shift(1)
        
        # ============ PRICE SLOPE ============
        df['price_slope'] = close.pct_change(5)
        
        # ============ PATTERNS ============
        df['higher_high'] = (high > high.shift()).astype(int)
        df['lower_low'] = (low < low.shift()).astype(int)
        df['inside_bar'] = ((high <= high.shift()) & (low >= low.shift())).astype(int)
        df['overnight_return'] = (open_ - close.shift()) / close.shift()
        df['intraday_return'] = (close - open_) / open_
        df['price_range'] = (high - low) / close
        df['rate_of_change'] = close.pct_change(10) * 100
        
        # ============ FUNDAMENTAL DATA (from yfinance) ============
        if ticker:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                df['fund_pe_ratio'] = info.get('trailingPE', info.get('forwardPE', 0)) or 0
                df['fund_dividend_yield'] = info.get('dividendYield', 0) or 0
                df['fund_forward_eps'] = info.get('forwardEps', 0) or 0
                df['fund_target_mean_price'] = info.get('targetMeanPrice', 0) or 0
            except:
                df['fund_pe_ratio'] = 0
                df['fund_dividend_yield'] = 0
                df['fund_forward_eps'] = 0
                df['fund_target_mean_price'] = 0
        else:
            df['fund_pe_ratio'] = 0
            df['fund_dividend_yield'] = 0
            df['fund_forward_eps'] = 0
            df['fund_target_mean_price'] = 0
        
        return df

def load_enhanced_model():
    """Load the enhanced model with preprocessing components"""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Enhanced model not found at {MODEL_FILE}")
    
    try:
        model_data = joblib.load(MODEL_FILE)
        
        # Handle both old and new model formats
        if isinstance(model_data, dict):
            model = model_data['model']
            scaler = model_data.get('scaler')
            selector = model_data.get('selector')
            selected_features = model_data.get('selected_features', model_data.get('features', []))
            metadata = model_data.get('metadata', {})
        else:
            # Old format - just the model
            model = model_data
            scaler = None
            selector = None
            selected_features = []
            metadata = {}
        
        print(f"✅ Enhanced model loaded: {metadata.get('algorithm', 'Unknown')} "
              f"(Test AUC: {metadata.get('test_auc', 'N/A'):.4f})")
        
        return {
            'model': model,
            'scaler': scaler,
            'selector': selector,
            'features': selected_features,
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"❌ Failed to load enhanced model: {e}")
        raise


def get_stock_data(ticker, period='1y'):
    """Fetch stock data from Yahoo Finance"""
    try:
        # Handle ticker suffix - support both NSE (.NS) and BSE (.BO)
        if not ticker.endswith('.NS') and not ticker.endswith('.BO'):
            # Try NSE first, then BSE if NSE fails
            nse_ticker = f"{ticker}.NS"
            stock = yf.Ticker(nse_ticker)
            data = stock.history(period=period)
            
            if data.empty:
                # Try BSE
                bse_ticker = f"{ticker}.BO"
                stock = yf.Ticker(bse_ticker)
                data = stock.history(period=period)
                ticker = bse_ticker
            else:
                ticker = nse_ticker
        else:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
        
        if data.empty:
            raise ValueError(f"No data available for {ticker}")
        
        # Reset index to have Date as column
        data = data.reset_index()
        data = data.set_index('Date')
        
        # Add Symbol column
        data['Symbol'] = ticker
        
        return data
        
    except Exception as e:
        raise ValueError(f"Failed to fetch data for {ticker}: {str(e)}")


def calculate_rag_adjustment(rag_features: dict) -> float:
    """
    Calculate probability adjustment based on RAG news features.
    
    Returns a value between -0.20 and +0.20 to add to model probability.
    Positive = bullish news, Negative = bearish news.
    """
    if not rag_features:
        return 0.0
    
    # Extract features with defaults
    sentiment = rag_features.get('rag_sentiment', 0.0)  # -1 to 1
    strength = rag_features.get('rag_sentiment_strength', 0.0)  # 0 to 1
    confidence = rag_features.get('rag_confidence', 0.0)  # 0 to 1
    num_bullish = rag_features.get('num_bullish_drivers', 0)
    num_bearish = rag_features.get('num_bearish_risks', 0)
    event_present = rag_features.get('event_present', 0)
    uncertainty = rag_features.get('uncertainty_present', 0)
    
    # Base adjustment from sentiment (weighted by strength and confidence)
    base_adjustment = sentiment * strength * confidence * 0.15  # Max ±15%
    
    # Driver/risk balance adjustment
    driver_balance = (num_bullish - num_bearish) * 0.02  # ±2% per driver difference
    driver_balance = max(-0.06, min(0.06, driver_balance))  # Cap at ±6%
    
    # Event boost (events make news more impactful)
    event_multiplier = 1.3 if event_present else 1.0
    
    # Uncertainty penalty (reduce adjustment magnitude)
    uncertainty_factor = 0.7 if uncertainty else 1.0
    
    # Combine all factors
    adjustment = (base_adjustment + driver_balance) * event_multiplier * uncertainty_factor
    
    # Final clamp to ±20%
    adjustment = max(-0.20, min(0.20, adjustment))
    
    return adjustment


def predict(ticker, owns_stock=False, rag_features=None):
    """
    Enhanced prediction with preprocessing pipeline
    """
    try:
        # Load model components
        model_components = load_enhanced_model()
        model = model_components['model']
        scaler = model_components['scaler']
        selector = model_components['selector']
        selected_features = model_components.get('selected_features', model_components.get('features', []))
        metadata = model_components['metadata']
        
        # Get stock data (this also resolves the correct ticker suffix)
        stock_data = get_stock_data(ticker)
        
        # Get the resolved ticker from the data (handles both .NS and .BO)
        clean_ticker = stock_data['Symbol'].iloc[0] if 'Symbol' in stock_data.columns else ticker
        if not clean_ticker.endswith('.NS') and not clean_ticker.endswith('.BO'):
            clean_ticker = f"{clean_ticker}.NS"
        
        # Feature engineering (pass ticker for fundamental data)
        featured_data = EnhancedFeatureEngineer.calculate_technical_indicators(stock_data, clean_ticker)
        
        # Get latest data point
        latest_data = featured_data.iloc[-1:].copy()

        # Inject RAG features if provided
        if rag_features is not None:
            for key, value in rag_features.items():
                latest_data[key] = value

        else:
            # Ensure RAG feature columns exist even if not provided
            latest_data["rag_sentiment"] = 0.0
            latest_data["rag_sentiment_strength"] = 0.0
            latest_data["rag_confidence"] = 0.0
            latest_data["num_bullish_drivers"] = 0
            latest_data["num_bearish_risks"] = 0
            latest_data["event_present"] = 0
            latest_data["uncertainty_present"] = 0


        # Use the saved selected_features from training
        if selected_features and len(selected_features) > 0:
            # Filter to available features
            available_features = [f for f in selected_features if f in latest_data.columns]
            missing_features = [f for f in selected_features if f not in latest_data.columns]
            
            if missing_features:
                print(f"Warning: Missing {len(missing_features)} features, filling with 0")
            
            # Create feature matrix with all required features
            X = pd.DataFrame(index=latest_data.index)
            for f in selected_features:
                if f in latest_data.columns:
                    X[f] = latest_data[f].values
                else:
                    X[f] = 0
            
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
            # Apply scaler if available
            if scaler is not None:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X.values
            
            X_processed = X_scaled
        else:
            # Fallback: use common technical features
            common_features = ['returns', 'rsi_14', 'macd', 'bb_position', 'volatility_20', 
                              'momentum_10', 'momentum_20', 'stoch_k', 'stoch_d', 'adx']
            available_features = [f for f in common_features if f in latest_data.columns]
            X = latest_data[available_features].fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            X_processed = X.values
        
        # Make prediction
        try:
            proba = model.predict_proba(X_processed)
            probability = float(proba[0][1])
            print(f"✅ Prediction for {ticker}: {probability:.4f} (using {len(selected_features)} features)")
            
            # Apply RAG-based adjustment if features provided
            if rag_features:
                rag_adjustment = calculate_rag_adjustment(rag_features)
                original_prob = probability
                probability += rag_adjustment
                probability = max(0.0, min(1.0, probability))  # Clamp to [0, 1]
                
                if abs(rag_adjustment) > 0.01:
                    print(f"📰 RAG adjustment: {original_prob:.4f} → {probability:.4f} (Δ {rag_adjustment:+.4f})")
                    
        except Exception as pred_error:
            print(f"❌ Prediction error for {ticker}: {pred_error}")
            # Use technical indicators for a simple rule-based fallback
            rsi = latest_data.get('rsi_14', pd.Series([50])).iloc[0]
            macd = latest_data.get('macd', pd.Series([0])).iloc[0]
            bb_pos = latest_data.get('bb_position', pd.Series([0.5])).iloc[0]
            
            # Simple rule-based probability when model fails
            score = 0.5
            if pd.notna(rsi):
                if rsi < 30:
                    score += 0.15  # Oversold = bullish
                elif rsi > 70:
                    score -= 0.15  # Overbought = bearish
            if pd.notna(macd):
                if macd > 0:
                    score += 0.1
                else:
                    score -= 0.1
            if pd.notna(bb_pos):
                if bb_pos < 0.2:
                    score += 0.1  # Near lower band = bullish
                elif bb_pos > 0.8:
                    score -= 0.1  # Near upper band = bearish
            
            probability = max(0.1, min(0.9, score))
            print(f"⚠️ Using rule-based fallback for {ticker}: {probability:.4f}")
        
        # Generate portfolio-aware signal
        signal_info = generate_portfolio_signal(probability, owns_stock, ticker)
        
        return {
            'ticker': ticker,
            'probability': float(probability),
            'signal': signal_info['signal'],
            'action': signal_info['action'],
            'reason': signal_info['reason'],
            'confidence': signal_info['confidence'],
            'owns_stock': owns_stock,
            'model_info': {
                'version': metadata.get('model_version', metadata.get('version', 'legacy')),
                'algorithm': metadata.get('algorithm', 'Unknown'),
                'test_auc': metadata.get('test_auc'),
                'features_used': len(selected_features) if selected_features else X_processed.shape[1] if hasattr(X_processed, 'shape') else 0
            },
            'technical_data': {
                'rsi_14': float(latest_data['rsi_14'].iloc[0]) if 'rsi_14' in latest_data.columns and not pd.isna(latest_data['rsi_14'].iloc[0]) else None,
                'macd': float(latest_data['macd'].iloc[0]) if 'macd' in latest_data.columns and not pd.isna(latest_data['macd'].iloc[0]) else None,
                'bb_position': float(latest_data['bb_position'].iloc[0]) if 'bb_position' in latest_data.columns and not pd.isna(latest_data['bb_position'].iloc[0]) else None,
                'volatility_20': float(latest_data['volatility_20'].iloc[0]) if 'volatility_20' in latest_data.columns and not pd.isna(latest_data['volatility_20'].iloc[0]) else None
            }
        }
        
    except Exception as e:
        raise ValueError(f"Failed to predict for {ticker}: {str(e)}")


def generate_portfolio_signal(probability, owns_stock, ticker):
    """Generate portfolio-aware trading signal"""
    
    # Thresholds for different scenarios
    if owns_stock:
        # Conservative thresholds when you own the stock
        if probability >= 0.7:
            return {
                'signal': 'HOLD',
                'action': 'Keep your position - strong upward probability',
                'reason': f'Model predicts {probability:.1%} upward probability. Hold for potential gains.',
                'confidence': 'High' if probability >= 0.8 else 'Medium'
            }
        elif probability >= 0.4:
            return {
                'signal': 'HOLD',
                'action': 'Hold position - neutral outlook',
                'reason': f'Model shows {probability:.1%} upward probability. Moderate outlook suggests holding.',
                'confidence': 'Medium'
            }
        else:
            return {
                'signal': 'SELL',
                'action': 'Consider selling - weak prospects',
                'reason': f'Model predicts only {probability:.1%} upward probability. Consider taking profits or cutting losses.',
                'confidence': 'Medium' if probability <= 0.3 else 'Low'
            }
    
    else:
        # More aggressive thresholds when you don't own the stock
        if probability >= 0.65:
            return {
                'signal': 'BUY',
                'action': 'Strong buy signal - high upward probability',
                'reason': f'Model predicts {probability:.1%} upward probability. Good entry opportunity.',
                'confidence': 'High' if probability >= 0.75 else 'Medium'
            }
        elif probability >= 0.45:
            return {
                'signal': 'WAIT',
                'action': 'Wait for better entry - unclear direction',
                'reason': f'Model shows {probability:.1%} upward probability. Wait for clearer signals.',
                'confidence': 'Medium'
            }
        else:
            return {
                'signal': 'WAIT',
                'action': 'Wait - bearish outlook',
                'reason': f'Model predicts only {probability:.1%} upward probability. Wait for better conditions.',
                'confidence': 'Medium'
            }