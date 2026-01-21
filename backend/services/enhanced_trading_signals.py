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
    def calculate_technical_indicators(df):
        """Calculate all technical indicators used in training"""
        df = df.copy()
        
        # Basic returns and momentum
        df['returns'] = df['Close'].pct_change()
        df['momentum_10'] = df['Close'].pct_change(10)
        df['momentum_20'] = df['Close'].pct_change(20)
        
        # Moving averages
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        
        # Relative positioning
        df['price_to_sma_10'] = df['Close'] / df['sma_10']
        df['price_to_sma_20'] = df['Close'] / df['sma_20']
        df['price_to_sma_50'] = df['Close'] / df['sma_50']
        
        # Bollinger Bands
        bb_window = 20
        bb_std = df['Close'].rolling(bb_window).std()
        bb_mean = df['Close'].rolling(bb_window).mean()
        df['bb_upper'] = bb_mean + (2 * bb_std)
        df['bb_lower'] = bb_mean - (2 * bb_std)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / bb_mean
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Short-term RSI
        gain_7 = (delta.where(delta > 0, 0)).rolling(7).mean()
        loss_7 = (-delta.where(delta < 0, 0)).rolling(7).mean()
        rs_7 = gain_7 / loss_7
        df['rsi_7'] = 100 - (100 / (1 + rs_7))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Stochastic
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
        
        # Volatility measures
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(14).mean()
        
        # Support and resistance (simplified)
        df['high_20'] = df['High'].rolling(20).max()
        df['low_20'] = df['Low'].rolling(20).min()
        df['distance_to_support'] = (df['Close'] - df['low_20']) / df['Close']
        df['distance_to_resistance'] = (df['high_20'] - df['Close']) / df['Close']
        
        # Overnight and intraday returns
        df['overnight_return'] = (df['Open'] - df['Close'].shift()) / df['Close'].shift()
        df['intraday_return'] = (df['Close'] - df['Open']) / df['Open']
        
        # Price range
        df['price_range'] = (df['High'] - df['Low']) / df['Close']
        
        # ADX (simplified)
        df['adx'] = df['atr'].rolling(14).mean()  # Simplified version
        
        # Williams %R
        df['williams_r'] = -100 * (high_14 - df['Close']) / (high_14 - low_14)
        
        # Rate of change
        df['rate_of_change'] = df['Close'].pct_change(10) * 100
        
        # Pattern features (simplified)
        df['higher_high'] = (df['High'] > df['High'].shift()).astype(int)
        df['lower_low'] = (df['Low'] < df['Low'].shift()).astype(int)
        df['inside_bar'] = ((df['High'] <= df['High'].shift()) & 
                           (df['Low'] >= df['Low'].shift())).astype(int)
        
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
        # Clean ticker
        if not ticker.endswith('.NS'):
            ticker = f"{ticker}.NS"
        
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


def predict(ticker, owns_stock=False):
    """
    Enhanced prediction with preprocessing pipeline
    """
    try:
        # Load model components
        model_components = load_enhanced_model()
        model = model_components['model']
        scaler = model_components['scaler']
        selector = model_components['selector']
        selected_features = model_components['features']
        metadata = model_components['metadata']
        
        # Get stock data
        stock_data = get_stock_data(ticker)
        
        # Feature engineering
        featured_data = EnhancedFeatureEngineer.calculate_technical_indicators(stock_data)
        
        # Get latest data point
        latest_data = featured_data.iloc[-1:].copy()
        
        # Prepare features based on model version
        if metadata.get('model_version', '').startswith('4.0'):
            # New model format with preprocessing
            
            # Core feature list (matching training)
            core_features = [
                'returns', 'momentum_10', 'momentum_20',
                'price_to_sma_10', 'price_to_sma_20', 'price_to_sma_50',
                'bb_position', 'distance_to_support', 'distance_to_resistance',
                'rsi_14', 'rsi_7', 'macd', 'macd_histogram', 'stoch_k',
                'volatility_10', 'volatility_20', 'atr', 'bb_squeeze',
                'overnight_return', 'intraday_return', 'price_range',
                'adx', 'williams_r', 'rate_of_change',
                'higher_high', 'lower_low', 'inside_bar'
            ]
            
            # Filter available features
            available_features = [f for f in core_features if f in latest_data.columns]
            X = latest_data[available_features].fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
            # Apply preprocessing if available
            if scaler is not None:
                X_scaled = pd.DataFrame(
                    scaler.transform(X),
                    columns=X.columns,
                    index=X.index
                )
            else:
                X_scaled = X
            
            if selector is not None:
                X_processed = pd.DataFrame(
                    selector.transform(X_scaled),
                    columns=selected_features,
                    index=X.index
                )
            else:
                X_processed = X_scaled
        
        else:
            # Legacy model format
            if selected_features:
                available_features = [f for f in selected_features if f in latest_data.columns]
            else:
                # Fallback to common features
                available_features = ['returns', 'rsi_14', 'macd', 'bb_position', 'volatility_20']
                available_features = [f for f in available_features if f in latest_data.columns]
            
            X_processed = latest_data[available_features].fillna(0)
            X_processed = X_processed.replace([np.inf, -np.inf], 0)
        
        # Make prediction
        try:
            # Ensure features match model expectations
            if hasattr(model, 'feature_names_in_'):
                expected_features = list(model.feature_names_in_)
                # Reorder columns to match expected order
                missing = [f for f in expected_features if f not in X_processed.columns]
                if missing:
                    print(f"Warning: Missing features {missing}, using 0 as default")
                    for f in missing:
                        X_processed[f] = 0
                X_processed = X_processed[expected_features]
            
            proba = model.predict_proba(X_processed)
            probability = float(proba[0][1])
            print(f"✅ Prediction for {ticker}: {probability:.4f}")
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
                'version': metadata.get('model_version', 'legacy'),
                'algorithm': metadata.get('algorithm', 'Unknown'),
                'test_auc': metadata.get('test_auc'),
                'features_used': len(X_processed.columns)
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