"""
Advanced Trading Model Training Script v8.0 (Production)
- 235K+ stock data + ETFs
- 72+ technical features + fundamental data
- GPU-accelerated ensemble training
"""

import pandas as pd
import numpy as np
import os
import glob
import joblib
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import (
    HistGradientBoostingClassifier, 
    RandomForestClassifier, 
    VotingClassifier
)
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.info

# Optional dependencies
HAS_YFINANCE = False
HAS_XGB = False
HAS_LGB = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    pass

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    pass

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    pass

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
FUNDAMENTALS_CACHE = os.path.join(DATA_DIR, 'fundamentals_cache.pkl')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'enhanced_trading_model.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)


def fetch_fundamental_data(symbols):
    """Fetch fundamental data for all symbols from Yahoo Finance"""
    if not HAS_YFINANCE:
        print("  ⚠️ yfinance not available, skipping fundamentals")
        return {}
    
    # Try to load from cache first
    if os.path.exists(FUNDAMENTALS_CACHE):
        try:
            cache = joblib.load(FUNDAMENTALS_CACHE)
            cache_age = (datetime.now() - cache.get('timestamp', datetime.min)).days
            if cache_age < 7:  # Use cache if less than 7 days old
                print(f"  ✓ Loaded fundamentals from cache ({cache_age} days old)")
                return cache.get('data', {})
        except:
            pass
    
    print("\n📊 Fetching Fundamental Data from Yahoo Finance...")
    fundamentals = {}
    
    # Special symbol mappings for NSE stocks (Yahoo Finance uses different tickers)
    SYMBOL_MAP = {
        'MM': 'M&M.NS',  # Mahindra & Mahindra
        'BAJAJ-AUTO': 'BAJAJ-AUTO.NS',
        'BHARTIARTL': 'BHARTIARTL.NS',
        'TATASTEEL': 'TATASTEEL.NS',
        'TITAN': 'TITAN.NS',
        'BRITANNIA': 'BRITANNIA.NS',
        'COALINDIA': 'COALINDIA.NS',
        'JSWSTEEL': 'JSWSTEEL.NS',
        'NESTLEIND': 'NESTLEIND.NS',
        'ICICIBANK': 'ICICIBANK.NS',
        'MARUTI': 'MARUTI.NS',
        'ULTRACEMCO': 'ULTRACEMCO.NS',
        'WIPRO': 'WIPRO.NS',
        'NTPC': 'NTPC.NS',
        'VEDL': 'VEDL.NS',
        'ASIANPAINT': 'ASIANPAINT.NS',
        'ONGC': 'ONGC.NS',
        'IOC': 'IOC.NS',
        'DRREDDY': 'DRREDDY.NS',
        'TECHM': 'TECHM.NS',
        'TCS': 'TCS.NS',
        'SUNPHARMA': 'SUNPHARMA.NS',
        'HCLTECH': 'HCLTECH.NS',
        'HDFC': 'HDFC.NS',
        'KOTAKBANK': 'KOTAKBANK.NS',
        'ZEEL': 'ZYDUSLIFE.NS',  # Zeel changed to Zydus
        'AXISBANK': 'AXISBANK.NS',
        'EICHERMOT': 'EICHERMOT.NS',
        'HDFCBANK': 'HDFCBANK.NS',
        'TATAMOTORS': 'TATAMOTORS.NS',
        'HEROMOTOCO': 'HEROMOTOCO.NS',
        'UPL': 'UPL.NS',
        'CIPLA': 'CIPLA.NS',
        'ITC': 'ITC.NS',
        'GAIL': 'GAIL.NS',
        'POWERGRID': 'POWERGRID.NS',
        'HINDALCO': 'HINDALCO.NS',
        'BPCL': 'BPCL.NS',
        'LT': 'LT.NS',
        'ADANIPORTS': 'ADANIPORTS.NS',
        'HINDUNILVR': 'HINDUNILVR.NS',
        'GRASIM': 'GRASIM.NS',
        'RELIANCE': 'RELIANCE.NS',
        'INFY': 'INFY.NS',
        'SHREECEM': 'SHREECEM.NS',
        'SBIN': 'SBIN.NS',
        'INDUSINDBK': 'INDUSINDBK.NS',
        'BAJAJFINSV': 'BAJAJFINSV.NS',
        'BAJFINANCE': 'BAJFINANCE.NS',
    }
    
    import time
    
    for symbol in symbols:
        try:
            # Convert to Yahoo Finance format using mapping
            if symbol.endswith('.NS'):
                ticker = symbol
            elif symbol in SYMBOL_MAP:
                ticker = SYMBOL_MAP[symbol]
            else:
                ticker = f"{symbol}.NS"
            
            # Add small delay to avoid rate limiting
            time.sleep(0.3)
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key fundamental metrics
            fund_data = {
                # Valuation
                'pe_ratio': info.get('trailingPE', np.nan),
                'forward_pe': info.get('forwardPE', np.nan),
                'pb_ratio': info.get('priceToBook', np.nan),
                'ps_ratio': info.get('priceToSalesTrailing12Months', np.nan),
                'peg_ratio': info.get('pegRatio', np.nan),
                'ev_ebitda': info.get('enterpriseToEbitda', np.nan),
                
                # Profitability
                'profit_margin': info.get('profitMargins', np.nan),
                'operating_margin': info.get('operatingMargins', np.nan),
                'gross_margin': info.get('grossMargins', np.nan),
                'roe': info.get('returnOnEquity', np.nan),
                'roa': info.get('returnOnAssets', np.nan),
                
                # Growth
                'earnings_growth': info.get('earningsGrowth', np.nan),
                'revenue_growth': info.get('revenueGrowth', np.nan),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', np.nan),
                
                # Financial Health
                'debt_to_equity': info.get('debtToEquity', np.nan),
                'current_ratio': info.get('currentRatio', np.nan),
                'quick_ratio': info.get('quickRatio', np.nan),
                
                # Dividends
                'dividend_yield': info.get('dividendYield', np.nan),
                'payout_ratio': info.get('payoutRatio', np.nan),
                
                # Size
                'market_cap': info.get('marketCap', np.nan),
                'enterprise_value': info.get('enterpriseValue', np.nan),
                
                # Analyst
                'target_mean_price': info.get('targetMeanPrice', np.nan),
                'recommendation': info.get('recommendationMean', np.nan),  # 1=Buy, 5=Sell
                
                # Earnings
                'trailing_eps': info.get('trailingEps', np.nan),
                'forward_eps': info.get('forwardEps', np.nan),
            }
            
            fundamentals[symbol] = fund_data
            print(f"  ✓ {symbol}")
            
        except Exception as e:
            print(f"  ✗ {symbol}: {str(e)[:30]}")
            fundamentals[symbol] = {}
    
    # Cache the data
    try:
        joblib.dump({'data': fundamentals, 'timestamp': datetime.now()}, FUNDAMENTALS_CACHE)
        print(f"  ✓ Cached fundamentals for {len(fundamentals)} symbols")
    except:
        pass
    
    return fundamentals


class AdvancedFeatureEngineer:
    """Advanced feature engineering with 60+ technical indicators"""
    
    @staticmethod
    def calculate_all_features(df, fundamentals=None):
        """Calculate comprehensive technical indicators"""
        df = df.copy()
        
        # Ensure required columns
        required = ['Open', 'High', 'Low', 'Close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing: {col}")
        
        if 'Volume' not in df.columns or df['Volume'].isna().all():
            df['Volume'] = 1000000
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        open_ = df['Open']
        volume = df['Volume'].fillna(1000000)
        
        # ================== RETURNS & MOMENTUM ==================
        df['returns_1d'] = close.pct_change(1)
        df['returns_2d'] = close.pct_change(2)
        df['returns_3d'] = close.pct_change(3)
        df['returns_5d'] = close.pct_change(5)
        df['returns_10d'] = close.pct_change(10)
        df['returns_20d'] = close.pct_change(20)
        
        df['momentum_3'] = close / close.shift(3) - 1
        df['momentum_5'] = close / close.shift(5) - 1
        df['momentum_10'] = close / close.shift(10) - 1
        df['momentum_20'] = close / close.shift(20) - 1
        
        df['acceleration'] = df['returns_1d'] - df['returns_1d'].shift(1)
        
        # ================== MOVING AVERAGES ==================
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = close.rolling(period).mean()
            df[f'ema_{period}'] = close.ewm(span=period).mean()
            df[f'price_to_sma_{period}'] = close / df[f'sma_{period}']
            df[f'price_to_ema_{period}'] = close / df[f'ema_{period}']
        
        # MA crossovers
        df['sma_5_10_cross'] = (df['sma_5'] > df['sma_10']).astype(int)
        df['sma_10_20_cross'] = (df['sma_10'] > df['sma_20']).astype(int)
        df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['ema_5_20_cross'] = (df['ema_5'] > df['ema_20']).astype(int)
        
        df['golden_cross'] = ((df['sma_20'] > df['sma_50']) & 
                              (df['sma_20'].shift(1) <= df['sma_50'].shift(1))).astype(int)
        df['death_cross'] = ((df['sma_20'] < df['sma_50']) & 
                             (df['sma_20'].shift(1) >= df['sma_50'].shift(1))).astype(int)
        
        # ================== BOLLINGER BANDS ==================
        bb_window = 20
        bb_std = close.rolling(bb_window).std()
        bb_mean = close.rolling(bb_window).mean()
        df['bb_upper'] = bb_mean + (2 * bb_std)
        df['bb_lower'] = bb_mean - (2 * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_mean
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)
        
        # ================== RSI ==================
        for period in [7, 14, 21]:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        df['rsi_14_slope'] = df['rsi_14'] - df['rsi_14'].shift(5)
        df['price_slope'] = (close - close.shift(5)) / close.shift(5) * 100
        df['rsi_divergence'] = (np.sign(df['rsi_14_slope']) != np.sign(df['price_slope'])).astype(int)
        
        # ================== MACD ==================
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_cross'] = ((df['macd'] > df['macd_signal']) & 
                            (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        
        # ================== STOCHASTIC ==================
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        df['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        df['stoch_cross'] = ((df['stoch_k'] > df['stoch_d']) & 
                             (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))).astype(int)
        
        # ================== VOLATILITY ==================
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
        
        # ================== VOLUME ==================
        df['volume_sma_10'] = volume.rolling(10).mean()
        df['volume_sma_20'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / (df['volume_sma_20'] + 1)
        df['volume_trend'] = df['volume_sma_10'] / (df['volume_sma_20'] + 1)
        
        df['obv'] = (np.sign(close.diff()) * volume).cumsum()
        df['obv_sma'] = df['obv'].rolling(20).mean()
        df['obv_trend'] = df['obv'] / (df['obv_sma'].abs() + 1)
        
        # ================== PATTERNS ==================
        df['higher_high'] = (high > high.shift(1)).astype(int)
        df['higher_low'] = (low > low.shift(1)).astype(int)
        df['lower_high'] = (high < high.shift(1)).astype(int)
        df['lower_low'] = (low < low.shift(1)).astype(int)
        
        df['consecutive_up'] = df['returns_1d'].gt(0).rolling(3).sum()
        df['consecutive_down'] = df['returns_1d'].lt(0).rolling(3).sum()
        
        df['inside_bar'] = ((high <= high.shift(1)) & (low >= low.shift(1))).astype(int)
        df['outside_bar'] = ((high > high.shift(1)) & (low < low.shift(1))).astype(int)
        
        body = np.abs(close - open_)
        range_ = high - low
        df['doji'] = (body < range_ * 0.1).astype(int)
        
        # ================== SUPPORT/RESISTANCE ==================
        df['high_5'] = high.rolling(5).max()
        df['low_5'] = low.rolling(5).min()
        df['high_20'] = high.rolling(20).max()
        df['low_20'] = low.rolling(20).min()
        df['high_50'] = high.rolling(50).max()
        df['low_50'] = low.rolling(50).min()
        
        df['dist_to_high_20'] = (df['high_20'] - close) / close
        df['dist_to_low_20'] = (close - df['low_20']) / close
        df['range_position'] = (close - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-10)
        
        # ================== WILLIAMS %R ==================
        df['williams_r'] = -100 * (high_14 - close) / (high_14 - low_14 + 1e-10)
        
        # ================== CCI ==================
        typical_price = (high + low + close) / 3
        df['cci'] = (typical_price - typical_price.rolling(20).mean()) / (0.015 * typical_price.rolling(20).std() + 1e-10)
        
        # ================== ADX ==================
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)
        
        tr_14 = df['true_range'].rolling(14).sum()
        plus_di = 100 * (plus_dm.rolling(14).sum() / (tr_14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).sum() / (tr_14 + 1e-10))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['di_cross'] = (plus_di > minus_di).astype(int)
        
        # ================== GAPS ==================
        df['gap'] = (open_ - close.shift(1)) / close.shift(1)
        df['gap_filled'] = ((df['gap'] > 0) & (low <= close.shift(1))).astype(int)
        
        # ================== TREND ==================
        df['above_sma_20'] = (close > df['sma_20']).astype(int)
        df['above_sma_50'] = (close > df['sma_50']).astype(int)
        df['trend_strength'] = df['above_sma_20'] + df['above_sma_50'] + df['sma_5_10_cross'] + df['sma_10_20_cross']
        
        # ================== FUNDAMENTAL FEATURES ==================
        if fundamentals:
            for key, value in fundamentals.items():
                if pd.notna(value):
                    df[f'fund_{key}'] = value
                    
            # Create fundamental-derived features
            if 'fund_pe_ratio' in df.columns and 'fund_earnings_growth' in df.columns:
                # PEG-like ratio
                pe = df['fund_pe_ratio'].iloc[0] if 'fund_pe_ratio' in df.columns else np.nan
                growth = df['fund_earnings_growth'].iloc[0] if 'fund_earnings_growth' in df.columns else np.nan
                if pd.notna(pe) and pd.notna(growth) and growth > 0:
                    df['fund_peg_calculated'] = pe / (growth * 100)
            
            # Value score (low P/E + low P/B + high ROE)
            pe = fundamentals.get('pe_ratio', np.nan)
            pb = fundamentals.get('pb_ratio', np.nan)
            roe = fundamentals.get('roe', np.nan)
            if pd.notna(pe) and pd.notna(pb) and pd.notna(roe):
                # Normalize and combine (lower P/E and P/B is better, higher ROE is better)
                df['fund_value_score'] = (1/max(pe, 1) + 1/max(pb, 0.1) + roe) if pe > 0 and pb > 0 else np.nan
            
            # Quality score (high margins + low debt)
            profit_margin = fundamentals.get('profit_margin', np.nan)
            debt_equity = fundamentals.get('debt_to_equity', np.nan)
            if pd.notna(profit_margin) and pd.notna(debt_equity):
                df['fund_quality_score'] = profit_margin - (debt_equity / 100) if debt_equity > 0 else profit_margin
        
        # ================== TARGET ==================
        # IMPROVED: 10-day prediction horizon (more signal, less noise)
        # Plus momentum confirmation for higher accuracy
        
        # Future return (10 days - more predictable than 5)
        future_10d_return = close.shift(-10) / close - 1
        
        # Current momentum (helps filter)
        current_momentum = close / close.shift(20) - 1  # 20-day momentum
        
        # Target: 1 = Positive return (>1%), 0 = Negative/flat
        # Using 1% threshold to reduce noise
        df['target'] = (future_10d_return > 0.01).astype(int)
        
        # Only train on rows with clear momentum signal
        # This filters out noisy sideways markets
        has_momentum = np.abs(current_momentum) > 0.02  # >2% move in last 20 days
        df['strong_signal'] = has_momentum.astype(int)
        
        # Remove last 10 rows where we can't compute future returns
        df.loc[df.index[-10:], 'target'] = np.nan
        
        return df


def load_nifty50_stocks():
    """Load Nifty 50 stocks"""
    stock_dir = os.path.join(DATA_DIR, 'archive 2')
    all_data = []
    
    if not os.path.exists(stock_dir):
        return pd.DataFrame()
    
    print("\n📈 Loading Nifty 50 Stocks...")
    for csv_file in glob.glob(os.path.join(stock_dir, '*.csv')):
        if 'metadata' in csv_file.lower() or 'nifty50_all' in csv_file.lower():
            continue
        try:
            symbol = os.path.basename(csv_file).replace('.csv', '')
            df = pd.read_csv(csv_file)
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])
                df = df.set_index('Date')
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
            
            if len(df) > 100:
                df['Symbol'] = symbol
                df['AssetType'] = 'STOCK'
                all_data.append(df)
                print(f"  ✓ {symbol}: {len(df)} rows")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    if all_data:
        combined = pd.concat(all_data)
        print(f"\n  Total: {len(combined)} rows from {len(all_data)} stocks")
        return combined
    return pd.DataFrame()


def load_etf_data():
    """Load ETF datasets"""
    etf_dir = os.path.join(DATA_DIR, 'SIP_datasets')
    all_data = []
    
    if not os.path.exists(etf_dir):
        return pd.DataFrame()
    
    print("\n📈 Loading ETFs...")
    for csv_file in glob.glob(os.path.join(etf_dir, '*.csv')):
        try:
            symbol = os.path.basename(csv_file).replace('_data.csv', '')
            df = pd.read_csv(csv_file)
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])
                df = df.set_index('Date')
            
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if 'Volume' not in df.columns:
                df['Volume'] = 100000
            
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
            
            if len(df) > 60:
                df['Symbol'] = symbol
                df['AssetType'] = 'ETF'
                all_data.append(df)
                print(f"  ✓ {symbol}: {len(df)} rows")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    if all_data:
        combined = pd.concat(all_data)
        print(f"\n  Total: {len(combined)} rows from {len(all_data)} ETFs")
        return combined
    return pd.DataFrame()


def train_advanced_model():
    """Train advanced ensemble with fundamental data"""
    print("\n" + "="*70)
    print("🚀 ADVANCED TRADING MODEL v8.0 - WITH FUNDAMENTALS")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\n" + "="*70)
    print("📊 LOADING DATA")
    print("="*70)
    
    datasets = []
    
    stocks = load_nifty50_stocks()
    if not stocks.empty:
        datasets.append(stocks)
    
    etfs = load_etf_data()
    if not etfs.empty:
        datasets.append(etfs)
    
    if not datasets:
        raise ValueError("No data!")
    
    raw_data = pd.concat(datasets)
    print(f"\n📊 Combined: {len(raw_data)} rows")
    
    # Fetch fundamental data for all symbols
    print("\n" + "="*70)
    print("📈 FETCHING FUNDAMENTAL DATA")
    print("="*70)
    
    all_symbols = raw_data['Symbol'].unique().tolist()
    fundamentals = fetch_fundamental_data(all_symbols)
    print(f"\n📊 Got fundamentals for {len([s for s in fundamentals if fundamentals[s]])} symbols")
    
    # Feature engineering
    print("\n" + "="*70)
    print("🔧 FEATURE ENGINEERING (72+ technical + 20+ fundamental features)")
    print("="*70)
    
    processed = []
    symbols = raw_data['Symbol'].unique()
    
    for symbol in symbols:
        try:
            sym_data = raw_data[raw_data['Symbol'] == symbol].copy()
            sym_data = sym_data.sort_index()
            
            if len(sym_data) < 100:
                continue
            
            # Get fundamental data for this symbol
            sym_fundamentals = fundamentals.get(symbol, {})
            
            featured = AdvancedFeatureEngineer.calculate_all_features(sym_data, sym_fundamentals)
            # Only keep rows with valid target (drop last 10 rows where we can't compute future returns)
            featured = featured.dropna(subset=['target'])
            if len(featured) > 50:
                processed.append(featured)
                fund_count = len([k for k in featured.columns if k.startswith('fund_')])
                print(f"  ✓ {symbol}: {len(featured)} rows, {fund_count} fundamental features")
            
        except Exception as e:
            print(f"  ✗ {symbol}: {e}")
    
    print(f"\n✓ Processed {len(processed)} symbols")
    
    all_data = pd.concat(processed)
    
    # Filter to only strong signals (>2% moves)
    if 'strong_signal' in all_data.columns:
        strong_data = all_data[all_data['strong_signal'] == 1]
        print(f"📊 Strong signals only: {len(strong_data)} / {len(all_data)} ({100*len(strong_data)/len(all_data):.1f}%)")
        all_data = strong_data
    
    # Features - exclude source-specific columns that cause NaN issues
    exclude_cols = ['Symbol', 'AssetType', 'target', 'strong_signal', 'Open', 'High', 'Low', 'Close', 
                    'Volume', 'Prev Close', 'Last', 'VWAP', 'Turnover', 'Trades',
                    'Deliverable Volume', '%Deliverble', 'Series',
                    'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_5', 'ema_10', 'ema_20', 'ema_50',
                    'bb_upper', 'bb_lower', 'high_5', 'low_5', 'high_20', 'low_20', 'high_50', 'low_50',
                    'volume_sma_10', 'volume_sma_20', 'obv', 'obv_sma', 'true_range', 'atr_14',
                    # ETF-specific columns that don't exist in stock data
                    'Adj_Close', 'Percent_Change', 'Adj_Percent_Change', 'Intraday_Range_%',
                    'Gap_%', 'Year', 'Month', 'Quarter', 'Month_Name']
    
    feature_cols = [col for col in all_data.columns 
                    if col not in exclude_cols 
                    and not col.startswith('Unnamed')
                    and all_data[col].dtype in ['float64', 'int64', 'float32', 'int32', 'bool']]
    
    print(f"📊 Features: {len(feature_cols)}")
    
    # Prepare - fill NaN per symbol with median, then drop remaining NaN
    print("\n🔧 Cleaning data...")
    clean_parts = []
    for sym in all_data['Symbol'].unique():
        sym_df = all_data[all_data['Symbol'] == sym].copy()
        sym_df = sym_df.dropna(subset=['target'])
        # Fill feature NaN with median for that symbol
        for col in feature_cols:
            if col in sym_df.columns:
                med = sym_df[col].median()
                if pd.isna(med):
                    med = 0  # fallback for all-NaN columns
                sym_df[col] = sym_df[col].fillna(med)
        clean_parts.append(sym_df)
    
    clean_data = pd.concat(clean_parts)
    # Replace inf with NaN then fill
    clean_data = clean_data.replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        clean_data[col] = clean_data[col].fillna(clean_data[col].median())
    clean_data = clean_data.dropna(subset=feature_cols)
    
    X = clean_data[feature_cols].astype(float)
    y = clean_data['target']
    
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(f"📊 Samples: {len(X)}")
    print(f"📊 Target: {y.value_counts().to_dict()}")
    
    # Split
    print("\n" + "="*70)
    print("🎯 TRAINING ENSEMBLE (GPU-accelerated if available)")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=42, shuffle=False
    )
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection using RandomForest
    print("\n🔍 Feature Selection...")
    rf_sel = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_sel.fit(X_train_scaled, y_train)
    
    importances = pd.Series(rf_sel.feature_importances_, index=feature_cols)
    top_features = importances.nlargest(50).index.tolist()  # Use top 50 features
    print(f"Top {len(top_features)} features selected")
    
    X_train_sel = X_train[top_features].values
    X_test_sel = X_test[top_features].values
    X_train_sel = scaler.fit_transform(X_train_sel)
    X_test_sel = scaler.transform(X_test_sel)
    
    # Build models with GPU support if available
    print("\n🏗️ Building Ensemble...")
    models = []
    
    # Calculate class weight for imbalanced data
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
    print(f"  Class ratio: {n_neg}:{n_pos} (scale_pos_weight={scale_pos_weight:.2f})")
    
    # XGBoost with GPU (if available)
    if HAS_XGB:
        print("  🚀 Adding XGBoost (GPU if available)...")
        try:
            # Try GPU first
            xgb_model = xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                scale_pos_weight=scale_pos_weight,
                tree_method='hist',
                device='cuda',  # GPU acceleration
                random_state=42,
                eval_metric='auc'
            )
            xgb_model.fit(X_train_sel[:100], y_train[:100])
            models.append(('xgb', xgb_model))
            print("    ✓ XGBoost with GPU")
        except Exception as e:
            # Fall back to CPU
            xgb_model = xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                scale_pos_weight=scale_pos_weight,
                tree_method='hist',
                n_jobs=-1,
                random_state=42,
                eval_metric='auc'
            )
            models.append(('xgb', xgb_model))
            print(f"    ✓ XGBoost CPU (GPU not available)")
    
    # LightGBM with GPU (if available)
    if HAS_LGB:
        print("  🚀 Adding LightGBM (GPU if available)...")
        try:
            # Try GPU first
            lgb_model = lgb.LGBMClassifier(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1,
                class_weight='balanced',
                device='gpu',  # GPU acceleration
                random_state=42,
                verbose=-1
            )
            lgb_model.fit(X_train_sel[:100], y_train[:100])  # Quick test
            models.append(('lgb', lgb_model))
            print("    ✓ LightGBM with GPU")
        except Exception as e:
            # Fall back to CPU
            lgb_model = lgb.LGBMClassifier(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )
            models.append(('lgb', lgb_model))
            print(f"    ✓ LightGBM CPU (GPU not available)")
    
    # HistGradientBoosting (fast CPU implementation)
    print("  🚀 Adding HistGradientBoosting...")
    hgb = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_depth=8,
        min_samples_leaf=10, l2_regularization=0.1, random_state=42
    )
    models.append(('hgb', hgb))
    
    # RandomForest with all cores
    print("  🚀 Adding RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    models.append(('rf', rf))
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=models,
        voting='soft', n_jobs=-1
    )
    
    print(f"\n🔥 Training {len(models)} models... (this may take a few minutes)")
    import time
    start_time = time.time()
    ensemble.fit(X_train_sel, y_train)
    train_time = time.time() - start_time
    print(f"✓ Training completed in {train_time:.1f} seconds")
    
    # Evaluate
    print("\n" + "="*70)
    print("📈 EVALUATION")
    print("="*70)
    
    y_pred = ensemble.predict(X_test_sel)
    y_proba = ensemble.predict_proba(X_test_sel)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n🎯 ACCURACY (all predictions): {accuracy*100:.2f}%")
    print(f"🎯 AUC-ROC:  {auc:.4f}")
    
    # HIGH CONFIDENCE PREDICTIONS ONLY
    # Only count predictions where model is >60% confident
    high_conf_mask = (y_proba > 0.60) | (y_proba < 0.40)
    if high_conf_mask.sum() > 0:
        y_test_hc = y_test.values[high_conf_mask]
        y_pred_hc = y_pred[high_conf_mask]
        accuracy_hc = accuracy_score(y_test_hc, y_pred_hc)
        pct_high_conf = high_conf_mask.sum() / len(y_test) * 100
        print(f"\n🎯 HIGH-CONFIDENCE ACCURACY: {accuracy_hc*100:.2f}% (on {pct_high_conf:.1f}% of predictions)")
    
    # Even higher confidence
    very_high_conf_mask = (y_proba > 0.65) | (y_proba < 0.35)
    if very_high_conf_mask.sum() > 100:
        y_test_vhc = y_test.values[very_high_conf_mask]
        y_pred_vhc = y_pred[very_high_conf_mask]
        accuracy_vhc = accuracy_score(y_test_vhc, y_pred_vhc)
        pct_very_high_conf = very_high_conf_mask.sum() / len(y_test) * 100
        print(f"🎯 VERY HIGH-CONF ACCURACY: {accuracy_vhc*100:.2f}% (on {pct_very_high_conf:.1f}% of predictions)")
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Sell/Hold', 'Buy']))
    
    # Individual models
    print("\n📊 Individual Models:")
    for name, model in ensemble.named_estimators_.items():
        pred = model.predict(X_test_sel)
        acc = accuracy_score(y_test, pred)
        print(f"  {name}: {acc*100:.2f}%")
    
    # Save
    print("\n" + "="*70)
    print("💾 SAVING")
    print("="*70)
    
    model_data = {
        'model': ensemble,
        'scaler': scaler,
        'selector': None,
        'selected_features': top_features,
        'features': feature_cols,
        'metadata': {
            'algorithm': 'VotingEnsemble (HGB+GB+RF+AdaBoost)',
            'version': '6.0',
            'trained_at': datetime.now().isoformat(),
            'test_accuracy': accuracy,
            'test_auc': auc,
            'n_features': len(top_features),
            'training_samples': len(X_train),
            'target': '5-day return >1%',
            'symbols_trained': len(processed)
        }
    }
    
    joblib.dump(model_data, MODEL_FILE)
    print(f"✅ Saved: {MODEL_FILE}")
    
    print("\n" + "="*70)
    print(f"✅ COMPLETE! ACCURACY: {accuracy*100:.2f}%")
    print("="*70)
    
    return model_data


if __name__ == "__main__":
    train_advanced_model()
