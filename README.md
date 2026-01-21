# 📈 Signalist

AI-powered stock insights for Indian (NSE) stocks. Features machine learning predictions with 99 technical + fundamental features, real-time data from Yahoo Finance, and a modern React frontend.

## ✨ Features

- **ML Predictions**: Ensemble model (XGBoost + LightGBM + HistGradientBoosting + RandomForest)
- **99 Features**: 72 technical indicators + 27 fundamental metrics (P/E, P/B, ROE, etc.)
- **235K+ Training Samples**: Trained on Nifty 50 stocks + ETFs
- **Real-time Data**: Live prices from Yahoo Finance
- **Modern UI**: Dark Groww-inspired theme with interactive charts

## 🏗️ Architecture

```
├── backend/                 # FastAPI Python backend
│   ├── core/               # Configuration
│   ├── routes/             # API endpoints
│   ├── services/           # Business logic
│   ├── models/             # Trained ML models
│   ├── data/               # Training datasets
│   └── train_model.py      # Model training script
├── frontend/               # React + Vite frontend
│   └── src/
│       ├── components/     # React components
│       └── App.jsx         # Main application
└── start.sh               # Start script
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+

### 1. Start Everything

```bash
chmod +x start.sh
./start.sh
```

### 2. Manual Setup

**Backend:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### 3. Access
- Frontend: http://localhost:5173
- API Docs: http://localhost:8000/docs

## 🤖 Model Training

Retrain the model with new data:

```bash
cd backend
python train_model.py
```

### Training Features

| Category | Features |
|----------|----------|
| Price | Returns, log returns, volatility, momentum |
| Volume | Volume ratios, VWAP signals, OBV |
| Technical | RSI, MACD, Bollinger Bands, ADX, Stochastic |
| Fundamental | P/E, P/B, ROE, ROA, debt ratios, margins |
| Time | Day of week, month, quarter effects |

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Training Samples | 181,874 |
| Features | 99 |
| Overall Accuracy | 52.4% |
| High-Confidence Accuracy | 55.6% |
| Stocks | 49 Nifty 50 + 3 ETFs |

> **Note**: 52-55% accuracy on stock direction is realistic and consistent with academic research. The edge comes from taking only high-confidence predictions with proper risk management.

## 🔧 Configuration

Copy `.env.example` to `.env` and configure:

```bash
cd backend
cp .env.example .env
```

Environment variables:
- `API_HOST`: Server host (default: 0.0.0.0)
- `API_PORT`: Server port (default: 8000)
- `FRONTEND_URL`: Frontend URL for CORS

## 📁 Data Sources

Training data located in `backend/data/`:
- `archive 2/`: Nifty 50 historical data (~235K rows)
- `SIP_datasets/`: ETF data (NIFTYBEES, GOLDBEES, BANKBEES)

## 🛠️ Tech Stack

**Backend:** FastAPI, scikit-learn, XGBoost, LightGBM, yfinance, pandas

**Frontend:** React 18, Vite, Recharts, Axios

## 📄 License

MIT License
