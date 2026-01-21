import { useState } from 'react';

/**
 * PredictionPanel - Premium dark mode ML prediction display
 */
const PredictionPanel = ({ ticker, stockName }) => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [ownsStock, setOwnsStock] = useState(false);

  const handlePredict = async () => {
    if (!ticker) {
      setError('Please select a stock first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`http://localhost:8000/predict?ticker=${ticker}&owns_stock=${ownsStock}`);
      if (!response.ok) throw new Error('Prediction failed');
      const data = await response.json();
      
      // Handle advanced model response
      const signal = data.signal || data.predicted_direction || 'WAIT';
      const probability = data.probability || data.probability_up || 0;
      const confidence = data.confidence || 'LOW';
      
      setPrediction({
        ...data,
        signal: signal,
        probability: probability,
        confidence: confidence,
        displayConfidence: `${(probability * 100).toFixed(1)}%`
      });
    } catch (err) {
      setError(err.message);
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  const getSignalClass = (signal) => {
    if (signal === 'BUY' || signal === 'UP' || signal === 'STRONG BUY') return 'up';
    if (signal === 'SELL' || signal === 'DOWN' || signal === 'STRONG SELL') return 'down';
    return 'hold';
  };

  const getSignalEmoji = (signal) => {
    if (signal === 'BUY' || signal === 'UP' || signal === 'STRONG BUY') return '🚀';
    if (signal === 'SELL' || signal === 'DOWN' || signal === 'STRONG SELL') return '📉';
    return '⏸️';
  };

  if (!ticker) {
    return (
      <div className="prediction-card">
        <div className="prediction-body">
          <div className="empty-state">
            <span className="empty-state-icon">🤖</span>
            <p className="empty-state-text">
              Select a stock to get AI-powered predictions
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="prediction-card">
      <div className="prediction-header">
        <h3 className="prediction-title">AI Prediction</h3>
        <button
          className="predict-btn"
          onClick={handlePredict}
          disabled={loading}
        >
          {loading ? (
            <>
              <span className="loading-spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
              Analyzing...
            </>
          ) : (
            <>
              🧠 Get Prediction
            </>
          )}
        </button>
      </div>

      {/* Ownership Toggle */}
      <div className="ownership-toggle-container">
        <div className="ownership-toggle">
          <input 
            type="checkbox"
            id="owns-stock-toggle"
            checked={ownsStock}
            onChange={(e) => setOwnsStock(e.target.checked)}
          />
          <label htmlFor="owns-stock-toggle" className="toggle-label">
            <span className="toggle-switch">
              <span className="toggle-slider"></span>
            </span>
            <span className="toggle-text">
              {ownsStock ? "📈 I own this stock" : "🛒 I don't own this stock"}
            </span>
          </label>
        </div>
        <div className="ownership-explanation">
          {ownsStock 
            ? "Get HOLD/SELL recommendations for your existing position" 
            : "Get BUY/WAIT signals for potential new positions"
          }
        </div>
      </div>

      <div className="prediction-body">
        {error && (
          <div className="prediction-disclaimer" style={{ borderLeftColor: 'var(--accent-red)' }}>
            <span className="prediction-disclaimer-icon">❌</span>
            <span>{error}</span>
          </div>
        )}

        {!prediction && !error && !loading && (
          <div className="empty-state" style={{ padding: '40px 20px' }}>
            <span className="empty-state-icon">📊</span>
            <p className="empty-state-text">
              Click "Get Prediction" to analyze {stockName || ticker}
            </p>
          </div>
        )}

        {prediction && (
          <div>
            <div className="prediction-main-result">
              <div className="prediction-signal">
                <span className="signal-label">AI Signal</span>
                <div className={`signal-badge ${getSignalClass(prediction.signal)}`}>
                  {getSignalEmoji(prediction.signal)} {prediction.signal}
                </div>
              </div>

              <div className="prediction-probability">
                <span className="probability-label">Probability</span>
                <div className="probability-value">
                  {prediction.displayConfidence || 'N/A'}
                </div>
              </div>

              <div className="prediction-confidence">
                <span className="signal-label">Confidence</span>
                <div className={`confidence-badge ${prediction.confidence?.toLowerCase()}`}>
                  {prediction.confidence}
                </div>
              </div>
            </div>

            {/* VIX Status */}
            {prediction.vix && (
              <div className="prediction-metrics">
                <div className="metric-item">
                  <span className="metric-label">VIX Status</span>
                  <span className={`metric-value ${prediction.vix.status === 'SAFE' ? 'text-green' : 'text-red'}`}>
                    {prediction.vix.status} ({prediction.vix.current})
                  </span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Algorithm</span>
                  <span className="metric-value" style={{fontSize: '11px'}}>
                    {prediction.algorithm || 'ML Model'}
                  </span>
                </div>
              </div>
            )}

            {/* Backtest Performance */}
            {prediction.backtest && prediction.backtest.total_signals > 0 && (
              <div className="prediction-metrics">
                <div className="metric-item">
                  <span className="metric-label">Win Rate</span>
                  <span className="metric-value text-green">
                    {(prediction.backtest.win_rate * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Avg Return</span>
                  <span className={`metric-value ${prediction.backtest.avg_return > 0 ? 'text-green' : 'text-red'}`}>
                    {(prediction.backtest.avg_return * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Total Signals</span>
                  <span className="metric-value">
                    {prediction.backtest.total_signals}
                  </span>
                </div>
              </div>
            )}

            {/* Technical Indicators */}
            {prediction.technicals && (
              <div className="prediction-metrics">
                <div className="metric-item">
                  <span className="metric-label">RSI</span>
                  <span className={`metric-value ${prediction.technicals.rsi < 30 ? 'text-green' : prediction.technicals.rsi > 70 ? 'text-red' : ''}`}>
                    {prediction.technicals.rsi}
                  </span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">MACD</span>
                  <span className={`metric-value ${prediction.technicals.macd > 0 ? 'text-green' : 'text-red'}`}>
                    {prediction.technicals.macd.toFixed(2)}
                  </span>
                </div>
              </div>
            )}

            {/* Trading Parameters */}
            {prediction.trading_params && (
              <div className="prediction-metrics" style={{borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '12px', marginTop: '12px'}}>
                <div className="metric-item">
                  <span className="metric-label">Target Profit</span>
                  <span className="metric-value text-green">
                    {prediction.trading_params.target_profit}
                  </span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Stop Loss</span>
                  <span className="metric-value text-red">
                    {prediction.trading_params.stop_loss}
                  </span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Horizon</span>
                  <span className="metric-value">
                    {prediction.trading_params.horizon_days} days
                  </span>
                </div>
              </div>
            )}

            <div className="prediction-disclaimer">
              <span className="prediction-disclaimer-icon">⚠️</span>
              <span>
                For educational purposes only. This is not financial advice. 
                Always do your own research before investing.
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionPanel;
