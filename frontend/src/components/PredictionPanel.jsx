import { useState } from 'react';
import { motion } from 'framer-motion';
import { Cpu, TrendingUp, TrendingDown, Pause, AlertTriangle, Target, Shield } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * PredictionPanel - Groww-style ML prediction display
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
      const response = await fetch(`${API_BASE_URL}/predict?ticker=${ticker}&owns_stock=${ownsStock}`);
      if (!response.ok) throw new Error('Prediction failed');
      const data = await response.json();
      
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

  const getSignalIcon = (signal) => {
    if (signal === 'BUY' || signal === 'UP' || signal === 'STRONG BUY') return <TrendingUp size={24} />;
    if (signal === 'SELL' || signal === 'DOWN' || signal === 'STRONG SELL') return <TrendingDown size={24} />;
    return <Pause size={24} />;
  };

  if (!ticker) {
    return (
      <motion.div 
        className="prediction-card"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <div className="prediction-body">
          <div className="empty-state">
            <Cpu size={48} style={{ color: 'var(--groww-purple)', marginBottom: 16 }} />
            <p className="empty-state-text">
              Select a stock to get AI-powered predictions
            </p>
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div 
      className="prediction-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
    >
      <div className="prediction-header">
        <h3 className="prediction-title">
          <Cpu size={20} />
          AI Price Prediction
        </h3>
        <motion.button
          className="predict-btn"
          onClick={handlePredict}
          disabled={loading}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {loading ? (
            <>
              <span className="loading-spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
              Analyzing...
            </>
          ) : (
            <>
              <Cpu size={16} />
              Get Prediction
            </>
          )}
        </motion.button>
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
              {ownsStock ? "I own this stock" : "I don't own this stock"}
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
          <div className="prediction-disclaimer" style={{ borderLeftColor: 'var(--groww-red)' }}>
            <AlertTriangle size={16} />
            <span>{error}</span>
          </div>
        )}

        {!prediction && !error && !loading && (
          <div className="empty-state" style={{ padding: '40px 20px' }}>
            <Cpu size={48} style={{ color: 'var(--groww-purple)', marginBottom: 16 }} />
            <p style={{ color: 'var(--text-secondary)' }}>
              Click "Get Prediction" to analyze <strong>{stockName || ticker}</strong>
            </p>
          </div>
        )}

        {prediction && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            <div className="prediction-main-result">
              <div className="prediction-signal">
                <span className="signal-label">AI Signal</span>
                <motion.div 
                  className={`signal-badge ${getSignalClass(prediction.signal)}`}
                  initial={{ scale: 0.8 }}
                  animate={{ scale: 1 }}
                  transition={{ type: 'spring', stiffness: 200 }}
                >
                  {getSignalIcon(prediction.signal)} {prediction.signal}
                </motion.div>
              </div>

              <div className="prediction-probability">
                <span className="probability-label">Confidence Score</span>
                <div className="probability-value">
                  {prediction.displayConfidence || 'N/A'}
                </div>
                <div className="probability-bar">
                  <motion.div 
                    className="probability-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${prediction.probability * 100}%` }}
                    transition={{ duration: 0.8 }}
                  />
                </div>
              </div>
            </div>

            <div className="prediction-details">
              {prediction.action && (
                <div className="prediction-detail-card">
                  <span className="detail-label">Recommended Action</span>
                  <span className="detail-value">{prediction.action}</span>
                </div>
              )}
              
              {prediction.reason && (
                <div className="prediction-detail-card">
                  <span className="detail-label">Analysis Reason</span>
                  <span className="detail-value">{prediction.reason}</span>
                </div>
              )}

              {prediction.confidence && (
                <div className="prediction-detail-card">
                  <span className="detail-label">Confidence Level</span>
                  <span className="detail-value">{prediction.confidence}</span>
                </div>
              )}

              {prediction.algorithm && (
                <div className="prediction-detail-card">
                  <span className="detail-label">Algorithm</span>
                  <span className="detail-value">{prediction.algorithm}</span>
                </div>
              )}
            </div>

            {/* Trading Parameters */}
            {prediction.trading_params && (
              <div className="stats-grid" style={{ marginTop: 16 }}>
                <div className="stat-card">
                  <div className="stat-label">
                    <Target size={14} style={{ marginRight: 4 }} />
                    Target Profit
                  </div>
                  <div className="stat-value green">
                    {prediction.trading_params.target_profit}
                  </div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">
                    <Shield size={14} style={{ marginRight: 4 }} />
                    Stop Loss
                  </div>
                  <div className="stat-value red">
                    {prediction.trading_params.stop_loss}
                  </div>
                </div>
              </div>
            )}

            <div className="prediction-disclaimer">
              <AlertTriangle size={16} className="prediction-disclaimer-icon" />
              <span>
                For educational purposes only. This is not financial advice. 
                Always do your own research before investing.
              </span>
            </div>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
};

export default PredictionPanel;
