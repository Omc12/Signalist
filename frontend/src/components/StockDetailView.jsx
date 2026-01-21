import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, TrendingUp, TrendingDown, BarChart3, Cpu, Info } from 'lucide-react';
import PriceChart from './PriceChart';
import PredictionPanel from './PredictionPanel';
import StockOverview from './StockOverview';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * StockDetailView - Groww-inspired full stock detail page
 */
const StockDetailView = ({ stock, onBack }) => {
  const [activeTab, setActiveTab] = useState('chart');
  const [details, setDetails] = useState(null);
  const [loading, setLoading] = useState(true);

  const tabs = [
    { id: 'overview', label: 'Overview', icon: <Info size={16} /> },
    { id: 'chart', label: 'Chart', icon: <BarChart3 size={16} /> },
    { id: 'prediction', label: 'AI Prediction', icon: <Cpu size={16} /> },
  ];

  useEffect(() => {
    if (!stock?.ticker) return;

    const fetchDetails = async () => {
      setLoading(true);
      try {
        const response = await fetch(
          `${API_BASE_URL}/stocks/details?ticker=${stock.ticker}`
        );
        if (!response.ok) throw new Error('Failed to fetch');
        const data = await response.json();
        setDetails(data);
      } catch (error) {
        console.error('Error:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchDetails();
  }, [stock?.ticker]);

  const formatPrice = (price) => {
    if (!price) return '—';
    return '₹' + new Intl.NumberFormat('en-IN', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(price);
  };

  const formatMarketCap = (value) => {
    if (!value) return '—';
    if (value >= 1e12) return '₹' + (value / 1e12).toFixed(2) + 'T';
    if (value >= 1e9) return '₹' + (value / 1e9).toFixed(2) + 'B';
    if (value >= 1e7) return '₹' + (value / 1e7).toFixed(2) + 'Cr';
    return '₹' + (value / 1e5).toFixed(2) + 'L';
  };

  const price = details?.currentPrice || details?.regularMarketPrice || 0;
  const prevClose = details?.previousClose || details?.regularMarketPreviousClose || price;
  const change = price - prevClose;
  const changePercent = prevClose > 0 ? (change / prevClose) * 100 : 0;
  const isPositive = change >= 0;

  return (
    <motion.div 
      className="stock-detail-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      {/* Header Card */}
      <motion.div 
        className="stock-header-card"
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.1 }}
      >
        <div className="stock-header-left">
          <div className="stock-header-top">
            <motion.button 
              className="back-button" 
              onClick={onBack}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <ArrowLeft size={20} />
            </motion.button>
            <div className="stock-header-info">
              <h1>{details?.shortName || details?.longName || stock.name}</h1>
              <div className="stock-ticker-badge">
                <span className="ticker">{stock.ticker}</span>
                <span>•</span>
                <span>{stock.sector || details?.sector || 'Equity'}</span>
              </div>
            </div>
          </div>

          <div className="stock-price-section">
            {loading ? (
              <>
                <div className="skeleton" style={{ width: 180, height: 48, marginBottom: 8 }} />
                <div className="skeleton" style={{ width: 120, height: 24 }} />
              </>
            ) : (
              <>
                <div className="current-price">{formatPrice(price)}</div>
                <div className="price-change">
                  <span className={`price-change-value ${isPositive ? 'up' : 'down'}`}>
                    {isPositive ? <TrendingUp size={18} /> : <TrendingDown size={18} />}
                    {isPositive ? '+' : ''}{formatPrice(change)}
                  </span>
                  <span className={`price-change-percent ${isPositive ? 'up' : 'down'}`}>
                    {isPositive ? '+' : ''}{changePercent.toFixed(2)}%
                  </span>
                </div>
              </>
            )}
          </div>
        </div>

        <div className="stock-header-right">
          <div className="header-stat">
            <span className="header-stat-label">Market Cap</span>
            <span className="header-stat-value">{formatMarketCap(details?.marketCap)}</span>
          </div>
          <div className="header-stat">
            <span className="header-stat-label">52W High</span>
            <span className="header-stat-value">{formatPrice(details?.fiftyTwoWeekHigh)}</span>
          </div>
          <div className="header-stat">
            <span className="header-stat-label">52W Low</span>
            <span className="header-stat-value">{formatPrice(details?.fiftyTwoWeekLow)}</span>
          </div>
          {details?.lastUpdateTime && (
            <div className="header-stat">
              <span className="header-stat-label">Last Updated</span>
              <span className="header-stat-value" style={{ fontSize: 12 }}>
                {details.lastUpdateTime}
              </span>
            </div>
          )}
          {details?.marketState && (
            <div className="header-stat">
              <span className="header-stat-label">Market</span>
              <span 
                className="header-stat-value" 
                style={{ 
                  fontSize: 12,
                  color: details.marketState === 'REGULAR' ? 'var(--groww-green)' : 'var(--text-muted)'
                }}
              >
                {details.marketState}
              </span>
            </div>
          )}
        </div>
      </motion.div>

      {/* Tabs */}
      <motion.div 
        className="tabs-container"
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        {tabs.map((tab) => (
          <motion.button
            key={tab.id}
            className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <span className="tab-icon">{tab.icon}</span>
            {tab.label}
          </motion.button>
        ))}
      </motion.div>

      {/* Tab Content */}
      <motion.div 
        key={activeTab}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
      >
        {activeTab === 'overview' && (
          <StockOverview ticker={stock.ticker} details={details} />
        )}
        
        {activeTab === 'chart' && (
          <PriceChart ticker={stock.ticker} />
        )}
        
        {activeTab === 'prediction' && (
          <PredictionPanel 
            ticker={stock.ticker} 
            stockName={stock.name} 
          />
        )}
      </motion.div>
    </motion.div>
  );
};

export default StockDetailView;
