import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, DollarSign, BarChart2, Calendar, Activity, Percent, Target, Building2, Users, Globe, Phone, MapPin, TrendingUp as Growth, Shield, Zap } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * StockOverview - Groww-style comprehensive overview with stock fundamentals
 */
const StockOverview = ({ ticker, details: passedDetails }) => {
  const [details, setDetails] = useState(passedDetails || null);
  const [loading, setLoading] = useState(!passedDetails);

  useEffect(() => {
    if (passedDetails) {
      setDetails(passedDetails);
      setLoading(false);
      return;
    }
    
    if (!ticker) return;

    const fetchDetails = async () => {
      setLoading(true);
      try {
        const response = await fetch(
          `${API_BASE_URL}/stocks/details?ticker=${ticker}`
        );
        if (!response.ok) throw new Error('Failed to fetch');
        const data = await response.json();
        setDetails(data);
      } catch (error) {
        console.error('Error fetching details:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchDetails();
  }, [ticker, passedDetails]);

  const formatNumber = (num, type = 'number', fallback = '—') => {
    if (num === null || num === undefined) return fallback;
    
    if (type === 'currency') {
      if (num >= 1e12) return `₹${(num / 1e12).toFixed(2)}T`;
      if (num >= 1e9) return `₹${(num / 1e9).toFixed(2)}B`;
      if (num >= 1e7) return `₹${(num / 1e7).toFixed(2)}Cr`;
      if (num >= 1e5) return `₹${(num / 1e5).toFixed(2)}L`;
      return `₹${num.toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;
    }
    
    if (type === 'price') {
      return `₹${num.toLocaleString('en-IN', { 
        minimumFractionDigits: 2, 
        maximumFractionDigits: 2 
      })}`;
    }
    
    if (type === 'percent') {
      return `${(num * 100).toFixed(2)}%`;
    }
    
    // For fields that yfinance returns as already-percentage (dividendYield, fiveYearAvgDividendYield)
    if (type === 'percentRaw') {
      return `${num.toFixed(2)}%`;
    }
    
    // For Unix timestamps -> readable date
    if (type === 'date') {
      if (typeof num === 'number') {
        const date = new Date(num * 1000);
        return date.toLocaleDateString('en-IN', { 
          year: 'numeric', 
          month: 'short', 
          day: 'numeric' 
        });
      }
      return num;
    }
    
    if (type === 'volume') {
      if (num >= 1e7) return `${(num / 1e7).toFixed(2)}Cr`;
      if (num >= 1e5) return `${(num / 1e5).toFixed(2)}L`;
      if (num >= 1000) return `${(num / 1000).toFixed(2)}K`;
      return num.toLocaleString('en-IN');
    }
    
    if (type === 'shares') {
      if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
      if (num >= 1e7) return `${(num / 1e7).toFixed(2)}Cr`;
      if (num >= 1e5) return `${(num / 1e5).toFixed(2)}L`;
      return num.toLocaleString('en-IN');
    }
    
    return num.toLocaleString('en-IN', { maximumFractionDigits: 2 });
  };

  if (loading) {
    return (
      <div className="overview-grid">
        {[...Array(9)].map((_, i) => (
          <motion.div 
            key={i} 
            className="overview-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
          >
            <div className="skeleton" style={{ width: 120, height: 16, marginBottom: 16 }} />
            {[...Array(3)].map((_, j) => (
              <div key={j} style={{ marginBottom: 12 }}>
                <div className="skeleton" style={{ width: 80, height: 12, marginBottom: 6 }} />
                <div className="skeleton" style={{ width: '100%', height: 20 }} />
              </div>
            ))}
          </motion.div>
        ))}
      </div>
    );
  }

  if (!details) {
    return (
      <motion.div 
        className="overview-card" 
        style={{ textAlign: 'center', padding: 48, gridColumn: '1 / -1' }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <Activity size={48} style={{ color: 'var(--text-muted)', marginBottom: 16 }} />
        <p style={{ color: 'var(--text-muted)' }}>Unable to load stock details</p>
      </motion.div>
    );
  }

  // Check if this is an ETF
  const isETF = details.quoteType === 'ETF' || 
                details.longName?.includes('ETF') || 
                details.shortName?.includes('ETF');

  const sections = [
    { 
      title: 'Market Data',
      icon: <DollarSign size={16} />,
      items: [
        { label: 'Market Cap', value: formatNumber(details.marketCap, 'currency', isETF ? 'N/A (ETF)' : '—') },
        { label: 'Enterprise Value', value: formatNumber(details.enterpriseValue, 'currency', isETF ? 'N/A (ETF)' : '—') },
        { label: 'Volume', value: formatNumber(details.volume, 'volume') },
        { label: 'Avg Volume (10D)', value: formatNumber(details.averageVolume10days, 'volume') },
        { label: 'Avg Volume (3M)', value: formatNumber(details.averageDailyVolume3Month, 'volume') },
        { label: 'Shares Outstanding', value: formatNumber(details.sharesOutstanding, 'shares', isETF ? 'N/A (ETF)' : '—') },
        { label: 'Float Shares', value: formatNumber(details.floatShares, 'shares', isETF ? 'N/A (ETF)' : '—') },
      ]
    },
    { 
      title: 'Valuation Ratios',
      icon: <BarChart2 size={16} />,
      items: [
        { label: 'P/E Ratio (TTM)', value: formatNumber(details.trailingPE) },
        { label: 'Forward P/E', value: formatNumber(details.forwardPE) },
        { label: 'P/B Ratio', value: formatNumber(details.priceToBook) },
        { label: 'P/S Ratio', value: formatNumber(details.priceToSalesTrailing12Months) },
        { label: 'PEG Ratio', value: formatNumber(details.pegRatio) },
        { label: 'Beta', value: formatNumber(details.beta) },
        { label: 'EV/Revenue', value: formatNumber(details.enterpriseToRevenue) },
        { label: 'EV/EBITDA', value: formatNumber(details.enterpriseToEbitda) },
      ]
    },
    { 
      title: 'Price Range & Moving Averages',
      icon: <Calendar size={16} />,
      items: [
        { label: 'Day High', value: formatNumber(details.dayHigh, 'price') },
        { label: 'Day Low', value: formatNumber(details.dayLow, 'price') },
        { label: '52W High', value: formatNumber(details.fiftyTwoWeekHigh, 'price') },
        { label: '52W Low', value: formatNumber(details.fiftyTwoWeekLow, 'price') },
        { label: '50-Day MA', value: formatNumber(details.fiftyDayAverage, 'price') },
        { label: '200-Day MA', value: formatNumber(details.twoHundredDayAverage, 'price') },
        { label: 'Open', value: formatNumber(details.open, 'price') },
        { label: 'Previous Close', value: formatNumber(details.previousClose, 'price') },
      ]
    },
    { 
      title: 'Earnings & EPS',
      icon: <TrendingUp size={16} />,
      items: [
        { label: 'EPS (TTM)', value: formatNumber(details.trailingEps, 'price') },
        { label: 'Forward EPS', value: formatNumber(details.forwardEps, 'price') },
        { label: 'EPS Current Year', value: formatNumber(details.epsCurrentYear, 'price') },
        { label: 'Revenue Per Share', value: formatNumber(details.revenuePerShare, 'price') },
        { label: 'Book Value', value: formatNumber(details.bookValue, 'price') },
        { label: 'Earnings Growth', value: formatNumber(details.earningsGrowth, 'percent') },
        { label: 'Quarterly Earnings Growth', value: formatNumber(details.earningsQuarterlyGrowth, 'percent') },
        { label: 'Revenue Growth', value: formatNumber(details.revenueGrowth, 'percent') },
      ]
    },
    { 
      title: 'Dividends',
      icon: <Percent size={16} />,
      items: [
        { label: 'Dividend Rate', value: formatNumber(details.dividendRate, 'price') },
        { label: 'Dividend Yield', value: formatNumber(details.dividendYield, 'percentRaw') },
        { label: '5Y Avg Dividend Yield', value: formatNumber(details.fiveYearAvgDividendYield, 'percentRaw') },
        { label: 'Last Dividend', value: formatNumber(details.lastDividendValue, 'price') },
        { label: 'Payout Ratio', value: formatNumber(details.payoutRatio, 'percent') },
        { label: 'Ex-Dividend Date', value: formatNumber(details.exDividendDate, 'date') },
      ]
    },
    { 
      title: 'Profitability Metrics',
      icon: <Growth size={16} />,
      items: [
        { label: 'Profit Margins', value: formatNumber(details.profitMargins, 'percent') },
        { label: 'Operating Margins', value: formatNumber(details.operatingMargins, 'percent') },
        { label: 'Gross Margins', value: formatNumber(details.grossMargins, 'percent') },
        { label: 'EBITDA Margins', value: formatNumber(details.ebitdaMargins, 'percent') },
        { label: 'ROA', value: formatNumber(details.returnOnAssets, 'percent') },
        { label: 'ROE', value: formatNumber(details.returnOnEquity, 'percent') },
      ]
    },
    { 
      title: 'Financial Health',
      icon: <Shield size={16} />,
      items: [
        { label: 'Total Revenue', value: formatNumber(details.totalRevenue, 'currency') },
        { label: 'EBITDA', value: formatNumber(details.ebitda, 'currency') },
        { label: 'Gross Profits', value: formatNumber(details.grossProfits, 'currency') },
        { label: 'Net Income', value: formatNumber(details.netIncomeToCommon, 'currency') },
        { label: 'Total Cash', value: formatNumber(details.totalCash, 'currency') },
        { label: 'Total Debt', value: formatNumber(details.totalDebt, 'currency') },
        { label: 'Debt to Equity', value: formatNumber(details.debtToEquity) },
        { label: 'Current Ratio', value: formatNumber(details.currentRatio) },
        { label: 'Quick Ratio', value: formatNumber(details.quickRatio) },
      ]
    },
    { 
      title: 'Cash Flow',
      icon: <Zap size={16} />,
      items: [
        { label: 'Operating Cash Flow', value: formatNumber(details.operatingCashflow, 'currency') },
        { label: 'Free Cash Flow', value: formatNumber(details.freeCashflow, 'currency') },
        { label: 'Total Cash Per Share', value: formatNumber(details.totalCashPerShare, 'price') },
      ]
    },
    { 
      title: 'Analyst Recommendations',
      icon: <Target size={16} />,
      items: [
        { label: 'Recommendation', value: details.recommendationKey ? details.recommendationKey.toUpperCase() : '—' },
        { label: 'Recommendation Mean', value: formatNumber(details.recommendationMean) },
        { label: 'Target Mean Price', value: formatNumber(details.targetMeanPrice, 'price') },
        { label: 'Target High', value: formatNumber(details.targetHighPrice, 'price') },
        { label: 'Target Low', value: formatNumber(details.targetLowPrice, 'price') },
        { label: 'Analyst Opinions', value: formatNumber(details.numberOfAnalystOpinions) },
      ]
    },
    { 
      title: 'Ownership & Risk',
      icon: <Users size={16} />,
      items: [
        { label: 'Insiders', value: formatNumber(details.heldPercentInsiders, 'percent') },
        { label: 'Institutions', value: formatNumber(details.heldPercentInstitutions, 'percent') },
        { label: 'Overall Risk', value: formatNumber(details.overallRisk) },
        { label: 'Audit Risk', value: formatNumber(details.auditRisk) },
        { label: 'Board Risk', value: formatNumber(details.boardRisk) },
        { label: 'Compensation Risk', value: formatNumber(details.compensationRisk) },
      ]
    },
  ];

  // Company Info Section (separate, larger display)
  const companyInfo = {
    sector: details.sector,
    industry: details.industry,
    employees: details.fullTimeEmployees,
    website: details.website,
    phone: details.phone,
    address: [details.address1, details.city, details.country].filter(Boolean).join(', '),
    description: details.longBusinessSummary || details.businessSummary,
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Company Information Card */}
      {(companyInfo.description || companyInfo.sector) && (
        <motion.div 
          className="overview-card"
          style={{ gridColumn: '1 / -1' }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="overview-card-title">
            <Building2 size={16} />
            Company Information
          </div>
          
          {companyInfo.description && (
            <p style={{ 
              color: 'var(--text-secondary)', 
              lineHeight: 1.7, 
              marginBottom: 20,
              fontSize: 14
            }}>
              {companyInfo.description}
            </p>
          )}
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 16 }}>
            {companyInfo.sector && (
              <div className="overview-item">
                <span className="overview-item-label">Sector</span>
                <span className="overview-item-value">{companyInfo.sector}</span>
              </div>
            )}
            {companyInfo.industry && (
              <div className="overview-item">
                <span className="overview-item-label">Industry</span>
                <span className="overview-item-value">{companyInfo.industry}</span>
              </div>
            )}
            {companyInfo.employees && (
              <div className="overview-item">
                <span className="overview-item-label">
                  <Users size={12} style={{ marginRight: 4 }} />
                  Employees
                </span>
                <span className="overview-item-value">{formatNumber(companyInfo.employees)}</span>
              </div>
            )}
            {companyInfo.website && (
              <div className="overview-item">
                <span className="overview-item-label">
                  <Globe size={12} style={{ marginRight: 4 }} />
                  Website
                </span>
                <a 
                  href={companyInfo.website} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="overview-item-value"
                  style={{ color: 'var(--groww-purple)', textDecoration: 'none' }}
                >
                  Visit →
                </a>
              </div>
            )}
            {companyInfo.phone && (
              <div className="overview-item">
                <span className="overview-item-label">
                  <Phone size={12} style={{ marginRight: 4 }} />
                  Phone
                </span>
                <span className="overview-item-value">{companyInfo.phone}</span>
              </div>
            )}
            {companyInfo.address && (
              <div className="overview-item" style={{ gridColumn: '1 / -1' }}>
                <span className="overview-item-label">
                  <MapPin size={12} style={{ marginRight: 4 }} />
                  Address
                </span>
                <span className="overview-item-value">{companyInfo.address}</span>
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* Metrics Grid */}
      <div className="overview-grid">
        {sections.map((section, sectionIndex) => {
          // Filter out items with missing values (but keep N/A messages for ETFs)
          const validItems = section.items.filter(item => 
            item.value !== '—'
          );
          if (validItems.length === 0) return null;
          
          return (
            <motion.div 
              key={section.title} 
              className="overview-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: sectionIndex * 0.05 }}
            >
              <div className="overview-card-title">
                {section.icon}
                {section.title}
              </div>
              {validItems.map((item, index) => (
                <div key={index} className="overview-item">
                  <span className="overview-item-label">{item.label}</span>
                  <span className="overview-item-value">{item.value}</span>
                </div>
              ))}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
};

export default StockOverview;
