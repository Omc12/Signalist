import { useState, useEffect, useRef } from 'react';
import { createChart } from 'lightweight-charts';
import { motion } from 'framer-motion';
import { BarChart3, AlertCircle, RefreshCw } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * PriceChart - Groww-style professional chart with dark theme
 */
const PriceChart = ({ ticker }) => {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);
  
  const [timeframe, setTimeframe] = useState('1mo');
  const [interval, setInterval] = useState('1d');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const timeframes = [
    { label: '1D', value: '1d' },
    { label: '1W', value: '5d' },
    { label: '1M', value: '1mo' },
    { label: '3M', value: '3mo' },
    { label: '6M', value: '6mo' },
    { label: '1Y', value: '1y' },
    { label: '5Y', value: '5y' },
  ];

  // Initialize chart with Groww theme colors
  useEffect(() => {
    if (!containerRef.current) return;

    // Clear previous chart with proper cleanup
    if (chartRef.current) {
      try {
        if (seriesRef.current) {
          seriesRef.current = null;
        }
        if (!chartRef.current._disposed) {
          chartRef.current.remove();
        }
      } catch (error) {
        console.warn('Error removing previous chart:', error);
      } finally {
        chartRef.current = null;
        seriesRef.current = null;
      }
    }

    try {
      const chart = createChart(containerRef.current, {
        width: containerRef.current.clientWidth,
        height: 400,
        layout: {
          background: { type: 'solid', color: '#17171f' },
          textColor: '#808080',
          fontFamily: 'Inter, -apple-system, sans-serif',
        },
        grid: {
          vertLines: { color: 'rgba(255, 255, 255, 0.03)' },
          horzLines: { color: 'rgba(255, 255, 255, 0.03)' },
        },
        crosshair: {
          mode: 1,
          vertLine: {
            color: 'rgba(255, 255, 255, 0.4)',
            width: 1,
            style: 2,
            labelBackgroundColor: '#b8b9c0',
          },
          horzLine: {
            color: 'rgba(255, 255, 255, 0.4)',
            width: 1,
            style: 2,
            labelBackgroundColor: '#b5b7bf',
          },
        },
        rightPriceScale: {
          borderColor: 'rgba(255, 255, 255, 0.05)',
          scaleMargins: { top: 0.1, bottom: 0.1 },
        },
        timeScale: {
          borderColor: 'rgba(255, 255, 255, 0.05)',
          timeVisible: true,
          secondsVisible: false,
        },
      });

      // Use Groww green for the area series
      const areaSeries = chart.addAreaSeries({
        lineColor: '#f2f2f2',
        topColor: 'rgba(153, 153, 153, 0.35)',
        bottomColor: 'rgba(0, 208, 156, 0.02)',
        lineWidth: 2,
        priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
      });

      chartRef.current = chart;
      seriesRef.current = areaSeries;

      // Handle container resize
      const resizeHandler = () => {
        if (chart && containerRef.current) {
          chart.applyOptions({ width: containerRef.current.clientWidth });
        }
      };

      window.addEventListener('resize', resizeHandler);

      return () => {
        window.removeEventListener('resize', resizeHandler);
        if (chartRef.current) {
          try {
            if (seriesRef.current) {
              seriesRef.current = null;
            }
            if (!chartRef.current._disposed) {
              chartRef.current.remove();
            }
          } catch (error) {
            console.warn('Error removing chart on cleanup:', error);
          } finally {
            chartRef.current = null;
            seriesRef.current = null;
          }
        }
      };

    } catch (error) {
      console.error('Error initializing chart:', error);
      setError('Failed to create chart: ' + error.message);
    }
  }, []);

  // Fetch and update chart data
  const fetchChartData = async () => {
    if (!ticker || !seriesRef.current) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/stocks/candles?ticker=${ticker}&period=${timeframe}&interval=${interval}`
      );
      
      if (!response.ok) {
        throw new Error('Failed to fetch chart data');
      }
      
      const result = await response.json();
      const data = result.candles || [];
      
      if (data && data.length > 0) {
        const chartData = data
          .map((candle) => {
            let timestamp;
            const dateValue = candle.time || candle.date || candle.Date || candle.Datetime || candle.datetime || 
                             candle.timestamp || candle.Timestamp;
            
            if (typeof dateValue === 'string') {
              timestamp = new Date(dateValue).getTime();
            } else if (typeof dateValue === 'number') {
              timestamp = dateValue > 1e12 ? dateValue : dateValue * 1000;
            } else {
              return null;
            }
            
            if (isNaN(timestamp) || timestamp <= 0) {
              return null;
            }
            
            return {
              time: Math.floor(timestamp / 1000),
              value: parseFloat(candle.close || candle.Close || 0),
            };
          })
          .filter(item => item !== null)
          .sort((a, b) => a.time - b.time);
        
        if (chartData.length > 0 && seriesRef.current) {
          try {
            seriesRef.current.setData(chartData);
            if (chartRef.current) {
              chartRef.current.timeScale().fitContent();
            }
          } catch (dataError) {
            console.error('Error setting chart data:', dataError);
            setError('Failed to display chart data: ' + dataError.message);
          }
        } else {
          console.warn('No valid chart data available');
          setError('No valid chart data available');
        }
      }
    } catch (err) {
      console.error('Error fetching chart data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Fetch data when ticker, timeframe, or interval changes
  useEffect(() => {
    const timer = setTimeout(fetchChartData, 100);
    return () => clearTimeout(timer);
  }, [ticker, timeframe, interval]);

  return (
    <motion.div 
      className="chart-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
    >
      <div className="chart-header">
        <div className="chart-title">
          <BarChart3 size={18} />
          Price Chart
        </div>
        <div className="chart-controls">
          {timeframes.map((tf) => (
            <motion.button
              key={tf.value}
              className={`timeframe-btn ${timeframe === tf.value ? 'active' : ''}`}
              onClick={() => setTimeframe(tf.value)}
              disabled={loading}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {tf.label}
            </motion.button>
          ))}
        </div>
      </div>

      <div className="chart-body">
        {loading && (
          <div className="chart-loading">
            <div className="loading-spinner"></div>
            <span>Loading chart data...</span>
          </div>
        )}
        {error && (
          <div className="chart-error">
            <AlertCircle size={20} />
            <span>{error}</span>
            <motion.button 
              onClick={fetchChartData} 
              className="timeframe-btn"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <RefreshCw size={14} />
              Retry
            </motion.button>
          </div>
        )}
        <div ref={containerRef} className="chart-container" />
      </div>
    </motion.div>
  );
};

export default PriceChart;
