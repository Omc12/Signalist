const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Health check
export const checkHealth = async () => {
  const response = await fetch(`${API_BASE_URL}/health`);
  return response.json();
};

// Get cache statistics
export const getCacheInfo = async () => {
  const response = await fetch(`${API_BASE_URL}/health/cache`);
  return response.json();
};

// Get all stocks
export const fetchStocks = async () => {
  const response = await fetch(`${API_BASE_URL}/stocks`);
  if (!response.ok) throw new Error('Failed to fetch stocks');
  return response.json();
};

// Search stocks with autocomplete
export const searchStocks = async (query) => {
  const response = await fetch(
    `${API_BASE_URL}/stocks/search?query=${encodeURIComponent(query)}`
  );
  if (!response.ok) throw new Error('Search failed');
  return response.json();
};

// Get detailed stock information
export const getStockDetails = async (ticker) => {
  const response = await fetch(
    `${API_BASE_URL}/stocks/details?ticker=${ticker}`
  );
  if (!response.ok) throw new Error('Failed to fetch stock details');
  return response.json();
};

// Get stock candles/OHLCV data
export const getStockCandles = async (ticker, period = '1mo') => {
  const response = await fetch(
    `${API_BASE_URL}/stocks/candles?ticker=${ticker}&period=${period}`
  );
  if (!response.ok) throw new Error('Failed to fetch candles');
  return response.json();
};

// Get ML prediction
export const predictStock = async (ticker) => {
  const response = await fetch(
    `${API_BASE_URL}/predict?ticker=${ticker}`
  );
  if (!response.ok) throw new Error('Prediction failed');
  return response.json();
};
