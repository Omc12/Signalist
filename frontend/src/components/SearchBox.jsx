import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, X, TrendingUp } from 'lucide-react';

// Use environment variable for API URL
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * SearchBox - Modern autocomplete search with animations
 */
const SearchBox = ({ onSelectStock, placeholder = "Search stocks..." }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const inputRef = useRef(null);
  const dropdownRef = useRef(null);

  // Click outside handler
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (
        dropdownRef.current && 
        !dropdownRef.current.contains(e.target) &&
        inputRef.current && 
        !inputRef.current.contains(e.target)
      ) {
        setShowDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Debounced search with dynamic API lookup
  useEffect(() => {
    if (!query || query.length < 1) {
      setResults([]);
      setShowDropdown(false);
      return;
    }

    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        // Use the dynamic search endpoint with lower limit for focused results
        const response = await fetch(
          `${API_BASE_URL}/stocks?search=${encodeURIComponent(query)}&limit=10`
        );
        
        if (!response.ok) {
          throw new Error('Search failed');
        }
        
        const data = await response.json();
        const stockResults = data.stocks || [];
        
        // Show top 8 most relevant results only
        setResults(stockResults.slice(0, 8));
        setShowDropdown(stockResults.length > 0);
        setSelectedIndex(-1);
      } catch (error) {
        console.error('Search error:', error);
        setResults([]);
        setShowDropdown(false);
      } finally {
        setLoading(false);
      }
    }, 250); // Reduced delay for faster response

    return () => clearTimeout(timer);
  }, [query]);

  const handleSelect = (stock) => {
    setQuery('');
    setShowDropdown(false);
    setResults([]);
    
    // Normalize stock data structure
    const normalizedStock = {
      name: stock.name,
      ticker: stock.symbol || stock.ticker,
      symbol: stock.symbol || stock.ticker,
      sector: stock.sector,
      exchange: stock.exchange,
      current_price: stock.current_price || stock.price,
      ...stock // Keep all other properties
    };
    
    onSelectStock(normalizedStock);
  };

  const handleKeyDown = (e) => {
    if (!showDropdown || results.length === 0) return;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex((prev) => 
          prev < results.length - 1 ? prev + 1 : 0
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex((prev) => 
          prev > 0 ? prev - 1 : results.length - 1
        );
        break;
      case 'Enter':
        e.preventDefault();
        if (selectedIndex >= 0) {
          handleSelect(results[selectedIndex]);
        }
        break;
      case 'Escape':
        setShowDropdown(false);
        setSelectedIndex(-1);
        break;
      default:
        break;
    }
  };

  const clearSearch = () => {
    setQuery('');
    setResults([]);
    setShowDropdown(false);
    inputRef.current?.focus();
  };

  return (
    <div className="search-box">
      <div className="search-input-wrapper">
        <Search size={18} className="search-icon" />
        <input
          ref={inputRef}
          type="text"
          className="search-input"
          placeholder={placeholder}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => results.length > 0 && setShowDropdown(true)}
        />
        {loading && <div className="search-spinner" />}
        {!loading && query && (
          <motion.button 
            className="search-clear" 
            onClick={clearSearch}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <X size={16} />
          </motion.button>
        )}
      </div>

      <AnimatePresence>
        {showDropdown && (
          <motion.div 
            className="search-dropdown" 
            ref={dropdownRef}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
          >
          {loading ? (
            <div className="search-loading">
              <div className="loading-spinner"></div>
              <span>Searching across NSE & BSE...</span>
            </div>
          ) : results.length > 0 ? (
            results.map((stock, index) => (
              <motion.div
                key={`${stock.symbol || stock.ticker}-${index}`}
                className={`search-result-item ${index === selectedIndex ? 'selected' : ''}`}
                onClick={() => handleSelect(stock)}
                onMouseEnter={() => setSelectedIndex(index)}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                whileHover={{ backgroundColor: 'rgba(99, 102, 241, 0.1)' }}
              >
                <div className="stock-result-info">
                  <span className="stock-result-name">{stock.name}</span>
                  <span className="stock-result-ticker">
                    {stock.symbol || stock.ticker}
                    {stock.exchange && (
                      <span className="exchange-tag">({stock.exchange})</span>
                    )}
                  </span>
                </div>
                <div className="stock-result-details">
                  {stock.sector && (
                    <span className="stock-result-sector">{stock.sector}</span>
                  )}
                  {stock.current_price && (
                    <span className="stock-result-price">₹{stock.current_price}</span>
                  )}
                </div>
              </motion.div>
            ))
          ) : query.length >= 1 ? (
            <div className="search-empty">
              <div>No stocks found for "{query}"</div>
              <div className="search-tip">Try searching by company name or stock symbol</div>
            </div>
          ) : null}
        </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default SearchBox;
