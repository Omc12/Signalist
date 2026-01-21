import { motion } from 'framer-motion';
import { TrendingUp, RefreshCw } from 'lucide-react';
import SearchBox from './SearchBox';

/**
 * Navbar - Groww-inspired sticky navigation with premium dark theme
 */
const Navbar = ({ onSelectStock, onRefresh, refreshing }) => {
  return (
    <motion.nav 
      className="navbar"
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <motion.div 
        className="navbar-brand" 
        onClick={() => window.location.reload()}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        style={{ cursor: 'pointer' }}
      >
        <div className="navbar-logo">
          <TrendingUp size={20} />
        </div>
        <span className="navbar-title">
          Signal<span>ist</span>
        </span>
      </motion.div>

      <div className="navbar-center">
        <SearchBox 
          onSelectStock={onSelectStock} 
          placeholder="Search stocks... (e.g., Reliance, TCS, INFY)"
        />
      </div>

      <div className="navbar-actions">
        <motion.button 
          className={`navbar-btn ${refreshing ? 'primary' : ''}`}
          onClick={onRefresh}
          title="Refresh market data"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <RefreshCw 
            size={18} 
            className={refreshing ? 'spinning' : ''}
          />
          <span>Refresh</span>
        </motion.button>
      </div>
    </motion.nav>
  );
};

export default Navbar;
