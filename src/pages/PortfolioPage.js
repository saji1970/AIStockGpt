import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Plus, X, Trash2, RefreshCw, ArrowLeft, Briefcase, TrendingUp, TrendingDown } from 'lucide-react';
import toast from 'react-hot-toast';
import PortfolioCard from '../components/PortfolioCard';
import {
  listPortfolios,
  createPortfolio,
  getPortfolioSummary,
  addStockToPortfolio,
  deleteStockFromPortfolio,
} from '../services/api';

export default function PortfolioPage() {
  const [portfolios, setPortfolios] = useState([]);
  const [selectedPortfolio, setSelectedPortfolio] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showAddStockModal, setShowAddStockModal] = useState(false);

  // Create portfolio form
  const [newName, setNewName] = useState('');
  const [newDesc, setNewDesc] = useState('');

  // Add stock form
  const [stockSymbol, setStockSymbol] = useState('');
  const [stockShares, setStockShares] = useState('');
  const [stockPrice, setStockPrice] = useState('');
  const [stockDate, setStockDate] = useState('');

  useEffect(() => {
    fetchPortfolios();
  }, []);

  const fetchPortfolios = async () => {
    try {
      setLoading(true);
      const data = await listPortfolios();
      setPortfolios(data.portfolios || []);
    } catch (err) {
      toast.error('Failed to load portfolios');
    } finally {
      setLoading(false);
    }
  };

  const openPortfolio = async (id) => {
    try {
      setLoadingSummary(true);
      const summary = await getPortfolioSummary(id);
      setSelectedPortfolio(summary);
    } catch (err) {
      toast.error('Failed to load portfolio details');
    } finally {
      setLoadingSummary(false);
    }
  };

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!newName.trim()) {
      toast.error('Portfolio name is required');
      return;
    }
    try {
      await createPortfolio(newName, newDesc || undefined);
      toast.success('Portfolio created!');
      setShowCreateModal(false);
      setNewName('');
      setNewDesc('');
      fetchPortfolios();
    } catch (err) {
      toast.error(err?.response?.data?.detail || 'Failed to create portfolio');
    }
  };

  const handleAddStock = async (e) => {
    e.preventDefault();
    if (!stockSymbol.trim() || !stockShares || !stockPrice || !stockDate) {
      toast.error('All fields are required');
      return;
    }
    try {
      await addStockToPortfolio(
        selectedPortfolio.id,
        stockSymbol.toUpperCase(),
        parseFloat(stockShares),
        parseFloat(stockPrice),
        stockDate
      );
      toast.success(`${stockSymbol.toUpperCase()} added!`);
      setShowAddStockModal(false);
      setStockSymbol('');
      setStockShares('');
      setStockPrice('');
      setStockDate('');
      openPortfolio(selectedPortfolio.id);
    } catch (err) {
      toast.error(err?.response?.data?.detail || 'Failed to add stock');
    }
  };

  const handleDeleteStock = async (symbol) => {
    if (!window.confirm(`Remove ${symbol} from this portfolio?`)) return;
    try {
      await deleteStockFromPortfolio(selectedPortfolio.id, symbol);
      toast.success(`${symbol} removed`);
      openPortfolio(selectedPortfolio.id);
    } catch (err) {
      toast.error(err?.response?.data?.detail || 'Failed to remove stock');
    }
  };

  // Modal backdrop
  const Modal = ({ show, onClose, children }) => (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 px-4"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-white rounded-xl shadow-lg w-full max-w-md p-6"
          >
            {children}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );

  // Portfolio detail view
  if (selectedPortfolio) {
    const summary = selectedPortfolio.summary || {};
    const isPositive = (summary.total_gain_loss || 0) >= 0;

    return (
      <div className="flex-1 max-w-5xl mx-auto w-full p-6">
        {/* Back button + title */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setSelectedPortfolio(null)}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-gray-600" />
            </button>
            <div>
              <h2 className="text-2xl font-bold text-gray-900">{selectedPortfolio.name}</h2>
              {selectedPortfolio.description && (
                <p className="text-sm text-gray-500">{selectedPortfolio.description}</p>
              )}
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => openPortfolio(selectedPortfolio.id)}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              title="Refresh prices"
            >
              <RefreshCw className={`w-5 h-5 text-gray-600 ${loadingSummary ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={() => setShowAddStockModal(true)}
              className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-lg text-sm font-medium hover:from-primary-600 hover:to-primary-700 transition-all"
            >
              <Plus className="w-4 h-4" />
              <span>Add Holding</span>
            </button>
          </div>
        </div>

        {/* Summary bar */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white border border-gray-200 rounded-xl p-4">
            <p className="text-xs text-gray-500 mb-1">Total Invested</p>
            <p className="text-xl font-bold text-gray-900">
              ${(summary.total_invested || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </p>
          </div>
          <div className="bg-white border border-gray-200 rounded-xl p-4">
            <p className="text-xs text-gray-500 mb-1">Current Value</p>
            <p className="text-xl font-bold text-gray-900">
              ${(summary.total_current_value || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </p>
          </div>
          <div className="bg-white border border-gray-200 rounded-xl p-4">
            <p className="text-xs text-gray-500 mb-1">Total Gain/Loss</p>
            <p className={`text-xl font-bold ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
              {isPositive ? '+' : ''}${(summary.total_gain_loss || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </p>
          </div>
          <div className="bg-white border border-gray-200 rounded-xl p-4">
            <p className="text-xs text-gray-500 mb-1">Return</p>
            <p className={`text-xl font-bold ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
              {isPositive ? '+' : ''}{(summary.total_gain_loss_percent || 0).toFixed(2)}%
            </p>
          </div>
        </div>

        {/* Holdings table */}
        {selectedPortfolio.stocks && selectedPortfolio.stocks.length > 0 ? (
          <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200 bg-gray-50">
                    <th className="text-left text-xs font-medium text-gray-500 uppercase px-4 py-3">Symbol</th>
                    <th className="text-right text-xs font-medium text-gray-500 uppercase px-4 py-3">Shares</th>
                    <th className="text-right text-xs font-medium text-gray-500 uppercase px-4 py-3">Avg Cost</th>
                    <th className="text-right text-xs font-medium text-gray-500 uppercase px-4 py-3">Current Price</th>
                    <th className="text-right text-xs font-medium text-gray-500 uppercase px-4 py-3">Value</th>
                    <th className="text-right text-xs font-medium text-gray-500 uppercase px-4 py-3">Gain/Loss</th>
                    <th className="text-right text-xs font-medium text-gray-500 uppercase px-4 py-3"></th>
                  </tr>
                </thead>
                <tbody>
                  {selectedPortfolio.stocks.map((stock) => {
                    const gain = stock.gain_loss;
                    const gainPct = stock.gain_loss_percent;
                    const isUp = gain !== null && gain >= 0;

                    return (
                      <tr key={stock.symbol} className="border-b border-gray-100 hover:bg-gray-50">
                        <td className="px-4 py-3">
                          <div>
                            <span className="font-semibold text-gray-900">{stock.symbol}</span>
                            {stock.name && stock.name !== stock.symbol && (
                              <p className="text-xs text-gray-500">{stock.name}</p>
                            )}
                          </div>
                        </td>
                        <td className="text-right px-4 py-3 text-sm text-gray-900">{stock.shares}</td>
                        <td className="text-right px-4 py-3 text-sm text-gray-900">
                          ${stock.purchase_price?.toFixed(2)}
                        </td>
                        <td className="text-right px-4 py-3 text-sm text-gray-900">
                          {stock.current_price != null ? `$${stock.current_price.toFixed(2)}` : '--'}
                        </td>
                        <td className="text-right px-4 py-3 text-sm text-gray-900">
                          {stock.current_value != null
                            ? `$${stock.current_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                            : '--'}
                        </td>
                        <td className="text-right px-4 py-3">
                          {gain !== null ? (
                            <div className="flex items-center justify-end space-x-1">
                              {isUp ? (
                                <TrendingUp className="w-4 h-4 text-green-500" />
                              ) : (
                                <TrendingDown className="w-4 h-4 text-red-500" />
                              )}
                              <span className={`text-sm font-medium ${isUp ? 'text-green-600' : 'text-red-600'}`}>
                                {isUp ? '+' : ''}${gain.toFixed(2)} ({isUp ? '+' : ''}{gainPct?.toFixed(2)}%)
                              </span>
                            </div>
                          ) : (
                            <span className="text-sm text-gray-400">--</span>
                          )}
                        </td>
                        <td className="text-right px-4 py-3">
                          <button
                            onClick={() => handleDeleteStock(stock.symbol)}
                            className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                            title="Remove"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="bg-white border border-gray-200 rounded-xl p-12 text-center">
            <Briefcase className="w-12 h-12 mx-auto mb-3 text-gray-300" />
            <p className="text-gray-500 mb-4">No holdings yet. Add stocks, ETFs, or crypto to your portfolio.</p>
            <button
              onClick={() => setShowAddStockModal(true)}
              className="px-4 py-2 bg-primary-500 text-white rounded-lg text-sm font-medium hover:bg-primary-600 transition-colors"
            >
              Add Your First Holding
            </button>
          </div>
        )}

        {/* Add Stock Modal */}
        <Modal show={showAddStockModal} onClose={() => setShowAddStockModal(false)}>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Add Holding</h3>
            <button onClick={() => setShowAddStockModal(false)} className="text-gray-400 hover:text-gray-600">
              <X className="w-5 h-5" />
            </button>
          </div>
          <form onSubmit={handleAddStock} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Symbol</label>
              <input
                type="text"
                value={stockSymbol}
                onChange={(e) => setStockSymbol(e.target.value.toUpperCase())}
                className="w-full px-3 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
                placeholder="e.g. AAPL, SPY, BTC-USD"
                required
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Shares</label>
                <input
                  type="number"
                  step="any"
                  min="0"
                  value={stockShares}
                  onChange={(e) => setStockShares(e.target.value)}
                  className="w-full px-3 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
                  placeholder="10"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Purchase Price</label>
                <input
                  type="number"
                  step="any"
                  min="0"
                  value={stockPrice}
                  onChange={(e) => setStockPrice(e.target.value)}
                  className="w-full px-3 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
                  placeholder="150.00"
                  required
                />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Purchase Date</label>
              <input
                type="date"
                value={stockDate}
                onChange={(e) => setStockDate(e.target.value)}
                className="w-full px-3 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
                required
              />
            </div>
            <button
              type="submit"
              className="w-full py-2.5 bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-lg font-medium text-sm hover:from-primary-600 hover:to-primary-700 transition-all"
            >
              Add Holding
            </button>
          </form>
        </Modal>
      </div>
    );
  }

  // Portfolio list view
  return (
    <div className="flex-1 max-w-5xl mx-auto w-full p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">My Portfolios</h2>
          <p className="text-sm text-gray-500 mt-1">Track your investments and monitor performance</p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-lg text-sm font-medium hover:from-primary-600 hover:to-primary-700 transition-all"
        >
          <Plus className="w-4 h-4" />
          <span>New Portfolio</span>
        </button>
      </div>

      {loading ? (
        <div className="text-center py-12 text-gray-500">Loading portfolios...</div>
      ) : portfolios.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {portfolios.map((portfolio) => (
            <motion.div
              key={portfolio.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <PortfolioCard
                name={portfolio.name}
                description={portfolio.description}
                totalValue={portfolio.total_value || 0}
                totalGainLoss={portfolio.total_gain_loss || 0}
                stockCount={(portfolio.stocks || []).length}
                onClick={() => openPortfolio(portfolio.id)}
              />
            </motion.div>
          ))}
        </div>
      ) : (
        <div className="bg-white border border-gray-200 rounded-xl p-12 text-center">
          <Briefcase className="w-16 h-16 mx-auto mb-4 text-gray-300" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">No portfolios yet</h3>
          <p className="text-sm text-gray-500 mb-6">
            Create a portfolio to start tracking your stocks, ETFs, and crypto.
          </p>
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-6 py-2.5 bg-primary-500 text-white rounded-lg text-sm font-medium hover:bg-primary-600 transition-colors"
          >
            Create Your First Portfolio
          </button>
        </div>
      )}

      {loadingSummary && (
        <div className="fixed inset-0 bg-black/20 flex items-center justify-center z-40">
          <div className="bg-white rounded-xl p-6 shadow-lg">
            <RefreshCw className="w-8 h-8 text-primary-500 animate-spin mx-auto mb-3" />
            <p className="text-sm text-gray-600">Fetching live prices...</p>
          </div>
        </div>
      )}

      {/* Create Portfolio Modal */}
      <Modal show={showCreateModal} onClose={() => setShowCreateModal(false)}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">New Portfolio</h3>
          <button onClick={() => setShowCreateModal(false)} className="text-gray-400 hover:text-gray-600">
            <X className="w-5 h-5" />
          </button>
        </div>
        <form onSubmit={handleCreate} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
            <input
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              className="w-full px-3 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
              placeholder="e.g. My Tech Stocks"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Description (optional)</label>
            <textarea
              value={newDesc}
              onChange={(e) => setNewDesc(e.target.value)}
              className="w-full px-3 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
              placeholder="Brief description of this portfolio"
              rows={2}
            />
          </div>
          <button
            type="submit"
            className="w-full py-2.5 bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-lg font-medium text-sm hover:from-primary-600 hover:to-primary-700 transition-all"
          >
            Create Portfolio
          </button>
        </form>
      </Modal>
    </div>
  );
}
