import React, {useState, useCallback} from 'react';
import {View, Text, FlatList, ScrollView, TouchableOpacity, TextInput, Modal, StyleSheet, Alert, ActivityIndicator, RefreshControl} from 'react-native';
import {useFocusEffect} from '@react-navigation/native';
import PortfolioCard from '../components/PortfolioCard';
import * as api from '../api/client';

interface Portfolio {
  id: string;
  name: string;
  description?: string;
  total_value: number;
  total_gain_loss: number;
  stocks: any[];
}

interface PortfolioDetail {
  id: string;
  name: string;
  description?: string;
  stocks: any[];
  summary: {
    total_invested: number;
    total_current_value: number;
    total_gain_loss: number;
    total_gain_loss_percent: number;
    stock_count: number;
  };
}

export default function PortfolioScreen() {
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Create portfolio modal
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [creating, setCreating] = useState(false);

  // Detail modal
  const [selectedPortfolio, setSelectedPortfolio] = useState<PortfolioDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);

  // Add stock modal
  const [showAddStock, setShowAddStock] = useState(false);
  const [stockSymbol, setStockSymbol] = useState('');
  const [stockShares, setStockShares] = useState('');
  const [stockPrice, setStockPrice] = useState('');
  const [stockDate, setStockDate] = useState('');
  const [addingStock, setAddingStock] = useState(false);

  const fetchPortfolios = useCallback(async () => {
    try {
      const data = await api.listPortfolios();
      setPortfolios(data.portfolios || []);
    } catch {
      // silently handle
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      fetchPortfolios();
    }, [fetchPortfolios]),
  );

  const handleCreate = async () => {
    if (!newName.trim()) return;
    setCreating(true);
    try {
      await api.createPortfolio(newName.trim(), newDesc.trim() || undefined);
      setShowCreate(false);
      setNewName('');
      setNewDesc('');
      fetchPortfolios();
    } catch (err: any) {
      Alert.alert('Error', err?.response?.data?.detail || 'Failed to create portfolio');
    } finally {
      setCreating(false);
    }
  };

  const handleAddStock = async () => {
    if (!selectedPortfolio || !stockSymbol.trim() || !stockShares || !stockPrice) return;
    setAddingStock(true);
    try {
      await api.addStockToPortfolio(
        selectedPortfolio.id,
        stockSymbol.trim().toUpperCase(),
        parseFloat(stockShares),
        parseFloat(stockPrice),
        stockDate || new Date().toISOString().split('T')[0],
      );
      setShowAddStock(false);
      setStockSymbol('');
      setStockShares('');
      setStockPrice('');
      setStockDate('');
      // Refresh with live data
      const updated = await api.getPortfolioSummary(selectedPortfolio.id);
      setSelectedPortfolio(updated);
      fetchPortfolios();
    } catch (err: any) {
      Alert.alert('Error', err?.response?.data?.detail || 'Failed to add stock');
    } finally {
      setAddingStock(false);
    }
  };

  const handleDeleteStock = async (symbol: string) => {
    if (!selectedPortfolio) return;
    Alert.alert('Remove Stock', `Remove ${symbol} from this portfolio?`, [
      {text: 'Cancel', style: 'cancel'},
      {
        text: 'Remove',
        style: 'destructive',
        onPress: async () => {
          try {
            await api.deleteStockFromPortfolio(selectedPortfolio.id, symbol);
            const updated = await api.getPortfolioSummary(selectedPortfolio.id);
            setSelectedPortfolio(updated);
            fetchPortfolios();
          } catch {
            Alert.alert('Error', 'Failed to remove stock');
          }
        },
      },
    ]);
  };

  const handleDeletePortfolio = (portfolio: {id: string; name: string}) => {
    Alert.alert(
      'Delete Portfolio',
      `Are you sure you want to delete "${portfolio.name}" and all its stocks? This cannot be undone.`,
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await api.deletePortfolio(portfolio.id);
              setSelectedPortfolio(null);
              fetchPortfolios();
            } catch {
              Alert.alert('Error', 'Failed to delete portfolio');
            }
          },
        },
      ],
    );
  };

  const openPortfolio = async (id: string) => {
    setLoadingDetail(true);
    try {
      const data = await api.getPortfolioSummary(id);
      setSelectedPortfolio(data);
    } catch {
      Alert.alert('Error', 'Failed to load portfolio');
    } finally {
      setLoadingDetail(false);
    }
  };

  const formatCurrency = (val: number | null | undefined) => {
    if (val == null) return '--';
    return `$${val.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
  };

  const formatPercent = (val: number | null | undefined) => {
    if (val == null) return '';
    const sign = val >= 0 ? '+' : '';
    return `${sign}${val.toFixed(2)}%`;
  };

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#6366f1" />
      </View>
    );
  }

  return (
    <View style={styles.flex}>
      <FlatList
        data={portfolios}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.list}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => {setRefreshing(true); fetchPortfolios();}} colors={['#6366f1']} />}
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={styles.emptyText}>No portfolios yet</Text>
            <Text style={styles.emptySubtext}>Create one to start tracking stocks</Text>
          </View>
        }
        renderItem={({item}) => (
          <PortfolioCard
            name={item.name}
            description={item.description}
            totalValue={item.total_value || 0}
            totalGainLoss={item.total_gain_loss || 0}
            stockCount={(item.stocks || []).length}
            onPress={() => openPortfolio(item.id)}
            onLongPress={() => handleDeletePortfolio({id: item.id, name: item.name})}
          />
        )}
      />

      {/* Loading overlay for detail fetch */}
      {loadingDetail && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color="#6366f1" />
          <Text style={styles.loadingText}>Fetching live prices...</Text>
        </View>
      )}

      {/* Create button */}
      <TouchableOpacity style={styles.fab} onPress={() => setShowCreate(true)} activeOpacity={0.8}>
        <Text style={styles.fabText}>+</Text>
      </TouchableOpacity>

      {/* Create Portfolio Modal */}
      <Modal visible={showCreate} animationType="slide" transparent>
        <View style={styles.modalOverlay}>
          <View style={styles.modal}>
            <Text style={styles.modalTitle}>New Portfolio</Text>
            <TextInput style={styles.modalInput} placeholder="Portfolio Name" value={newName} onChangeText={setNewName} placeholderTextColor="#9ca3af" />
            <TextInput style={styles.modalInput} placeholder="Description (optional)" value={newDesc} onChangeText={setNewDesc} placeholderTextColor="#9ca3af" />
            <View style={styles.modalButtons}>
              <TouchableOpacity style={styles.cancelBtn} onPress={() => setShowCreate(false)}>
                <Text style={styles.cancelText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.confirmBtn} onPress={handleCreate} disabled={creating}>
                {creating ? <ActivityIndicator color="#fff" size="small" /> : <Text style={styles.confirmText}>Create</Text>}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Portfolio Detail Modal */}
      <Modal visible={!!selectedPortfolio} animationType="slide" transparent>
        <View style={styles.modalOverlay}>
          <View style={[styles.modal, {maxHeight: '85%'}]}>
            <Text style={styles.modalTitle}>{selectedPortfolio?.name}</Text>
            {selectedPortfolio?.description ? <Text style={styles.detailDesc}>{selectedPortfolio.description}</Text> : null}

            {/* Portfolio Summary */}
            {selectedPortfolio?.summary && (
              <View style={styles.summaryContainer}>
                <View style={styles.summaryRow}>
                  <View style={styles.summaryItem}>
                    <Text style={styles.summaryLabel}>Total Value</Text>
                    <Text style={styles.summaryValue}>{formatCurrency(selectedPortfolio.summary.total_current_value)}</Text>
                  </View>
                  <View style={[styles.summaryItem, {alignItems: 'flex-end'}]}>
                    <Text style={styles.summaryLabel}>Total Invested</Text>
                    <Text style={styles.summaryValue}>{formatCurrency(selectedPortfolio.summary.total_invested)}</Text>
                  </View>
                </View>
                <View style={styles.summaryRow}>
                  <View style={styles.summaryItem}>
                    <Text style={styles.summaryLabel}>Gain / Loss</Text>
                    <Text style={[styles.summaryValue, {color: selectedPortfolio.summary.total_gain_loss >= 0 ? '#10b981' : '#ef4444'}]}>
                      {selectedPortfolio.summary.total_gain_loss >= 0 ? '+' : ''}{formatCurrency(selectedPortfolio.summary.total_gain_loss)}
                    </Text>
                  </View>
                  <View style={[styles.summaryItem, {alignItems: 'flex-end'}]}>
                    <Text style={styles.summaryLabel}>Return</Text>
                    <Text style={[styles.summaryValue, {color: selectedPortfolio.summary.total_gain_loss_percent >= 0 ? '#10b981' : '#ef4444'}]}>
                      {formatPercent(selectedPortfolio.summary.total_gain_loss_percent)}
                    </Text>
                  </View>
                </View>
              </View>
            )}

            <Text style={styles.sectionTitle}>Holdings ({selectedPortfolio?.summary?.stock_count || 0})</Text>

            <ScrollView style={styles.stockList} showsVerticalScrollIndicator={false}>
              {(selectedPortfolio?.stocks || []).length === 0 ? (
                <Text style={styles.emptySubtext}>No stocks in this portfolio. Tap "Add Stock" to get started.</Text>
              ) : (
                (selectedPortfolio?.stocks || []).map((s: any, i: number) => {
                  const gainColor = s.gain_loss != null ? (s.gain_loss >= 0 ? '#10b981' : '#ef4444') : '#6b7280';
                  return (
                    <View key={i} style={styles.stockCard}>
                      <View style={styles.stockHeader}>
                        <View style={{flex: 1}}>
                          <Text style={styles.stockSymbol}>{s.symbol}</Text>
                          {s.name && s.name !== s.symbol ? <Text style={styles.stockName}>{s.name}</Text> : null}
                        </View>
                        <View style={{alignItems: 'flex-end', flexDirection: 'row', gap: 10}}>
                          <View style={{alignItems: 'flex-end'}}>
                            <Text style={styles.stockCurrentPrice}>{formatCurrency(s.current_price)}</Text>
                            {s.gain_loss_percent != null && (
                              <Text style={[styles.stockGainBadge, {color: gainColor}]}>
                                {s.gain_loss >= 0 ? '+' : ''}{s.gain_loss_percent.toFixed(2)}%
                              </Text>
                            )}
                          </View>
                          <TouchableOpacity style={styles.deleteStockBtn} onPress={() => handleDeleteStock(s.symbol)} activeOpacity={0.7}>
                            <Text style={styles.deleteStockIcon}>X</Text>
                          </TouchableOpacity>
                        </View>
                      </View>
                      <View style={styles.stockMetrics}>
                        <View style={styles.stockMetric}>
                          <Text style={styles.metricLabel}>Shares</Text>
                          <Text style={styles.metricVal}>{s.shares}</Text>
                        </View>
                        <View style={styles.stockMetric}>
                          <Text style={styles.metricLabel}>Avg Cost</Text>
                          <Text style={styles.metricVal}>{formatCurrency(s.purchase_price)}</Text>
                        </View>
                        <View style={styles.stockMetric}>
                          <Text style={styles.metricLabel}>Value</Text>
                          <Text style={styles.metricVal}>{formatCurrency(s.current_value)}</Text>
                        </View>
                        <View style={styles.stockMetric}>
                          <Text style={styles.metricLabel}>P/L</Text>
                          <Text style={[styles.metricVal, {color: gainColor}]}>
                            {s.gain_loss != null ? `${s.gain_loss >= 0 ? '+' : ''}${formatCurrency(s.gain_loss)}` : '--'}
                          </Text>
                        </View>
                      </View>
                    </View>
                  );
                })
              )}
            </ScrollView>

            <View style={styles.modalButtons}>
              <TouchableOpacity style={styles.deletePfBtn} onPress={() => selectedPortfolio && handleDeletePortfolio({id: selectedPortfolio.id, name: selectedPortfolio.name})}>
                <Text style={styles.deletePfText}>Delete Portfolio</Text>
              </TouchableOpacity>
              <View style={{flex: 1}} />
              <TouchableOpacity style={styles.cancelBtn} onPress={() => setSelectedPortfolio(null)}>
                <Text style={styles.cancelText}>Close</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.confirmBtn} onPress={() => setShowAddStock(true)}>
                <Text style={styles.confirmText}>Add Stock</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Add Stock Modal */}
      <Modal visible={showAddStock} animationType="slide" transparent>
        <View style={styles.modalOverlay}>
          <View style={styles.modal}>
            <Text style={styles.modalTitle}>Add Stock</Text>
            <TextInput style={styles.modalInput} placeholder="Symbol (e.g. AAPL)" value={stockSymbol} onChangeText={setStockSymbol} autoCapitalize="characters" placeholderTextColor="#9ca3af" />
            <TextInput style={styles.modalInput} placeholder="Shares" value={stockShares} onChangeText={setStockShares} keyboardType="numeric" placeholderTextColor="#9ca3af" />
            <TextInput style={styles.modalInput} placeholder="Purchase Price" value={stockPrice} onChangeText={setStockPrice} keyboardType="numeric" placeholderTextColor="#9ca3af" />
            <TextInput style={styles.modalInput} placeholder="Date (YYYY-MM-DD)" value={stockDate} onChangeText={setStockDate} placeholderTextColor="#9ca3af" />
            <View style={styles.modalButtons}>
              <TouchableOpacity style={styles.cancelBtn} onPress={() => setShowAddStock(false)}>
                <Text style={styles.cancelText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.confirmBtn} onPress={handleAddStock} disabled={addingStock}>
                {addingStock ? <ActivityIndicator color="#fff" size="small" /> : <Text style={styles.confirmText}>Add</Text>}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  flex: {flex: 1, backgroundColor: '#f9fafb'},
  center: {flex: 1, justifyContent: 'center', alignItems: 'center'},
  list: {paddingVertical: 8},
  empty: {alignItems: 'center', paddingTop: 60},
  emptyText: {fontSize: 18, fontWeight: '600', color: '#374151'},
  emptySubtext: {fontSize: 13, color: '#9ca3af', marginTop: 4},
  fab: {position: 'absolute', bottom: 20, right: 20, width: 56, height: 56, borderRadius: 28, backgroundColor: '#6366f1', justifyContent: 'center', alignItems: 'center', elevation: 6},
  fabText: {color: '#fff', fontSize: 28, fontWeight: '300', marginTop: -2},
  loadingOverlay: {position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.3)', justifyContent: 'center', alignItems: 'center'},
  loadingText: {color: '#fff', fontSize: 14, fontWeight: '600', marginTop: 10},
  modalOverlay: {flex: 1, backgroundColor: 'rgba(0,0,0,0.4)', justifyContent: 'center', padding: 20},
  modal: {backgroundColor: '#fff', borderRadius: 16, padding: 20},
  modalTitle: {fontSize: 20, fontWeight: '700', color: '#111827', marginBottom: 16},
  modalInput: {backgroundColor: '#f3f4f6', borderRadius: 10, paddingHorizontal: 14, paddingVertical: 12, fontSize: 15, marginBottom: 12, color: '#111827'},
  modalButtons: {flexDirection: 'row', justifyContent: 'flex-end', gap: 10, marginTop: 12},
  cancelBtn: {paddingHorizontal: 18, paddingVertical: 10, borderRadius: 10, backgroundColor: '#f3f4f6'},
  cancelText: {color: '#374151', fontWeight: '600'},
  confirmBtn: {paddingHorizontal: 18, paddingVertical: 10, borderRadius: 10, backgroundColor: '#6366f1'},
  confirmText: {color: '#fff', fontWeight: '600'},
  detailDesc: {fontSize: 13, color: '#6b7280', marginBottom: 12},
  sectionTitle: {fontSize: 15, fontWeight: '700', color: '#111827', marginTop: 12, marginBottom: 8},
  // Summary
  summaryContainer: {backgroundColor: '#f8fafc', borderRadius: 12, padding: 14, marginBottom: 4, borderWidth: 1, borderColor: '#e2e8f0'},
  summaryRow: {flexDirection: 'row', justifyContent: 'space-between', marginBottom: 8},
  summaryItem: {flex: 1},
  summaryLabel: {fontSize: 11, color: '#6b7280', marginBottom: 2},
  summaryValue: {fontSize: 16, fontWeight: '700', color: '#111827'},
  // Stock cards
  stockList: {maxHeight: 300},
  stockCard: {backgroundColor: '#f9fafb', borderRadius: 10, padding: 12, marginBottom: 8, borderWidth: 1, borderColor: '#e5e7eb'},
  stockHeader: {flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8},
  stockSymbol: {fontSize: 15, fontWeight: '700', color: '#111827'},
  stockName: {fontSize: 11, color: '#6b7280', marginTop: 1},
  stockCurrentPrice: {fontSize: 15, fontWeight: '700', color: '#111827'},
  stockGainBadge: {fontSize: 12, fontWeight: '600', marginTop: 1},
  stockMetrics: {flexDirection: 'row', justifyContent: 'space-between'},
  stockMetric: {flex: 1},
  metricLabel: {fontSize: 10, color: '#9ca3af'},
  metricVal: {fontSize: 12, fontWeight: '600', color: '#374151', marginTop: 1},
  // Delete stock button
  deleteStockBtn: {width: 28, height: 28, borderRadius: 14, backgroundColor: '#fee2e2', justifyContent: 'center', alignItems: 'center'},
  deleteStockIcon: {color: '#ef4444', fontSize: 13, fontWeight: '700'},
  // Delete portfolio button
  deletePfBtn: {paddingHorizontal: 14, paddingVertical: 10, borderRadius: 10, backgroundColor: '#fee2e2'},
  deletePfText: {color: '#ef4444', fontWeight: '600', fontSize: 13},
});
