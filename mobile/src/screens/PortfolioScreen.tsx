import React, {useState, useCallback} from 'react';
import {View, Text, FlatList, TouchableOpacity, TextInput, Modal, StyleSheet, Alert, ActivityIndicator, RefreshControl} from 'react-native';
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
  const [selectedPortfolio, setSelectedPortfolio] = useState<Portfolio | null>(null);

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
      // Refresh the selected portfolio
      const updated = await api.getPortfolio(selectedPortfolio.id);
      setSelectedPortfolio(updated);
      fetchPortfolios();
    } catch (err: any) {
      Alert.alert('Error', err?.response?.data?.detail || 'Failed to add stock');
    } finally {
      setAddingStock(false);
    }
  };

  const openPortfolio = async (id: string) => {
    try {
      const data = await api.getPortfolio(id);
      setSelectedPortfolio(data);
    } catch {
      Alert.alert('Error', 'Failed to load portfolio');
    }
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
          />
        )}
      />

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
          <View style={[styles.modal, {maxHeight: '80%'}]}>
            <Text style={styles.modalTitle}>{selectedPortfolio?.name}</Text>
            {selectedPortfolio?.description ? <Text style={styles.detailDesc}>{selectedPortfolio.description}</Text> : null}

            <Text style={styles.sectionTitle}>Stocks</Text>
            {(selectedPortfolio?.stocks || []).length === 0 ? (
              <Text style={styles.emptySubtext}>No stocks in this portfolio</Text>
            ) : (
              (selectedPortfolio?.stocks || []).map((s: any, i: number) => (
                <View key={i} style={styles.stockRow}>
                  <Text style={styles.stockSymbol}>{s.symbol}</Text>
                  <Text style={styles.stockDetail}>{s.shares} shares @ ${s.purchase_price?.toFixed(2)}</Text>
                </View>
              ))
            )}

            <View style={styles.modalButtons}>
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
  modalOverlay: {flex: 1, backgroundColor: 'rgba(0,0,0,0.4)', justifyContent: 'center', padding: 20},
  modal: {backgroundColor: '#fff', borderRadius: 16, padding: 20},
  modalTitle: {fontSize: 20, fontWeight: '700', color: '#111827', marginBottom: 16},
  modalInput: {backgroundColor: '#f3f4f6', borderRadius: 10, paddingHorizontal: 14, paddingVertical: 12, fontSize: 15, marginBottom: 12, color: '#111827'},
  modalButtons: {flexDirection: 'row', justifyContent: 'flex-end', gap: 10, marginTop: 8},
  cancelBtn: {paddingHorizontal: 18, paddingVertical: 10, borderRadius: 10, backgroundColor: '#f3f4f6'},
  cancelText: {color: '#374151', fontWeight: '600'},
  confirmBtn: {paddingHorizontal: 18, paddingVertical: 10, borderRadius: 10, backgroundColor: '#6366f1'},
  confirmText: {color: '#fff', fontWeight: '600'},
  detailDesc: {fontSize: 13, color: '#6b7280', marginBottom: 12},
  sectionTitle: {fontSize: 15, fontWeight: '700', color: '#111827', marginTop: 12, marginBottom: 8},
  stockRow: {flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: '#f3f4f6'},
  stockSymbol: {fontSize: 14, fontWeight: '700', color: '#111827'},
  stockDetail: {fontSize: 13, color: '#6b7280'},
});
