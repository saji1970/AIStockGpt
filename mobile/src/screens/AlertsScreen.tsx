import React, {useState, useCallback} from 'react';
import {View, Text, FlatList, TouchableOpacity, TextInput, Modal, StyleSheet, Alert, ActivityIndicator, RefreshControl} from 'react-native';
import {useFocusEffect} from '@react-navigation/native';
import * as api from '../api/client';

interface AlertItem {
  id: string;
  symbol: string;
  alert_type: string;
  threshold: number;
  email: string;
  is_active: boolean;
  created_at: string;
}

const ALERT_TYPES = ['price', 'technical', 'prediction'];

export default function AlertsScreen() {
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const [showCreate, setShowCreate] = useState(false);
  const [symbol, setSymbol] = useState('');
  const [alertType, setAlertType] = useState('price');
  const [threshold, setThreshold] = useState('');
  const [email, setEmail] = useState('');
  const [creating, setCreating] = useState(false);

  const fetchAlerts = useCallback(async () => {
    try {
      const data = await api.listAlerts();
      setAlerts(data.alerts || []);
    } catch {
      // silently handle
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      fetchAlerts();
    }, [fetchAlerts]),
  );

  const handleCreate = async () => {
    if (!symbol.trim() || !threshold || !email.trim()) {
      Alert.alert('Error', 'All fields are required');
      return;
    }
    setCreating(true);
    try {
      await api.createAlert(symbol.trim().toUpperCase(), alertType, parseFloat(threshold), email.trim());
      setShowCreate(false);
      setSymbol('');
      setThreshold('');
      setEmail('');
      fetchAlerts();
    } catch (err: any) {
      Alert.alert('Error', err?.response?.data?.detail || 'Failed to create alert');
    } finally {
      setCreating(false);
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
        data={alerts}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.list}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => {setRefreshing(true); fetchAlerts();}} colors={['#6366f1']} />}
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={styles.emptyText}>No alerts yet</Text>
            <Text style={styles.emptySubtext}>Create alerts to get notified about stock movements</Text>
          </View>
        }
        renderItem={({item}) => (
          <View style={styles.alertCard}>
            <View style={styles.alertHeader}>
              <Text style={styles.alertSymbol}>{item.symbol}</Text>
              <View style={[styles.typeBadge, item.is_active ? styles.activeBadge : styles.inactiveBadge]}>
                <Text style={styles.typeBadgeText}>{item.alert_type}</Text>
              </View>
            </View>
            <Text style={styles.alertDetail}>Threshold: ${item.threshold.toFixed(2)}</Text>
            <Text style={styles.alertEmail}>{item.email}</Text>
          </View>
        )}
      />

      <TouchableOpacity style={styles.fab} onPress={() => setShowCreate(true)} activeOpacity={0.8}>
        <Text style={styles.fabText}>+</Text>
      </TouchableOpacity>

      {/* Create Alert Modal */}
      <Modal visible={showCreate} animationType="slide" transparent>
        <View style={styles.modalOverlay}>
          <View style={styles.modal}>
            <Text style={styles.modalTitle}>New Alert</Text>
            <TextInput style={styles.modalInput} placeholder="Symbol (e.g. AAPL)" value={symbol} onChangeText={setSymbol} autoCapitalize="characters" placeholderTextColor="#9ca3af" />

            {/* Alert type picker */}
            <Text style={styles.pickerLabel}>Alert Type</Text>
            <View style={styles.typeRow}>
              {ALERT_TYPES.map(t => (
                <TouchableOpacity
                  key={t}
                  style={[styles.typeOption, alertType === t && styles.typeSelected]}
                  onPress={() => setAlertType(t)}>
                  <Text style={[styles.typeOptionText, alertType === t && styles.typeSelectedText]}>{t}</Text>
                </TouchableOpacity>
              ))}
            </View>

            <TextInput style={styles.modalInput} placeholder="Threshold Price" value={threshold} onChangeText={setThreshold} keyboardType="numeric" placeholderTextColor="#9ca3af" />
            <TextInput style={styles.modalInput} placeholder="Email" value={email} onChangeText={setEmail} keyboardType="email-address" autoCapitalize="none" placeholderTextColor="#9ca3af" />

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
    </View>
  );
}

const styles = StyleSheet.create({
  flex: {flex: 1, backgroundColor: '#f9fafb'},
  center: {flex: 1, justifyContent: 'center', alignItems: 'center'},
  list: {paddingVertical: 8},
  empty: {alignItems: 'center', paddingTop: 60},
  emptyText: {fontSize: 18, fontWeight: '600', color: '#374151'},
  emptySubtext: {fontSize: 13, color: '#9ca3af', marginTop: 4, textAlign: 'center', paddingHorizontal: 40},
  alertCard: {backgroundColor: '#fff', borderRadius: 12, padding: 14, marginVertical: 4, marginHorizontal: 12, borderWidth: 1, borderColor: '#e5e7eb'},
  alertHeader: {flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6},
  alertSymbol: {fontSize: 16, fontWeight: '700', color: '#111827'},
  typeBadge: {borderRadius: 10, paddingHorizontal: 10, paddingVertical: 3},
  activeBadge: {backgroundColor: '#d1fae5'},
  inactiveBadge: {backgroundColor: '#fef3c7'},
  typeBadgeText: {fontSize: 11, fontWeight: '600', color: '#374151', textTransform: 'capitalize'},
  alertDetail: {fontSize: 13, color: '#374151'},
  alertEmail: {fontSize: 12, color: '#9ca3af', marginTop: 2},
  fab: {position: 'absolute', bottom: 20, right: 20, width: 56, height: 56, borderRadius: 28, backgroundColor: '#6366f1', justifyContent: 'center', alignItems: 'center', elevation: 6},
  fabText: {color: '#fff', fontSize: 28, fontWeight: '300', marginTop: -2},
  modalOverlay: {flex: 1, backgroundColor: 'rgba(0,0,0,0.4)', justifyContent: 'center', padding: 20},
  modal: {backgroundColor: '#fff', borderRadius: 16, padding: 20},
  modalTitle: {fontSize: 20, fontWeight: '700', color: '#111827', marginBottom: 16},
  modalInput: {backgroundColor: '#f3f4f6', borderRadius: 10, paddingHorizontal: 14, paddingVertical: 12, fontSize: 15, marginBottom: 12, color: '#111827'},
  pickerLabel: {fontSize: 13, color: '#6b7280', marginBottom: 6},
  typeRow: {flexDirection: 'row', gap: 8, marginBottom: 12},
  typeOption: {flex: 1, paddingVertical: 10, borderRadius: 10, backgroundColor: '#f3f4f6', alignItems: 'center'},
  typeSelected: {backgroundColor: '#6366f1'},
  typeOptionText: {fontSize: 13, fontWeight: '600', color: '#374151', textTransform: 'capitalize'},
  typeSelectedText: {color: '#fff'},
  modalButtons: {flexDirection: 'row', justifyContent: 'flex-end', gap: 10, marginTop: 8},
  cancelBtn: {paddingHorizontal: 18, paddingVertical: 10, borderRadius: 10, backgroundColor: '#f3f4f6'},
  cancelText: {color: '#374151', fontWeight: '600'},
  confirmBtn: {paddingHorizontal: 18, paddingVertical: 10, borderRadius: 10, backgroundColor: '#6366f1'},
  confirmText: {color: '#fff', fontWeight: '600'},
});
