import React, {useState, useCallback} from 'react';
import {View, Text, TouchableOpacity, StyleSheet, ScrollView, ActivityIndicator, RefreshControl} from 'react-native';
import {useFocusEffect} from '@react-navigation/native';
import {useAuth} from '../auth/AuthContext';
import * as api from '../api/client';

interface Analytics {
  total_predictions: number;
  accurate_predictions: number;
  average_accuracy: number;
  total_portfolio_value: number;
  total_gain_loss: number;
}

export default function ProfileScreen({navigation}: any) {
  const {user, logout} = useAuth();
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchAnalytics = useCallback(async () => {
    try {
      const data = await api.getUserAnalytics();
      setAnalytics(data);
    } catch {
      // silently handle
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      fetchAnalytics();
    }, [fetchAnalytics]),
  );

  return (
    <ScrollView
      style={styles.flex}
      contentContainerStyle={styles.content}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => {setRefreshing(true); fetchAnalytics();}} colors={['#6366f1']} />}>
      {/* User info card */}
      <View style={styles.card}>
        <View style={styles.avatarLarge}>
          <Text style={styles.avatarLargeText}>
            {(user?.first_name?.[0] || '') + (user?.last_name?.[0] || '')}
          </Text>
        </View>
        <Text style={styles.name}>{user?.first_name} {user?.last_name}</Text>
        <Text style={styles.email}>{user?.email}</Text>
        {user?.username && <Text style={styles.username}>@{user.username}</Text>}
      </View>

      {/* Analytics */}
      {loading ? (
        <ActivityIndicator size="large" color="#6366f1" style={{marginTop: 30}} />
      ) : analytics ? (
        <>
          <Text style={styles.sectionTitle}>Analytics</Text>
          <View style={styles.statsGrid}>
            <View style={styles.statCard}>
              <Text style={styles.statValue}>{analytics.total_predictions}</Text>
              <Text style={styles.statLabel}>Predictions</Text>
            </View>
            <View style={styles.statCard}>
              <Text style={styles.statValue}>{analytics.accurate_predictions}</Text>
              <Text style={styles.statLabel}>Accurate</Text>
            </View>
            <View style={styles.statCard}>
              <Text style={styles.statValue}>{(analytics.average_accuracy * 100).toFixed(1)}%</Text>
              <Text style={styles.statLabel}>Accuracy</Text>
            </View>
          </View>

          <View style={styles.card}>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Portfolio Value</Text>
              <Text style={styles.infoValue}>${analytics.total_portfolio_value.toFixed(2)}</Text>
            </View>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Total Gain/Loss</Text>
              <Text style={[styles.infoValue, {color: analytics.total_gain_loss >= 0 ? '#10b981' : '#ef4444'}]}>
                {analytics.total_gain_loss >= 0 ? '+' : ''}${analytics.total_gain_loss.toFixed(2)}
              </Text>
            </View>
          </View>
        </>
      ) : null}

      {/* Change Password */}
      <TouchableOpacity style={styles.changePassBtn} onPress={() => navigation.navigate('ChangePassword')} activeOpacity={0.7}>
        <Text style={styles.changePassText}>Change Password</Text>
      </TouchableOpacity>

      {/* Logout */}
      <TouchableOpacity style={styles.logoutBtn} onPress={logout} activeOpacity={0.7}>
        <Text style={styles.logoutText}>Sign Out</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  flex: {flex: 1, backgroundColor: '#f9fafb'},
  content: {padding: 16, paddingBottom: 40},
  card: {backgroundColor: '#fff', borderRadius: 14, padding: 20, marginBottom: 16, borderWidth: 1, borderColor: '#e5e7eb', alignItems: 'center'},
  avatarLarge: {width: 64, height: 64, borderRadius: 32, backgroundColor: '#6366f1', justifyContent: 'center', alignItems: 'center', marginBottom: 12},
  avatarLargeText: {color: '#fff', fontSize: 22, fontWeight: '700'},
  name: {fontSize: 20, fontWeight: '700', color: '#111827'},
  email: {fontSize: 14, color: '#6b7280', marginTop: 2},
  username: {fontSize: 13, color: '#9ca3af', marginTop: 2},
  sectionTitle: {fontSize: 17, fontWeight: '700', color: '#111827', marginBottom: 12, marginTop: 4},
  statsGrid: {flexDirection: 'row', gap: 10, marginBottom: 16},
  statCard: {flex: 1, backgroundColor: '#fff', borderRadius: 12, padding: 16, alignItems: 'center', borderWidth: 1, borderColor: '#e5e7eb'},
  statValue: {fontSize: 22, fontWeight: '800', color: '#6366f1'},
  statLabel: {fontSize: 11, color: '#6b7280', marginTop: 4},
  infoRow: {flexDirection: 'row', justifyContent: 'space-between', width: '100%', paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: '#f3f4f6'},
  infoLabel: {fontSize: 14, color: '#6b7280'},
  infoValue: {fontSize: 14, fontWeight: '700', color: '#111827'},
  changePassBtn: {backgroundColor: '#eef2ff', borderRadius: 12, paddingVertical: 16, alignItems: 'center', marginTop: 16, borderWidth: 1, borderColor: '#c7d2fe'},
  changePassText: {color: '#6366f1', fontSize: 16, fontWeight: '700'},
  logoutBtn: {backgroundColor: '#fef2f2', borderRadius: 12, paddingVertical: 16, alignItems: 'center', marginTop: 12, borderWidth: 1, borderColor: '#fecaca'},
  logoutText: {color: '#ef4444', fontSize: 16, fontWeight: '700'},
});
